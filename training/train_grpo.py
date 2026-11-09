"""Training entry point for GRPO."""

from typing import Any, Dict, Iterator, List, Optional, Tuple
import math
import random

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.data_loader import ArithmeticDataset
from evaluation.evaluator import eval_expression
from data.generator import ExpressionGenerator
from configs.grpo_config import GRPOConfig
from training.common import (
    check_resume_compatibility,
    create_run_dir,
    file_sha256,
    json_sha256,
    legacy_resume_position,
    path_sha256,
    resolve_resume_dir,
    resolve_warmup_steps,
    restore_process_rng_state,
)
from training.grpo_trainer import GRPOTrainer, GRPOTrainingState


def _batch_iter(items: List[dict], batch_size: int) -> Iterator[Tuple[List[str], List[int]]]:
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [entry["prompt"] for entry in batch]
        ground_truth = [entry["ground_truth"] for entry in batch]
        yield prompts, ground_truth


def _load_instruction_pairs(
    instruction_corpus_path: str,
    tokenizer: ArithmeticBPETokenizer,
    validate_expressions: bool
) -> List[dict]:
    dataset = ArithmeticDataset(
        corpus_path=instruction_corpus_path,
        tokenizer=tokenizer,
        mode="instruction"
    )
    return dataset.get_instruction_pairs(validate_expressions=validate_expressions)


def _generate_pairs(
    num_samples: int,
    max_depth: int,
    num_range: Tuple[int, int],
    seed: int
) -> List[dict]:
    # ExpressionGenerator draws from the global `random` module: seed it for
    # this call only, then restore the previous state.
    rng_state = random.getstate()
    random.seed(seed)
    try:
        generator = ExpressionGenerator(
            max_depth=max_depth,
            num_range=num_range,
            invalid_rate=0.0
        )
        pairs = []
        for _ in range(num_samples):
            expression = generator.generate()
            result = eval_expression(expression)
            if result["answer"] == "ERROR":
                continue
            pairs.append({
                "prompt": result["problem"] + " <think>",
                "ground_truth": int(result["answer"])
            })
    finally:
        random.setstate(rng_state)
    return pairs


def split_and_shard_pairs(
    pairs: List[dict],
    val_size: int,
    seed: int,
    num_processes: int,
    process_index: int
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """Shuffle, hold out validation prompts, and give each process an equal share.

    Every process must run the same number of batches, otherwise one process
    waits forever in a DDP all-reduce (training) or in the reward reduction
    (validation). Each partition is therefore trimmed to a multiple of
    ``num_processes`` *after* the split.

    Returns:
        (local training pairs, local validation pairs, sizes for logging)
    """
    if val_size < 0:
        raise ValueError(f"val_size must be non-negative, got {val_size}")
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)

    if val_size > 0 and val_size < num_processes:
        raise ValueError(
            f"val_size ({val_size}) must be at least the number of processes "
            f"({num_processes}) so every process gets validation prompts"
        )
    if len(pairs) - val_size < num_processes:
        raise ValueError(
            f"Not enough prompts: {len(pairs)} available, {val_size} held out for "
            f"validation, but at least {num_processes} training prompts are needed "
            f"(one per process). Lower --val-size or provide more data."
        )

    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    train_pairs = train_pairs[:len(train_pairs) - len(train_pairs) % num_processes]
    val_pairs = val_pairs[:len(val_pairs) - len(val_pairs) % num_processes]

    sizes = {
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "dropped_for_equal_shares": len(pairs) - len(train_pairs) - len(val_pairs),
    }
    # Strided sharding, like DistributedSampler.
    local_train = train_pairs[process_index::num_processes]
    local_val = val_pairs[process_index::num_processes]
    return local_train, local_val, sizes


def train_grpo_model(
    instruction_corpus_path: Optional[str],
    tokenizer_path: str,
    sft_checkpoint_path: str,
    output_dir: str,
    config: GRPOConfig,
    data_mode: str = "instruction",
    num_samples: int = 1000,
    max_depth: int = 5,
    num_range: Tuple[int, int] = (1, 20),
    filter_invalid_instruction: bool = True,
    candidate_sub_batch_size: Optional[int] = None,
    accelerator: Accelerator = None,
    resume_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Train GRPO model using instruction corpus or generated data."""

    # Require an Accelerator instance for distributed training.
    if accelerator is None:
        raise ValueError("Accelerator must be passed from main script!")

    config.validate()

    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)

    # Initialize the GRPO trainer. Both the policy and the reference model
    # start from the SFT checkpoint; the reference never changes afterwards.
    trainer = GRPOTrainer(
        config=config,
        sft_checkpoint_path=sft_checkpoint_path,
        tokenizer=tokenizer,
        candidate_sub_batch_size=candidate_sub_batch_size,
        accelerator=accelerator
    )

    # Build the prompt list on the main process and send it to the others,
    # so every process works from exactly the same list.
    shared = [None]
    if accelerator.is_main_process:
        if data_mode == "instruction":
            if instruction_corpus_path is None:
                raise ValueError("instruction_corpus_path required for instruction mode")
            shared[0] = _load_instruction_pairs(
                instruction_corpus_path,
                tokenizer,
                filter_invalid_instruction
            )
        elif data_mode == "generated":
            shared[0] = _generate_pairs(num_samples, max_depth, num_range, config.seed)
        else:
            raise ValueError("data_mode must be 'instruction' or 'generated'")
    broadcast_object_list(shared)
    pairs = shared[0]

    if not pairs:
        raise ValueError("No training pairs available")

    train_pairs, val_pairs, sizes = split_and_shard_pairs(
        pairs,
        val_size=config.val_size,
        seed=config.seed,
        num_processes=accelerator.num_processes,
        process_index=accelerator.process_index
    )

    if accelerator.is_main_process:
        print(f"Total data size: {sizes['total_pairs']}")
        print(f"Training prompts: {sizes['train_pairs']} ({len(train_pairs)} per process)")
        print(f"Validation prompts: {sizes['val_pairs']} ({len(val_pairs)} per process)")
        if sizes["dropped_for_equal_shares"]:
            print(f"Dropped {sizes['dropped_for_equal_shares']} prompt(s) so every process gets an equal share")

    # Build local batch lists.
    train_batches = list(_batch_iter(train_pairs, config.batch_size))
    val_batches = list(_batch_iter(val_pairs, config.batch_size))

    # Optimizer updates per process: the last, shorter accumulation group of
    # an epoch also produces an update.
    accum = config.gradient_accumulation_steps
    updates_per_epoch = math.ceil(len(train_batches) / accum)
    total_steps = updates_per_epoch * config.num_epochs
    warmup_steps = resolve_warmup_steps(total_steps, config.warmup_steps, config.warmup_ratio)
    trainer.setup_optimizer_and_scheduler(total_steps=total_steps, warmup_steps=warmup_steps)

    schedule_info = {
        "num_processes": accelerator.num_processes,
        "batch_size_per_process": config.batch_size,
        "gradient_accumulation_steps": accum,
        "batches_per_epoch": len(train_batches),
        "updates_per_epoch": updates_per_epoch,
        "num_epochs": config.num_epochs,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "learning_rate": config.learning_rate,
    }
    data_info = {
        "data_mode": data_mode,
        "pairs_sha256": json_sha256(pairs),
        "tokenizer_sha256": path_sha256(tokenizer_path),
        "seed": config.seed,
        **sizes,
    }
    sft_info = {
        "path": sft_checkpoint_path,
        "sha256": file_sha256(sft_checkpoint_path),
    }
    run_info = {
        "stage": "grpo",
        "schedule": schedule_info,
        "data": data_info,
        "sft_checkpoint": sft_info,
        "mixed_precision": accelerator.mixed_precision,
    }

    if accelerator.is_main_process:
        print(
            f"Schedule: {len(train_batches)} batches/epoch per process, "
            f"{updates_per_epoch} updates/epoch, {total_steps} updates in total, "
            f"{warmup_steps} warmup updates"
        )

    # Resume from checkpoint if provided.
    state = GRPOTrainingState()
    rng_states = None
    if resume_checkpoint is not None:
        if accelerator.is_main_process:
            print(f"Resuming GRPO training from checkpoint: {resume_checkpoint}")
        meta = trainer.load_checkpoint(resume_checkpoint)
        if meta["format_version"] >= 2:
            saved_info = meta["run_info"] or {}
            if saved_info.get("stage") != "grpo":
                raise ValueError(f"Not a GRPO checkpoint: {resume_checkpoint}")
            # The reference model is loaded from --sft-checkpoint; it must be
            # the same SFT model the interrupted run used.
            check_resume_compatibility(saved_info["sft_checkpoint"], sft_info, "SFT checkpoint")
            check_resume_compatibility(saved_info["schedule"], schedule_info, "schedule")
            check_resume_compatibility(saved_info["data"], data_info, "training data")
            state = GRPOTrainingState.from_dict(meta["training_state"])
            state.epoch = meta["epoch"]
            state.batches_done = meta["batches_done"]
            state.global_step = meta["step"]
            rng_states = meta["rng_states"]
        else:
            state.global_step = meta["step"]
            state.epoch, state.batches_done = legacy_resume_position(
                state.global_step, updates_per_epoch, accum
            )
            if accelerator.is_main_process:
                print(
                    "Warning: this checkpoint predates exact resume (no format_version). "
                    "Weights, optimizer and scheduler are restored and the position is "
                    "derived from the step count; the data order, RNG and "
                    "early-stopping state cannot be recovered, and the SFT reference "
                    "model cannot be verified against the original run."
                )
        output_dir = resolve_resume_dir(
            accelerator, resume_checkpoint, state.global_step,
            force_branch=meta["format_version"] < 2
        )
        if accelerator.is_main_process:
            print(
                f"Resumed at epoch {state.epoch + 1}, batch {state.batches_done} of "
                f"{len(train_batches)}, global step {state.global_step}"
            )
    else:
        output_dir = create_run_dir(accelerator, output_dir, "grpo")

    if accelerator.is_main_process:
        print(f"GRPO output directory: {output_dir}")

    # Restore each process's RNG last, after all setup that consumes randomness.
    restore_process_rng_state(accelerator, rng_states)

    # Start training.
    return trainer.train(
        train_batches,
        val_dataloader=val_batches or None,
        output_dir=output_dir,
        state=state,
        run_info=run_info
    )
