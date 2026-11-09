"""Training loop shared by the supervised stages.

Foundational pre-training, full instruction fine-tuning and LoRA fine-tuning
differ only in how the model and optimizer are built. Everything after that
(the learning-rate schedule, the epoch loop, validation, early stopping,
checkpointing and exact resume) lives here so it is implemented once.
"""

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm

from configs.training_config import TrainingConfig
from training.common import (
    CHECKPOINT_FORMAT_VERSION,
    check_resume_compatibility,
    create_run_dir,
    gather_rng_states,
    get_linear_schedule_with_warmup,
    legacy_resume_position,
    load_checkpoint,
    path_sha256,
    resolve_resume_dir,
    resolve_warmup_steps,
    restore_process_rng_state,
    save_checkpoint,
    step_scheduler_if_updated,
    write_json,
)


@dataclass
class TrainingState:
    """Everything besides model/optimizer/scheduler needed to resume a run.

    Attributes:
        epoch: 0-based epoch to run next (or the one in progress)
        batches_done: Batches of ``epoch`` already trained
        global_step: Optimizer update attempts (accumulation boundaries)
        skipped_updates: Updates the fp16 GradScaler skipped (overflow)
        epoch_loss_sum: Sum of batch losses of the current epoch, all processes
        epoch_loss_count: Number of batches in ``epoch_loss_sum``
        best_val_loss: Best validation loss so far (None before the first)
        prev_val_loss: Validation loss of the previous epoch
        patience_counter: Epochs in a row without sufficient improvement
        early_stopped: Whether early stopping has ended the run
        training_log: One entry per completed epoch
    """
    epoch: int = 0
    batches_done: int = 0
    global_step: int = 0
    skipped_updates: int = 0
    epoch_loss_sum: float = 0.0
    epoch_loss_count: int = 0
    best_val_loss: Optional[float] = None
    prev_val_loss: Optional[float] = None
    patience_counter: int = 0
    early_stopped: bool = False
    training_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TrainingState":
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def describe_data(
    corpus_path: str,
    tokenizer_path: str,
    mode: str,
    config: TrainingConfig,
    train_dataloader,
    val_dataloader
) -> Dict[str, Any]:
    """Identity of the training data, stored in checkpoints and checked on resume.

    Call before ``accelerator.prepare()``, while the loaders still hold the
    full train/validation subsets.
    """
    return {
        'corpus_sha256': path_sha256(corpus_path),
        'tokenizer_sha256': path_sha256(tokenizer_path),
        'mode': mode,
        'split_seed': config.seed,
        'train_size': len(train_dataloader.dataset),
        'val_size': len(val_dataloader.dataset),
    }


def _reduce_loss_totals(
    accelerator: Accelerator,
    loss_sum: torch.Tensor,
    count: int,
    carry: Tuple[float, int]
) -> Tuple[float, int]:
    """Sum the epoch's batch losses and batch counts over all processes."""
    totals = torch.stack([
        loss_sum.float() + carry[0],
        torch.tensor(float(count + carry[1]), device=loss_sum.device),
    ])
    totals = accelerator.reduce(totals, reduction="sum")
    return totals[0].item(), int(round(totals[1].item()))


def train_epoch(
    model: nn.Module,
    train_dataloader,
    optimizer,
    scheduler,
    config: TrainingConfig,
    accelerator: Accelerator,
    state: TrainingState,
    save_fn: Callable[[str, float], str],
    carry: Tuple[float, int] = (0.0, 0),
    total_batches: Optional[int] = None
) -> float:
    """Train for one epoch (or the rest of one, when resuming).

    Updates ``state.global_step``, ``state.batches_done`` and
    ``state.skipped_updates`` in place, and calls ``save_fn`` every
    ``config.save_every`` updates.

    Args:
        carry: (loss sum, batch count) of this epoch's batches trained before a
            resume. Pass it on the main process only, so the epoch average
            counts those batches exactly once.
        total_batches: Batches in a full epoch, for the progress bar.

    Returns:
        Mean batch loss of the whole epoch over all processes.
    """
    model.train()
    loss_sum = torch.zeros((), device=accelerator.device)
    num_batches = 0

    if accelerator.is_local_main_process:
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {state.epoch + 1}",
            initial=state.batches_done,
            total=total_batches,
        )
    else:
        progress_bar = train_dataloader

    for input_ids, attention_mask, labels in progress_bar:
        # Use accumulate() to wrap forward+backward: gradients are only
        # synchronized across processes when accumulation_steps is reached.
        with accelerator.accumulate(model):
            # Prepare inputs and targets
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            input_attention_mask = attention_mask[:, :-1]

            # Forward pass
            logits = model(inputs, input_attention_mask)

            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

            # Backward pass
            accelerator.backward(loss)

            # Clip gradients only after they have been synchronized.
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)

            # Under accumulate(), step() only takes effect when gradients
            # are synchronized (i.e. accumulation is complete).
            optimizer.step()
            if accelerator.sync_gradients:
                if step_scheduler_if_updated(optimizer, scheduler):
                    state.skipped_updates += 1

            optimizer.zero_grad()

        loss_sum += loss.detach().float()
        num_batches += 1
        state.batches_done += 1

        # global_step counts accumulation boundaries, i.e. update attempts.
        if accelerator.sync_gradients:
            state.global_step += 1

            if state.global_step % config.save_every == 0:
                state.epoch_loss_sum, state.epoch_loss_count = _reduce_loss_totals(
                    accelerator, loss_sum, num_batches, carry
                )
                save_fn(f"checkpoint_step_{state.global_step}.pt", loss.item())

        # Update progress bar.
        if num_batches % 10 == 1 and accelerator.is_local_main_process:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{(loss_sum / num_batches).item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'step': state.global_step
            })

    total_loss, total_count = _reduce_loss_totals(accelerator, loss_sum, num_batches, carry)
    return total_loss / total_count if total_count > 0 else 0.0


def evaluate(
    model: nn.Module,
    val_dataloader,
    accelerator: Accelerator
) -> float:
    """Token-weighted validation loss over the unique validation samples.

    Each sample's summed token loss and token count are gathered with
    ``gather_for_metrics``, which drops the samples a prepared dataloader
    repeats to give every process the same number of batches. The result is
    the mean loss per predicted token, identical on every process.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_dataloader:
            # Prepare inputs and targets
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            input_attention_mask = attention_mask[:, :-1]

            # Forward pass
            logits = model(inputs, input_attention_mask)

            token_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
                reduction='none'
            ).view(targets.shape)
            valid = targets != -100
            sample_loss = (token_loss * valid).sum(dim=1)
            sample_tokens = valid.sum(dim=1)

            sample_loss, sample_tokens = accelerator.gather_for_metrics(
                (sample_loss, sample_tokens)
            )
            total_loss += sample_loss.sum().item()
            total_tokens += int(sample_tokens.sum().item())

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def run_supervised_training(
    stage: str,
    model: nn.Module,
    optimizer,
    train_dataloader,
    val_dataloader,
    config: TrainingConfig,
    accelerator: Accelerator,
    output_parent: str,
    data_info: Dict[str, Any],
    tokenizer_vocab_size: int,
    model_config: Dict[str, Any],
    resume_checkpoint: Optional[str] = None,
    write_run_files: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """Run (or resume) the epoch loop for a supervised stage.

    ``model``, ``optimizer`` and both dataloaders must already have gone
    through ``accelerator.prepare()``. The learning-rate scheduler is created
    here and intentionally *not* prepared (see ``step_scheduler_if_updated``).

    Args:
        stage: Run name prefix and checkpoint tag ('foundational', ...)
        output_parent: Directory in which a new run directory is created
        data_info: Identity of the training data (corpus hash, split sizes,
            seed, ...). A resume against different data is rejected.
        write_run_files: Optional callback writing stage-specific files into
            the run directory (main process only)

    Returns:
        Dict with the run directory, final checkpoint path, final state and
        a base summary that the caller can extend and write.
    """
    accum = accelerator.gradient_accumulation_steps
    batches_per_epoch = len(train_dataloader)
    # Both counts are per process: every process performs the same number of
    # optimizer updates, and the scheduler steps once per update.
    updates_per_epoch = math.ceil(batches_per_epoch / accum)
    total_steps = updates_per_epoch * config.num_epochs
    warmup_steps = resolve_warmup_steps(total_steps, config.warmup_steps, config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    schedule_info = {
        'num_processes': accelerator.num_processes,
        'batch_size_per_process': config.batch_size,
        'gradient_accumulation_steps': accum,
        'batches_per_epoch': batches_per_epoch,
        'updates_per_epoch': updates_per_epoch,
        'num_epochs': config.num_epochs,
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'learning_rate': config.learning_rate,
    }
    run_info = {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'stage': stage,
        'schedule': schedule_info,
        'data': data_info,
        'mixed_precision': accelerator.mixed_precision,
    }

    state = TrainingState()
    rng_states = None

    if resume_checkpoint is not None:
        if accelerator.is_main_process:
            print(f"\nResuming training from checkpoint: {resume_checkpoint}")
        meta = load_checkpoint(
            checkpoint_path=resume_checkpoint,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=accelerator.scaler
        )
        if meta['format_version'] >= 2:
            saved_info = meta['run_info'] or {}
            if saved_info.get('stage') != stage:
                raise ValueError(
                    f"Cannot resume {stage} training from a "
                    f"{saved_info.get('stage')} checkpoint."
                )
            check_resume_compatibility(saved_info['schedule'], schedule_info, "schedule")
            check_resume_compatibility(saved_info['data'], data_info, "training data")
            state = TrainingState.from_dict(meta['training_state'])
            state.epoch = meta['epoch']
            state.batches_done = meta['batches_done']
            state.global_step = meta['step']
            rng_states = meta['rng_states']
        else:
            state.global_step = meta['step']
            state.epoch, state.batches_done = legacy_resume_position(
                state.global_step, updates_per_epoch, accum
            )
            if accelerator.is_main_process:
                print(
                    "Warning: this checkpoint predates exact resume (no format_version). "
                    "Weights, optimizer and scheduler are restored and the position is "
                    "derived from the step count, but the original data order, RNG "
                    "state, early-stopping counters and training log cannot be "
                    "recovered, and the schedule settings cannot be verified."
                )
        output_dir = resolve_resume_dir(
            accelerator, resume_checkpoint, state.global_step,
            force_branch=meta['format_version'] < 2
        )
        if accelerator.is_main_process:
            print(
                f"Resumed at epoch {state.epoch + 1}, batch {state.batches_done} "
                f"of {batches_per_epoch}, global step {state.global_step}"
            )
    else:
        output_dir = create_run_dir(accelerator, output_parent, stage)

    if accelerator.is_main_process:
        print(f"Run directory: {output_dir}")
        print(f"Configuration: {config.to_dict()}")
        print(
            f"Schedule: {batches_per_epoch} batches/epoch per process, "
            f"{updates_per_epoch} updates/epoch, {total_steps} updates in total, "
            f"{warmup_steps} warmup updates, {accelerator.num_processes} process(es)"
        )
        config.to_json(os.path.join(output_dir, 'training_config.json'))
        write_json(os.path.join(output_dir, 'model_config.json'), model_config)
        write_json(os.path.join(output_dir, 'run_info.json'), run_info)
        if write_run_files is not None:
            write_run_files(output_dir)

    def save_fn(filename: str, loss: float) -> str:
        # Collective: every process contributes its RNG state.
        process_rng_states = gather_rng_states(accelerator)
        path = os.path.join(output_dir, filename)
        if accelerator.is_main_process:
            save_checkpoint(
                model=accelerator.unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=state.epoch,
                step=state.global_step,
                loss=loss,
                config=config,
                tokenizer_vocab_size=tokenizer_vocab_size,
                output_dir=output_dir,
                filename=filename,
                extra_state={
                    'batches_done': state.batches_done,
                    'training_state': state.to_dict(),
                    'rng_states': process_rng_states,
                    'scaler_state_dict': (
                        accelerator.scaler.state_dict() if accelerator.scaler is not None else None
                    ),
                    'run_info': run_info,
                }
            )
        accelerator.wait_for_everyone()
        return path

    # Restore each process's RNG last, after all setup that consumes randomness.
    restore_process_rng_state(accelerator, rng_states)

    train_loss = None
    val_loss = None
    if state.early_stopped or state.epoch >= config.num_epochs:
        if accelerator.is_main_process:
            print("\nThis checkpoint has already finished training; nothing left to run.")

    while not state.early_stopped and state.epoch < config.num_epochs:
        epoch = state.epoch
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"{'='*60}")

        # The seedable sampler makes the batch order a function of the epoch,
        # so a resumed epoch sees exactly the batches it would have seen.
        train_dataloader.set_epoch(epoch)
        epoch_loader = train_dataloader
        carry = (0.0, 0)
        if state.batches_done > 0:
            epoch_loader = accelerator.skip_first_batches(train_dataloader, state.batches_done)
            # skip_first_batches() builds a new loader whose epoch counter starts at 0.
            epoch_loader.set_epoch(epoch)
            if accelerator.is_main_process:
                carry = (state.epoch_loss_sum, state.epoch_loss_count)

        train_loss = train_epoch(
            model=model,
            train_dataloader=epoch_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            accelerator=accelerator,
            state=state,
            save_fn=save_fn,
            carry=carry,
            total_batches=batches_per_epoch
        )

        # Evaluate; every process gets the same gathered validation loss.
        if accelerator.is_main_process:
            print("\nEvaluating on validation set...")
        val_loss = evaluate(model, val_dataloader, accelerator)

        # The epoch is complete: checkpoints from here on point at the next one.
        state.epoch = epoch + 1
        state.batches_done = 0
        state.epoch_loss_sum = 0.0
        state.epoch_loss_count = 0

        state.training_log.append({
            'epoch': epoch + 1,
            'step': state.global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0],
            'skipped_updates': state.skipped_updates,
        })

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss (per token): {val_loss:.4f}")

        # Early stopping (every process sees the same val_loss).
        if config.early_stopping and state.prev_val_loss is not None:
            if state.prev_val_loss == 0:
                improvement_ratio = 0.0 if val_loss == 0 else float('inf')
            else:
                improvement_ratio = (state.prev_val_loss - val_loss) / abs(state.prev_val_loss)

            if improvement_ratio < config.early_stopping_epsilon:
                state.patience_counter += 1
                if accelerator.is_main_process:
                    print(f"  Early stopping: insufficient improvement ({improvement_ratio:.6f} < {config.early_stopping_epsilon}), patience {state.patience_counter}/{config.early_stopping_patience}")
                if state.patience_counter >= config.early_stopping_patience:
                    if accelerator.is_main_process:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                    state.early_stopped = True
            else:
                state.patience_counter = 0

        state.prev_val_loss = val_loss

        # Save the best model after the early-stopping update, so the
        # checkpoint carries the counters for the evaluation it is based on.
        if state.best_val_loss is None or val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            best_model_path = save_fn('best_model.pt', val_loss)
            if accelerator.is_main_process:
                print(f"  New best model saved: {best_model_path}")

    # Wait for all processes, then save the final model.
    accelerator.wait_for_everyone()

    if train_loss is None:
        # Nothing was trained in this invocation (resumed a finished run).
        last = state.training_log[-1] if state.training_log else {}
        train_loss = last.get('train_loss')
        val_loss = last.get('val_loss')

    if accelerator.is_main_process:
        print("\nSaving final model...")
    final_checkpoint_path = save_fn('final_model.pt', train_loss)

    summary = {
        'stage': stage,
        'epochs_completed': state.epoch,
        'num_epochs_configured': config.num_epochs,
        'early_stopped': state.early_stopped,
        'total_steps': state.global_step,
        'skipped_updates': state.skipped_updates,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'best_val_loss': state.best_val_loss,
        'model_config': model_config,
        'training_config': config.to_dict(),
        'tokenizer_vocab_size': tokenizer_vocab_size,
        'run_info': run_info,
    }

    if accelerator.is_main_process:
        print(f"Final model saved: {final_checkpoint_path}")
        write_json(os.path.join(output_dir, 'training_log.json'), state.training_log)

    return {
        'output_dir': output_dir,
        'final_checkpoint_path': final_checkpoint_path,
        'state': state,
        'summary': summary,
    }


def write_summary(accelerator: Accelerator, output_dir: str, summary: Dict[str, Any]) -> None:
    """Write training_summary.json on the main process."""
    if accelerator.is_main_process:
        write_json(os.path.join(output_dir, 'training_summary.json'), summary)
