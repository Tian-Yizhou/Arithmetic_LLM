"""Utilities shared by all training stages.

This module holds everything that the supervised stages and GRPO need to agree
on: how the Accelerator is built, how the learning-rate schedule is sized, how
checkpoints are written and read, and how a run is resumed exactly.

Checkpoint format
-----------------
Checkpoints written by this version carry ``format_version = 2``:

- ``epoch``: the 0-based epoch to continue from. For an end-of-epoch
  checkpoint this is the number of completed epochs.
- ``batches_done``: dataloader batches of ``epoch`` that are already trained
  (0 for end-of-epoch checkpoints).
- ``step``: optimizer update attempts so far (gradient-accumulation
  boundaries, including fp16 updates that the GradScaler skipped).
- ``training_state``: early-stopping counters, best metric, training log and
  the running loss of the current epoch.
- ``rng_states``: one RNG snapshot per process.
- ``scaler_state_dict``: the fp16 GradScaler state (scale, growth tracker).
- ``run_info``: the resolved schedule and data identity, checked on resume.

Older checkpoints (no ``format_version``) are still loadable, but their
``epoch`` field has a different meaning and they lack the extra state; see
``legacy_resume_position``.

All values are tensors or plain Python types so that ``torch.load`` with its
default ``weights_only=True`` can read them.
"""

import contextlib
import hashlib
import json
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedType,
    broadcast_object_list,
    gather_object,
    set_seed,
)


CHECKPOINT_FORMAT_VERSION = 2

# Checkpointing here unwraps the model and saves one full state dict from the
# main process. That is correct for single-device and DDP runs only.
SUPPORTED_DISTRIBUTED_TYPES = (
    DistributedType.NO,
    DistributedType.MULTI_GPU,
    DistributedType.MULTI_CPU,
)


# ---------------------------------------------------------------------------
# Accelerator and learning-rate schedule
# ---------------------------------------------------------------------------

def build_accelerator(
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 1,
    seed: int = 42
) -> Accelerator:
    """Create the single Accelerator used by a training run.

    - The seedable sampler makes each epoch's batch order a function of
      ``seed`` and the epoch number on every process, which exact resume
      relies on.
    - ``set_seed(..., device_specific=True)`` gives each process its own
      dropout/sampling stream. Initial weights still match because DDP
      broadcasts rank 0's parameters when the model is wrapped.
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=seed,
        ),
    )
    if accelerator.distributed_type not in SUPPORTED_DISTRIBUTED_TYPES:
        raise NotImplementedError(
            f"Distributed type {accelerator.distributed_type} is not supported: "
            "checkpointing in this project assumes single-device or DDP training "
            "(FSDP/DeepSpeed need sharded state handling)."
        )
    set_seed(seed, device_specific=True)
    return accelerator


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with linear warmup and decay.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch number for resuming training

    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def resolve_warmup_steps(
    total_steps: int,
    warmup_steps: Optional[int],
    warmup_ratio: float
) -> int:
    """Return the number of warmup updates actually used.

    An explicit ``warmup_steps`` (including 0) wins; otherwise
    ``int(total_steps * warmup_ratio)``. A warmup longer than the whole
    schedule is rejected, because the learning rate would never reach its
    peak value.
    """
    if warmup_steps is None:
        resolved = int(total_steps * warmup_ratio)
    else:
        resolved = int(warmup_steps)
    if resolved < 0:
        raise ValueError(f"warmup steps must be non-negative, got {resolved}")
    if resolved > total_steps:
        raise ValueError(
            f"warmup_steps ({resolved}) exceeds the total number of optimizer "
            f"updates ({total_steps}). Lower --warmup-steps or use --warmup-ratio."
        )
    return resolved


def step_scheduler_if_updated(optimizer, scheduler) -> bool:
    """Advance the scheduler only if the optimizer really updated the weights.

    Under fp16 the GradScaler skips the update when gradients overflow. The
    scheduler is deliberately *not* passed to ``accelerator.prepare()``: a
    prepared scheduler steps ``num_processes`` times per call, while our
    schedules are sized in real optimizer updates. So the skip check that the
    Accelerate wrapper would do has to happen here.

    Returns:
        True if the update was skipped.
    """
    if getattr(optimizer, "step_was_skipped", False):
        return True
    scheduler.step()
    return False


# ---------------------------------------------------------------------------
# Random number generator state
# ---------------------------------------------------------------------------

def capture_rng_state() -> Dict[str, Any]:
    """Snapshot the Python, NumPy and torch RNGs of this process.

    NumPy's state is converted to a tensor plus primitives so the checkpoint
    stays loadable with ``torch.load(weights_only=True)``.
    """
    np_state = np.random.get_state()
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": np_state[0],
            "keys": torch.from_numpy(np_state[1].astype(np.int64)),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore a snapshot made by ``capture_rng_state``."""
    py_state = state["python"]
    random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
    np_state = state["numpy"]
    np.random.set_state((
        np_state["algorithm"],
        np_state["keys"].numpy().astype(np.uint32),
        np_state["pos"],
        np_state["has_gauss"],
        np_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def gather_rng_states(accelerator: Accelerator) -> List[Dict[str, Any]]:
    """Collect every process's RNG snapshot (collective: call on all ranks)."""
    return gather_object([capture_rng_state()])


def restore_process_rng_state(
    accelerator: Accelerator,
    rng_states: Optional[List[Dict[str, Any]]]
) -> None:
    """Give each process back its own RNG snapshot from a checkpoint."""
    if not rng_states:
        return
    if len(rng_states) != accelerator.num_processes:
        if accelerator.is_main_process:
            print(
                f"Warning: checkpoint holds RNG state for {len(rng_states)} "
                f"process(es) but this run has {accelerator.num_processes}; "
                "RNG state not restored."
            )
        return
    restore_rng_state(rng_states[accelerator.process_index])


@contextlib.contextmanager
def isolated_rng(seed: int):
    """Run a block with a fixed seed, then put the previous RNG state back.

    Used for validation sampling so that (a) the metric is reproducible and
    (b) evaluating does not change the random stream training continues with.
    """
    saved = capture_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        restore_rng_state(saved)


# ---------------------------------------------------------------------------
# Identity of data and checkpoints
# ---------------------------------------------------------------------------

def file_sha256(path: str) -> str:
    """SHA-256 of a file, used to recognise a corpus or checkpoint on resume."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: str) -> str:
    """SHA-256 of a file, or of all files under a directory (sorted by name)."""
    if os.path.isfile(path):
        return file_sha256(path)
    digest = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            full = os.path.join(root, name)
            digest.update(os.path.relpath(full, path).encode())
            digest.update(file_sha256(full).encode())
    return digest.hexdigest()


def json_sha256(obj: Any) -> str:
    """SHA-256 of a JSON-serialisable object (e.g. a list of prompt pairs)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def check_resume_compatibility(
    saved: Dict[str, Any],
    current: Dict[str, Any],
    what: str
) -> None:
    """Raise if a resumed run would not continue the checkpointed run.

    ``saved`` and ``current`` are flat dicts (e.g. the resolved schedule, or
    the data identity). Any differing key means the restored scheduler or the
    restored data position would no longer line up with this run.
    """
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, now={current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"Cannot resume: the {what} differs from the checkpointed run.\n  "
            + "\n  ".join(mismatches)
            + "\nResume with the same settings, or start a new run from this "
            "checkpoint's weights instead."
        )


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

def create_run_dir(accelerator: Accelerator, parent: str, prefix: str) -> str:
    """Create ``parent/prefix_<timestamp>`` on the main process; share the path."""
    path = [None]
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path[0] = os.path.join(parent, f"{prefix}_{timestamp}")
        os.makedirs(path[0], exist_ok=True)
    broadcast_object_list(path)
    return path[0]


_STEP_FILE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def resolve_resume_dir(
    accelerator: Accelerator,
    checkpoint_path: str,
    step: int,
    force_branch: bool = False
) -> str:
    """Pick the directory a resumed run writes to.

    A resumed run normally continues in the checkpoint's own run directory, so
    its logs and later checkpoints sit next to the earlier ones. If that
    directory already holds *later* progress (a higher-step checkpoint, or a
    final model when resuming from something else), resuming would branch
    from an older state and overwrite newer files; in that case a new sibling
    directory is created instead. ``force_branch`` always creates one (used
    for legacy checkpoints, whose best-model bookkeeping cannot be restored).
    """
    path = [None]
    if accelerator.is_main_process:
        run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        later_steps = [
            int(m.group(1))
            for name in os.listdir(run_dir)
            if (m := _STEP_FILE.match(name)) and int(m.group(1)) > step
        ]
        finished = (
            os.path.exists(os.path.join(run_dir, "final_model.pt"))
            and os.path.basename(checkpoint_path) != "final_model.pt"
        )
        if later_steps or finished or force_branch:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path[0] = f"{run_dir}_resumed_step{step}_{timestamp}"
            os.makedirs(path[0], exist_ok=True)
            print(f"Note: writing the resumed run to a new directory, {path[0]}, "
                  f"so that files of {run_dir} are not overwritten")
        else:
            path[0] = run_dir
    broadcast_object_list(path)
    return path[0]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def atomic_torch_save(obj: Any, path: str) -> None:
    """Write via a temporary file and ``os.replace`` so a crash never leaves a
    half-written checkpoint under the final name."""
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def describe_model(model: torch.nn.Module) -> Dict[str, Any]:
    """Architecture hyper-parameters needed to rebuild an ArithmeticTransformer."""
    return {
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'nhead': model.nhead,
        'num_layers': model.num_layers,
        'dim_feedforward': model.dim_feedforward,
        'dropout': model.dropout,
        'max_seq_length': model.max_seq_length,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    step: int,
    loss: float,
    config,
    tokenizer_vocab_size: int,
    output_dir: str,
    is_final: bool = False,
    filename: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None
) -> str:
    """Save model checkpoint with metadata (call on the main process only).

    Args:
        model: Unwrapped model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: 0-based epoch to continue from (see module docstring)
        step: Current training step
        loss: Current training or validation loss
        config: Training configuration
        tokenizer_vocab_size: Size of tokenizer vocabulary
        output_dir: Directory to save checkpoint
        is_final: Whether this is the final checkpoint
        filename: Explicit file name. Defaults to ``final_model.pt`` or
            ``checkpoint_step_{step}.pt``. Giving the best model its own name
            avoids clobbering a periodic checkpoint of the same step.
        extra_state: Additional resume state merged into the checkpoint

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': config.to_dict(),
        'model_config': describe_model(model),
        'tokenizer_vocab_size': tokenizer_vocab_size,
    }
    if extra_state:
        checkpoint.update(extra_state)

    if filename is None:
        filename = 'final_model.pt' if is_final else f'checkpoint_step_{step}.pt'
    checkpoint_path = os.path.join(output_dir, filename)
    atomic_torch_save(checkpoint, checkpoint_path)

    return checkpoint_path


def _merge_lora_state_dict(state_dict: dict, checkpoint_config: dict) -> dict:
    """Merge LoRA weights into base weights, producing a plain state dict.

    When a checkpoint was saved from a LoRA-adapted model, the state dict
    contains keys like 'layers.0.self_attention.q_proj.base_layer.weight'
    and 'layers.0.self_attention.q_proj.lora_A/lora_B'. This function
    merges them into plain keys like 'layers.0.self_attention.q_proj.weight'
    so the checkpoint can be loaded into a plain ArithmeticTransformer.
    """
    lora_config = checkpoint_config.get('lora_config', {})
    if isinstance(lora_config, dict):
        alpha = lora_config.get('alpha', 16.0)
        rank = lora_config.get('rank', 8)
    else:
        alpha = 16.0
        rank = 8
    scaling = alpha / rank

    merged = {}
    processed_prefixes = set()

    for key in state_dict:
        if '.base_layer.' in key:
            prefix, suffix = key.split('.base_layer.')
            plain_key = f"{prefix}.{suffix}"

            if suffix == 'weight' and prefix not in processed_prefixes:
                lora_a_key = f"{prefix}.lora_A"
                lora_b_key = f"{prefix}.lora_B"
                base_weight = state_dict[key]
                if lora_a_key in state_dict and lora_b_key in state_dict:
                    lora_a = state_dict[lora_a_key]
                    lora_b = state_dict[lora_b_key]
                    merged[plain_key] = base_weight + (lora_b @ lora_a) * scaling
                else:
                    merged[plain_key] = base_weight
                processed_prefixes.add(prefix)
            else:
                merged[plain_key] = state_dict[key]
        elif key.endswith('.lora_A') or key.endswith('.lora_B'):
            continue
        else:
            merged[key] = state_dict[key]

    return merged


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    scaler: Optional[Any] = None
) -> Dict:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        scaler: Optional fp16 GradScaler to load state into

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']

    # Detect LoRA checkpoint being loaded into a plain model.
    has_lora_keys = any('.base_layer.' in k for k in state_dict)
    model_expects_lora = any('.base_layer.' in k for k in model.state_dict())

    if has_lora_keys and not model_expects_lora:
        state_dict = _merge_lora_state_dict(state_dict, checkpoint.get('config', {}))

    model.load_state_dict(state_dict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            raise ValueError(
                f"The optimizer state in {checkpoint_path} was created for a "
                "different set of trainable parameters (for example a LoRA "
                "checkpoint resumed as full fine-tuning). Its weights can be "
                "used as a starting checkpoint, but training cannot be resumed "
                f"from it. Original error: {e}"
            ) from e

    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return {
        'format_version': checkpoint.get('format_version', 1),
        'epoch': checkpoint.get('epoch', 0),
        'batches_done': checkpoint.get('batches_done', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0.0),
        'config': checkpoint.get('config', {}),
        'model_config': checkpoint.get('model_config'),
        'tokenizer_vocab_size': checkpoint.get('tokenizer_vocab_size', 0),
        'training_state': checkpoint.get('training_state'),
        'rng_states': checkpoint.get('rng_states'),
        'run_info': checkpoint.get('run_info'),
    }


def legacy_resume_position(step: int, updates_per_epoch: int, accumulation_steps: int):
    """Derive (epoch, batches_done) for a checkpoint without ``format_version``.

    Old checkpoints stored the 1-based epoch in progress for periodic saves
    and the 1-based finished epoch for best/final saves, so ``epoch`` cannot be
    trusted. The update count is unambiguous: it gives the position assuming
    the same batch size, accumulation and number of processes. The old data
    order was unseeded, so the exact remaining batches cannot be recovered.
    """
    epoch, updates_into_epoch = divmod(step, max(1, updates_per_epoch))
    return epoch, updates_into_epoch * accumulation_steps


def write_json(path: str, obj: Any) -> None:
    """Write JSON atomically (main process only)."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)
