"""Command-line options shared by the supervised training launchers.

Precedence of training settings: explicit command-line flag > ``--config``
JSON file > stage default. To tell "not given" apart from "given with the
default value", every training flag defaults to ``None`` here and the stage
defaults are applied by ``TrainingConfig.resolve``.
"""

import argparse
from typing import Any, Dict

from configs.training_config import MIXED_PRECISION_CHOICES, TrainingConfig


# TrainingConfig fields that have a command-line flag of the same name.
TRAINING_FLAG_FIELDS = (
    "learning_rate",
    "batch_size",
    "num_epochs",
    "warmup_steps",
    "warmup_ratio",
    "gradient_clip",
    "save_every",
    "gradient_accumulation_steps",
    "mixed_precision",
    "seed",
    "early_stopping",
    "early_stopping_patience",
    "early_stopping_epsilon",
)


def add_training_arguments(parser: argparse.ArgumentParser, defaults: Dict[str, Any]) -> None:
    """Add the training flags, showing the stage defaults in the help text."""
    group = parser.add_argument_group("training configuration")
    group.add_argument("--config", type=str, help="Path to training configuration JSON file; explicit flags override its values")
    group.add_argument("--learning-rate", type=float, default=None, help=f"Learning rate (default: {defaults['learning_rate']})")
    group.add_argument("--batch-size", type=int, default=None, help=f"Batch size per device (default: {defaults['batch_size']})")
    group.add_argument("--num-epochs", type=int, default=None, help=f"Number of training epochs (default: {defaults['num_epochs']})")
    group.add_argument("--warmup-steps", type=int, default=None, help="Number of warmup updates; overrides --warmup-ratio when given (default: unset)")
    group.add_argument("--warmup-ratio", type=float, default=None, help="Fraction of all updates used for warmup (default: 0.05)")
    group.add_argument("--gradient-clip", type=float, default=None, help="Gradient clipping value (default: 1.0)")
    group.add_argument("--save-every", type=int, default=None, help=f"Save checkpoint every N optimizer updates (default: {defaults['save_every']})")
    group.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Number of steps to accumulate gradients (default: 1)")
    group.add_argument("--mixed-precision", type=str, default=None, choices=MIXED_PRECISION_CHOICES, help="Mixed precision mode passed to Accelerate (default: fp16)")
    group.add_argument("--seed", type=int, default=None, help="Seed for the data split, data order and initialization (default: 42)")

    # Early stopping
    group.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=None, help="Enable early stopping based on validation loss improvement ratio (default: off)")
    group.add_argument("--early-stopping-patience", type=int, default=None, help="Number of epochs with insufficient improvement before stopping (default: 3)")
    group.add_argument("--early-stopping-epsilon", type=float, default=None, help="Minimum loss improvement ratio to continue training (default: 1e-4)")

    # Resume from checkpoint
    group.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a checkpoint of this stage to resume training from")

    # Device argument is kept for CLI compatibility but ignored by Accelerate.
    group.add_argument("--device", type=str, default="auto", help="Ignored when using accelerate")


def resolve_training_config(args: argparse.Namespace, defaults: Dict[str, Any]) -> TrainingConfig:
    """Merge stage defaults, the optional JSON file and explicit flags."""
    overrides = {name: getattr(args, name) for name in TRAINING_FLAG_FIELDS}
    return TrainingConfig.resolve(defaults, json_path=args.config, overrides=overrides)


def print_training_config(config: TrainingConfig) -> None:
    print("\nTraining Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size (per device): {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    if config.warmup_steps is not None:
        print(f"  Warmup steps: {config.warmup_steps}")
    else:
        print(f"  Warmup ratio: {config.warmup_ratio}")
    print(f"  Gradient clip: {config.gradient_clip}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print(f"  Seed: {config.seed}")
    print(f"  Save every: {config.save_every} updates")
    if config.early_stopping:
        print(f"  Early stopping: patience {config.early_stopping_patience}, epsilon {config.early_stopping_epsilon}")
