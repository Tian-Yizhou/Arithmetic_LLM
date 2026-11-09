"""Training configuration module for arithmetic LLM training."""

import json
from dataclasses import dataclass, asdict, fields
from typing import Optional, Any, Dict

from configs.lora_config import LoRAConfig


MIXED_PRECISION_CHOICES = ("no", "fp16", "bf16")


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        learning_rate: Learning rate for optimizer
        batch_size: Batch size per device (not global batch size!)
        num_epochs: Number of training epochs
        warmup_steps: Explicit number of warmup updates. None means "use
            warmup_ratio"; an explicit value (including 0) overrides the ratio.
        warmup_ratio: Fraction of all optimizer updates used for warmup when
            warmup_steps is None
        gradient_clip: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        save_every: Save checkpoint every N optimizer updates
        eval_every: Kept for config compatibility; supervised stages evaluate
            once per epoch
        mixed_precision: 'no', 'fp16', or 'bf16'; passed to the Accelerator
        seed: Seed for the train/validation split, the data order and
            parameter initialization
        lora_config: Optional LoRA configuration
    """

    learning_rate: float = 1e-4
    batch_size: int = 32  # Note: this is per-device batch size
    num_epochs: int = 10
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    save_every: int = 1000
    eval_every: int = 500
    mixed_precision: str = "fp16"
    seed: int = 42

    # Device field removed: Accelerator handles device placement automatically.

    # Early stopping configuration
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_epsilon: float = 1e-4

    lora_config: Optional['LoRAConfig'] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative or None, got {self.warmup_steps}")

        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")

        if self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be positive, got {self.gradient_clip}")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")

        if self.save_every <= 0:
            raise ValueError(f"save_every must be positive, got {self.save_every}")

        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {self.eval_every}")

        if self.mixed_precision not in MIXED_PRECISION_CHOICES:
            raise ValueError(
                f"mixed_precision must be one of {MIXED_PRECISION_CHOICES}, got {self.mixed_precision}"
            )

        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.early_stopping_patience}")

        if self.early_stopping_epsilon <= 0:
            raise ValueError(f"early_stopping_epsilon must be positive, got {self.early_stopping_epsilon}")

        # Device validation removed: Accelerator manages device placement,
        # so config.device is no longer used for .to() calls.

        if self.lora_config is not None:
            if hasattr(self.lora_config, 'validate'):
                self.lora_config.validate()

    @classmethod
    def _filter_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Keep known fields only and rebuild the nested LoRA configuration."""
        # Filter out deprecated fields (e.g. 'device') from older JSON configs
        # to prevent TypeError on __init__
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}

        # Handle nested lora_config deserialization
        if isinstance(filtered.get("lora_config"), dict):
            filtered["lora_config"] = LoRAConfig(**filtered["lora_config"])
        return filtered

    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        config = cls(**cls._filter_dict(cls._read_json(json_path)))
        config.validate()
        return config

    @classmethod
    def resolve(
        cls,
        defaults: Dict[str, Any],
        json_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> 'TrainingConfig':
        """Build a config with the precedence: CLI overrides > JSON file > defaults.

        Args:
            defaults: Stage-specific default values
            json_path: Optional JSON config file (values replace the defaults)
            overrides: Values given explicitly on the command line. Entries
                whose value is None were not given and are ignored.
        """
        values = cls._filter_dict(dict(defaults))
        if json_path:
            values.update(cls._filter_dict(cls._read_json(json_path)))
        if overrides:
            values.update({k: v for k, v in overrides.items() if v is not None})
        config = cls(**values)
        config.validate()
        return config

    @staticmethod
    def _read_json(json_path: str) -> Dict[str, Any]:
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def to_dict(self) -> dict:
        config_dict: Dict[str, Any] = asdict(self)
        # asdict() already converts a LoRAConfig dataclass into a plain dict.
        if self.lora_config is not None and hasattr(self.lora_config, 'to_dict'):
            config_dict["lora_config"] = self.lora_config.to_dict()
        return config_dict

    def to_json(self, json_path: str) -> None:
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
