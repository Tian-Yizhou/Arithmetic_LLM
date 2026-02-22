"""Training configuration module for arithmetic LLM training."""

import json
import torch
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

from configs.lora_config import LoRAConfig


@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        learning_rate: Learning rate for optimizer
        batch_size: Batch size per device (not global batch size!)
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        gradient_clip: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        save_every: Save checkpoint every N steps
        eval_every: Evaluate model every N steps
        mixed_precision: 'no', 'fp16', or 'bf16' (optional, can be overridden by accelerate config)
        lora_config: Optional LoRA configuration
    """
    
    learning_rate: float = 1e-4
    batch_size: int = 32  # Note: this is per-device batch size
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    save_every: int = 1000
    eval_every: int = 500
    mixed_precision: str = "no"

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
        
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        
        if self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be positive, got {self.gradient_clip}")
            
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        if self.save_every <= 0:
            raise ValueError(f"save_every must be positive, got {self.save_every}")
        
        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {self.eval_every}")

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
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Handle nested lora_config deserialization
        if "lora_config" in config_dict and config_dict["lora_config"] is not None:
            pass

        # Filter out deprecated fields (e.g. 'device') from older JSON configs
        # to prevent TypeError on __init__
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        config = cls(**filtered_dict)
        config.validate()
        return config

    def to_dict(self) -> dict:
        config_dict: Dict[str, Any] = asdict(self)
        # Serialize lora_config using its own to_dict if available
        if self.lora_config is not None and hasattr(self.lora_config, 'to_dict'):
             config_dict["lora_config"] = self.lora_config.to_dict()
        elif self.lora_config is not None:
             config_dict["lora_config"] = asdict(self.lora_config)
        return config_dict

    def to_json(self, json_path: str) -> None:
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


