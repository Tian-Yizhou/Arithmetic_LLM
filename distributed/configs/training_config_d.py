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
    batch_size: int = 32  # 注意：这是单卡 batch size
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1  # [新增] 默认为 1
    save_every: int = 1000
    eval_every: int = 500
    mixed_precision: str = "no" # [新增] 用于显式控制精度，可选

    # [修改] device 字段不再作为核心配置，或者设为 Optional
    # 因为由 Accelerator 决定，这里只作为只读属性或不需要设置
    # device: str = ... (建议删除，或设为 None)

    # Early stopping configuration
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_epsilon: float = 1e-4

    lora_config: Optional['LoRAConfig'] = None # 假设 LoRAConfig 已定义
    
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
            
        # [新增] 验证 accumulation steps
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

        # [修改] 移除了关于 device 的验证逻辑
        # 因为在 accelerate launch 启动时，某些环境变量可能让 torch.cuda.is_available() 行为发生变化，
        # 且我们不再依赖 config.device 来做 .to() 操作。

        if self.lora_config is not None:
            # 假设 LoRAConfig 也有 validate 方法
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
        
        # 处理嵌套的 lora_config
        if "lora_config" in config_dict and config_dict["lora_config"] is not None:
            # 确保 LoRAConfig 类在作用域内可用
            # lora_config = LoRAConfig(**config_dict["lora_config"]) 
            # 这里保持你原有的逻辑，假设 LoRAConfig 可用
            pass 

        # 过滤掉 JSON 中可能存在的旧字段 (比如 'device')，防止报错
        # 这样即使旧的 json 文件里有 "device": "cuda"，也不会因为 __init__ 不接受而崩溃
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        config = cls(**filtered_dict)
        config.validate()
        return config

    # to_json 和 to_dict 方法通常不需要大改，保持原样即可
    def to_dict(self) -> dict:
        config_dict: Dict[str, Any] = asdict(self)
        # 处理 lora_config 的序列化
        if self.lora_config is not None and hasattr(self.lora_config, 'to_dict'):
             config_dict["lora_config"] = self.lora_config.to_dict()
        elif self.lora_config is not None:
             config_dict["lora_config"] = asdict(self.lora_config)
        return config_dict

    def to_json(self, json_path: str) -> None:
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


