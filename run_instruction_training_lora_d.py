#!/usr/bin/env python3
"""Command-line interface for LoRA instruction fine-tuning."""

import argparse
import json

from lora_config import LoRAConfig
from train_instruction_lora_d import train_instruction_model_lora
from training_config_d import TrainingConfig
from accelerate import Accelerator


def main():
    """Fine-tune instruction model with LoRA from command line."""
    
    # [新增 2] 初始化 Accelerator
    # 我们先解析部分参数或者允许 accelerator 稍后初始化，
    # 但最简单的方法是先把 parser 定义好，解析完 args 后再初始化 accelerator (如果需要 args 参数)
    # 或者直接在这里初始化 (如果不依赖 args 中的 mixed_precision 等参数)
    # 按照之前的最佳实践，我们在 parse_args 之后初始化，以便传入 gradient_accumulation_steps
    
    parser = argparse.ArgumentParser(
        description="Fine-tune arithmetic LLM with LoRA adapters"
    )

    # Required arguments
    parser.add_argument("--instruction-corpus-path", type=str, required=True, help="Path to instruction-formatted corpus file")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to trained tokenizer directory")
    parser.add_argument("--foundational-checkpoint", type=str, required=True, help="Path to foundational model checkpoint")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model checkpoints")

    # Training configuration
    parser.add_argument("--config", type=str, help="Path to training configuration JSON file")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device (default: 32)")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of fine-tuning epochs (default: 5)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Number of warmup steps (default: 500)")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value (default: 1.0)")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps (default: 500)")
    
    # [新增 3] 梯度累积参数
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients (default: 1)")

    # [修改] Device 参数保留但忽略
    parser.add_argument("--device", type=str, default="auto", help="Ignored when using accelerate")

    # LoRA configuration
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling (default: 16.0)")
    parser.add_argument("--lora-target-modules", type=str, default="attention", help="Comma-separated target modules")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--save-merged-model", action="store_true", help="Save merged model after training")

    # Model configuration
    parser.add_argument("--model-config", type=str, help="Path to model configuration JSON file (optional)")

    args = parser.parse_args()

    # [新增 4] 在解析完参数后初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" # 建议默认开启混合精度，LoRA 非常适合
    )

    # Load or create training configuration
    if args.config:
        if accelerator.is_local_main_process:
            print(f"Loading training configuration from: {args.config}")
        config = TrainingConfig.from_json(args.config)
    else:
        # [修改] 移除 device 检测逻辑，传入 gradient_accumulation_steps
        config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            gradient_clip=args.gradient_clip,
            save_every=args.save_every,
            gradient_accumulation_steps=args.gradient_accumulation_steps
            # device=device <-- 删除
        )

    target_modules = [
        module.strip() for module in args.lora_target_modules.split(",") if module.strip()
    ]
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=target_modules,
        dropout=args.lora_dropout
    )

    # Load model configuration if provided
    model_config = None
    if args.model_config:
        if accelerator.is_local_main_process:
            print(f"Loading model configuration from: {args.model_config}")
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)

    # [修改] Display configuration - 只在主进程打印
    if accelerator.is_local_main_process:
        print("\n" + "=" * 60)
        print("LORA INSTRUCTION FINE-TUNING (Accelerated)")
        print("=" * 60)
        print(f"\nInstruction corpus: {args.instruction_corpus_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Foundational checkpoint: {args.foundational_checkpoint}")
        print(f"Output directory: {args.output_dir}")
        print("\nLoRA Configuration:")
        print(f"  Rank: {lora_config.rank}")
        print(f"  Alpha: {lora_config.alpha}")
        print(f"  Target modules: {', '.join(lora_config.target_modules)}")
        print(f"  Dropout: {lora_config.dropout}")
        print("\nTraining Configuration:")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size (per device): {config.batch_size}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Gradient clip: {config.gradient_clip}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Save every: {config.save_every} steps")
        print("=" * 60 + "\n")

    # Train model
    try:
        # [修改] 传入 accelerator 对象
        adapter_path = train_instruction_model_lora(
            instruction_corpus_path=args.instruction_corpus_path,
            tokenizer_path=args.tokenizer_path,
            foundational_checkpoint=args.foundational_checkpoint,
            output_dir=args.output_dir,
            config=config,
            lora_config=lora_config,
            model_config=model_config,
            save_merged_model=args.save_merged_model,
            accelerator=accelerator # <--- 关键修改
        )

        if accelerator.is_local_main_process:
            print("\n" + "=" * 60)
            print("LORA FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Adapter checkpoint: {adapter_path}")
            print("=" * 60)

    except Exception as e:
        print(f"\n[Process {accelerator.process_index}] LORA FINE-TUNING FAILED!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()


