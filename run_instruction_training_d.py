#!/usr/bin/env python3
"""Command-line interface for instruction fine-tuning."""

import argparse
import json
from train_instruction_d import train_instruction_model
from training_config_d import TrainingConfig
from accelerate import Accelerator


def main():
    """Fine-tune instruction model from command line."""

    parser = argparse.ArgumentParser(
        description="Fine-tune arithmetic LLM with instruction formatting"
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
    
    # [新增] 梯度累积参数
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients (default: 1)")
    
    # [修改] Device 参数保留但忽略
    parser.add_argument("--device", type=str, default="auto", help="Ignored when using accelerate")
    
    # Model configuration
    parser.add_argument("--model-config", type=str, help="Path to model configuration JSON file")
    
    args = parser.parse_args()

    # [modified] Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16"
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
        print("INSTRUCTION FINE-TUNING (Accelerated)")
        print("=" * 60)
        print(f"\nInstruction corpus: {args.instruction_corpus_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Foundational checkpoint: {args.foundational_checkpoint}")
        print(f"Output directory: {args.output_dir}")
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
        final_checkpoint = train_instruction_model(
            instruction_corpus_path=args.instruction_corpus_path,
            tokenizer_path=args.tokenizer_path,
            foundational_checkpoint=args.foundational_checkpoint,
            output_dir=args.output_dir,
            config=config,
            model_config=model_config,
            accelerator=accelerator
        )
        
        if accelerator.is_local_main_process:
            print("\n" + "=" * 60)
            print("FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Final checkpoint: {final_checkpoint}")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n[Process {accelerator.process_index}] FINE-TUNING FAILED!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()


