#!/usr/bin/env python3
"""Command-line interface for LoRA instruction fine-tuning."""

import argparse
import json

from configs.lora_config import LoRAConfig
from training.train_instruction_lora_d import train_instruction_model_lora
from configs.training_config_d import TrainingConfig
from accelerate import Accelerator


def main():
    """Fine-tune instruction model with LoRA from command line."""
    
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
    
    # Gradient accumulation.
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients (default: 1)")

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on validation loss improvement ratio")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Number of epochs with insufficient improvement before stopping (default: 3)")
    parser.add_argument("--early-stopping-epsilon", type=float, default=1e-4, help="Minimum loss improvement ratio to continue training (default: 1e-4)")

    # Device argument is kept for CLI compatibility but ignored by Accelerate.
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

    # Initialize Accelerator after parsing args to use gradient_accumulation_steps.
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
        # Create training config from CLI arguments.
        config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            gradient_clip=args.gradient_clip,
            save_every=args.save_every,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping=args.early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_epsilon=args.early_stopping_epsilon,
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

    # Display configuration (main process only).
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
        # Train the model.
        adapter_path = train_instruction_model_lora(
            instruction_corpus_path=args.instruction_corpus_path,
            tokenizer_path=args.tokenizer_path,
            foundational_checkpoint=args.foundational_checkpoint,
            output_dir=args.output_dir,
            config=config,
            lora_config=lora_config,
            model_config=model_config,
            save_merged_model=args.save_merged_model,
            accelerator=accelerator
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


