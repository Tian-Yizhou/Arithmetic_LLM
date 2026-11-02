#!/usr/bin/env python3
"""Command-line interface for LoRA instruction fine-tuning."""

import argparse
import json
from dataclasses import asdict

from configs.cli import add_training_arguments, print_training_config, resolve_training_config
from configs.lora_config import LoRAConfig
from training.common import build_accelerator
from training.train_instruction_lora import train_instruction_model_lora


# Stage defaults; a --config JSON file and explicit flags override them.
DEFAULTS = {
    "learning_rate": 5e-5,
    "batch_size": 32,
    "num_epochs": 5,
    "save_every": 500,
}


def resolve_lora_config(args: argparse.Namespace, json_lora: LoRAConfig = None) -> LoRAConfig:
    """LoRA settings with the same precedence: flag > JSON lora_config > default."""
    values = asdict(json_lora) if json_lora is not None else asdict(LoRAConfig())
    if args.lora_rank is not None:
        values["rank"] = args.lora_rank
    if args.lora_alpha is not None:
        values["alpha"] = args.lora_alpha
    if args.lora_target_modules is not None:
        values["target_modules"] = [
            module.strip() for module in args.lora_target_modules.split(",") if module.strip()
        ]
    if args.lora_dropout is not None:
        values["dropout"] = args.lora_dropout
    lora_config = LoRAConfig(**values)
    lora_config.validate()
    return lora_config


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
    add_training_arguments(parser, DEFAULTS)

    # LoRA configuration
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha scaling (default: 16.0)")
    parser.add_argument("--lora-target-modules", type=str, default=None, help="Comma-separated target modules (default: attention)")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--save-merged-model", action="store_true", help="Save merged model after training")

    # Model configuration
    parser.add_argument("--model-config", type=str, help="Path to model configuration JSON file (optional)")

    args = parser.parse_args()

    # Resolve the configuration first, then build the one Accelerator from it.
    config = resolve_training_config(args, DEFAULTS)
    lora_config = resolve_lora_config(args, config.lora_config)
    config.lora_config = lora_config
    accelerator = build_accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed
    )

    # Load model configuration if provided
    model_config = None
    if args.model_config:
        if accelerator.is_main_process:
            print(f"Loading model configuration from: {args.model_config}")
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)

    # Display configuration (main process only).
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("LORA INSTRUCTION FINE-TUNING (Accelerated)")
        print("=" * 60)
        print(f"\nInstruction corpus: {args.instruction_corpus_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Foundational checkpoint: {args.foundational_checkpoint}")
        print(f"Output directory: {args.output_dir}")
        print(f"Processes: {accelerator.num_processes}")
        if args.config:
            print(f"Training configuration file: {args.config}")
        print("\nLoRA Configuration:")
        print(f"  Rank: {lora_config.rank}")
        print(f"  Alpha: {lora_config.alpha}")
        print(f"  Target modules: {', '.join(lora_config.target_modules)}")
        print(f"  Dropout: {lora_config.dropout}")
        print_training_config(config)
        if args.resume_checkpoint:
            print(f"\nResuming from: {args.resume_checkpoint}")
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
            accelerator=accelerator,
            resume_checkpoint=args.resume_checkpoint
        )

        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("LORA FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Adapter checkpoint: {adapter_path}")
            print("=" * 60)

        # Destroy the process group cleanly (avoids teardown errors at exit).
        accelerator.end_training()

    except Exception as e:
        print(f"\n[Process {accelerator.process_index}] LORA FINE-TUNING FAILED!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
