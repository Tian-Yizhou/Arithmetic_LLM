#!/usr/bin/env python3
"""Command-line interface for instruction fine-tuning."""

import argparse
import json
from configs.cli import add_training_arguments, print_training_config, resolve_training_config
from training.common import build_accelerator
from training.train_instruction import train_instruction_model


# Stage defaults; a --config JSON file and explicit flags override them.
DEFAULTS = {
    "learning_rate": 5e-5,
    "batch_size": 32,
    "num_epochs": 5,
    "save_every": 500,
}


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
    add_training_arguments(parser, DEFAULTS)

    # Model configuration
    parser.add_argument("--model-config", type=str, help="Path to model configuration JSON file")

    args = parser.parse_args()

    # Resolve the configuration first, then build the one Accelerator from it.
    config = resolve_training_config(args, DEFAULTS)
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
        print("INSTRUCTION FINE-TUNING (Accelerated)")
        print("=" * 60)
        print(f"\nInstruction corpus: {args.instruction_corpus_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Foundational checkpoint: {args.foundational_checkpoint}")
        print(f"Output directory: {args.output_dir}")
        print(f"Processes: {accelerator.num_processes}")
        if args.config:
            print(f"Training configuration file: {args.config}")
        print_training_config(config)
        if args.resume_checkpoint:
            print(f"\nResuming from: {args.resume_checkpoint}")
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
            accelerator=accelerator,
            resume_checkpoint=args.resume_checkpoint
        )

        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Final checkpoint: {final_checkpoint}")
            print("=" * 60)

        # Destroy the process group cleanly (avoids teardown errors at exit).
        accelerator.end_training()

    except Exception as e:
        print(f"\n[Process {accelerator.process_index}] FINE-TUNING FAILED!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
