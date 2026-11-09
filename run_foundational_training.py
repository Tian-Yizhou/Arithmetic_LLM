import argparse
import json
from configs.cli import add_training_arguments, print_training_config, resolve_training_config
from training.common import build_accelerator
from training.train_foundational import train_foundational_model


# Stage defaults; a --config JSON file and explicit flags override them.
DEFAULTS = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 10,
    "save_every": 1000,
}


def main():
    """Train foundational model from command line."""

    parser = argparse.ArgumentParser(
        description="Train foundational arithmetic LLM model"
    )

    # Required arguments
    parser.add_argument("--corpus-path", type=str, required=True, help="Path to training corpus file")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to trained tokenizer directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model checkpoints")

    # Training configuration
    add_training_arguments(parser, DEFAULTS)

    # Model configuration
    parser.add_argument("--model-config", type=str, help="Path to model configuration JSON file")
    parser.add_argument("--d-model", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    # Resolve the configuration first, then build the one Accelerator from it.
    config = resolve_training_config(args, DEFAULTS)
    accelerator = build_accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed
    )

    # Load or create model configuration
    if args.model_config:
        if accelerator.is_main_process:
            print(f"Loading model configuration from: {args.model_config}")
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        model_config = {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'max_seq_length': args.max_seq_length
        }

    # Display configuration (main process only).
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("FOUNDATIONAL MODEL TRAINING (Accelerated)")
        print("=" * 60)
        print(f"\nCorpus: {args.corpus_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Processes: {accelerator.num_processes}")
        if args.config:
            print(f"Training configuration file: {args.config}")
        print_training_config(config)
        if args.resume_checkpoint:
            print(f"\nResuming from: {args.resume_checkpoint}")
        print("\nModel Configuration:")
        print(f"  d_model: {model_config['d_model']}")
        print(f"  nhead: {model_config['nhead']}")
        print(f"  num_layers: {model_config['num_layers']}")
        print(f"  dim_feedforward: {model_config['dim_feedforward']}")
        print(f"  dropout: {model_config['dropout']}")
        print(f"  max_seq_length: {model_config['max_seq_length']}")
        print("=" * 60 + "\n")

    # Train model
    try:
        final_checkpoint = train_foundational_model(
            corpus_path=args.corpus_path,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            config=config,
            model_config=model_config,
            accelerator=accelerator,
            resume_checkpoint=args.resume_checkpoint
        )

        # Print success message on main process only.
        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Final checkpoint: {final_checkpoint}")
            print("=" * 60)

        # Destroy the process group cleanly (avoids teardown errors at exit).
        accelerator.end_training()

    except Exception as e:
        # Print errors on all processes to help identify which one failed.
        print(f"\n[Process {accelerator.process_index}] TRAINING FAILED!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
