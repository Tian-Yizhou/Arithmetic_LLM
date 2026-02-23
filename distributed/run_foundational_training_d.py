import argparse
import json
import torch
from accelerate import Accelerator
from configs.training_config_d import TrainingConfig
from training.train_foundational_d import train_foundational_model

def main():
    """Train foundational model from command line."""
    
    # Initialize Accelerator for distributed training and log control
    accelerator = Accelerator(
        mixed_precision="fp16"
    )

    parser = argparse.ArgumentParser(
        description="Train foundational arithmetic LLM model"
    )
    
    # Required arguments
    parser.add_argument("--corpus-path", type=str, required=True, help="Path to training corpus file")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to trained tokenizer directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model checkpoints")
    
    # Training configuration
    parser.add_argument("--config", type=str, help="Path to training configuration JSON file")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device (default: 32)")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps (default: 1000)")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value (default: 1.0)")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps (default: 1000)")
    
    # Gradient accumulation.
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients (default: 1)")

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on validation loss improvement ratio")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Number of epochs with insufficient improvement before stopping (default: 3)")
    parser.add_argument("--early-stopping-epsilon", type=float, default=1e-4, help="Minimum loss improvement ratio to continue training (default: 1e-4)")

    # Device argument is kept for CLI compatibility but ignored by Accelerate.
    parser.add_argument("--device", type=str, default="auto", help="Ignored when using accelerate")
    
    # Resume from checkpoint
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to checkpoint file to resume training from")

    # Model configuration
    parser.add_argument("--model-config", type=str, help="Path to model configuration JSON file")
    parser.add_argument("--d-model", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load or create training configuration
    if args.config:
        # Print only on main process.
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
    
    # Load or create model configuration
    if args.model_config:
        if accelerator.is_local_main_process:
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
    if accelerator.is_local_main_process:
        print("\n" + "=" * 60)
        print("FOUNDATIONAL MODEL TRAINING (Accelerated)")
        print("=" * 60)
        print(f"\nCorpus: {args.corpus_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Output directory: {args.output_dir}")
        print("\nTraining Configuration:")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size (per device): {config.batch_size}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Gradient clip: {config.gradient_clip}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Save every: {config.save_every} steps")
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
            resume_checkpoint=args.resume_checkpoint
        )
        
        # Print success message on main process only.
        if accelerator.is_local_main_process:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Final checkpoint: {final_checkpoint}")
            print("=" * 60)
            
    except Exception as e:
        # Print errors on all processes to help identify which one failed.
        print(f"\n[Process {accelerator.process_index}] TRAINING FAILED!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()