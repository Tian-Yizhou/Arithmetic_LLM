"""CLI for GRPO training."""

import argparse
import os

from configs.grpo_config import GRPOConfig
from training.train_grpo import train_grpo_model
from accelerate import Accelerator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument(
        "--instruction-corpus",
        type=str,
        default=None,
        help="Path to instruction corpus JSONL (required for instruction mode)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        required=True,
        help="Path to SFT checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        choices=["instruction", "generated"],
        default="instruction",
        help="Training data mode"
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.05)
    parser.add_argument("--max-gen-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on reward_rate improvement ratio")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Number of evaluations with insufficient improvement before stopping (default: 3)")
    parser.add_argument("--early-stopping-epsilon", type=float, default=1e-4, help="Minimum reward_rate improvement ratio to continue training (default: 1e-4)")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to GRPO checkpoint file to resume training from")
    parser.add_argument(
        "--candidate-sub-batch-size",
        type=int,
        default=None,
        help="Optional sub-batch size for candidate processing"
    )
    parser.add_argument(
        "--filter-invalid-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter instruction entries with invalid/mismatched expressions"
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-range-min", type=int, default=1)
    parser.add_argument("--num-range-max", type=int, default=20)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.data_mode == "instruction" and not args.instruction_corpus:
        raise ValueError("--instruction-corpus is required for instruction mode")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(args.tokenizer)
    if not os.path.exists(args.sft_checkpoint):
        raise FileNotFoundError(args.sft_checkpoint)
    if args.num_range_min > args.num_range_max:
        raise ValueError("num-range-min must be <= num-range-max")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    
    # Initialize Accelerator for distributed training and device placement.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16"
    )

    # Validate CLI arguments.
    _validate_args(args)

    config = GRPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        eval_every=args.eval_every,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        kl_penalty_coef=args.kl_penalty_coef,
        max_gen_length=args.max_gen_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every=args.log_every,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_epsilon=args.early_stopping_epsilon,
    )
    
    # Validate configuration.
    config.validate()

    # Display configuration (main process only)
    if accelerator.is_local_main_process:
        print("\n" + "=" * 60)
        print("GRPO TRAINING (Accelerated)")
        print("=" * 60)
        print(f"\nData mode: {args.data_mode}")
        if args.instruction_corpus:
            print(f"Instruction corpus: {args.instruction_corpus}")
        print(f"Tokenizer: {args.tokenizer}")
        print(f"SFT checkpoint: {args.sft_checkpoint}")
        print(f"Output directory: {args.output_dir}")
        print("\nTraining Configuration:")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size (per device): {config.batch_size}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Gradient clip: {config.gradient_clip}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Save every: {config.save_every} steps")
        print(f"  Eval every: {config.eval_every} steps")
        print(f"  Log every: {config.log_every} steps")
        print("\nGRPO Configuration:")
        print(f"  Num candidates: {config.num_candidates}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-k: {config.top_k}")
        print(f"  Top-p: {config.top_p}")
        print(f"  KL penalty coef: {config.kl_penalty_coef}")
        print(f"  Max generation length: {config.max_gen_length}")
        if args.resume_checkpoint:
            print(f"\nResuming from: {args.resume_checkpoint}")
        if config.early_stopping:
            print("\nEarly Stopping:")
            print(f"  Patience: {config.early_stopping_patience}")
            print(f"  Epsilon: {config.early_stopping_epsilon}")
        print("=" * 60 + "\n")

    # Train model
    try:
        result = train_grpo_model(
            instruction_corpus_path=args.instruction_corpus,
            tokenizer_path=args.tokenizer,
            sft_checkpoint_path=args.sft_checkpoint,
            output_dir=args.output_dir,
            config=config,
            data_mode=args.data_mode,
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            num_range=(args.num_range_min, args.num_range_max),
            filter_invalid_instruction=args.filter_invalid_instruction,
            candidate_sub_batch_size=args.candidate_sub_batch_size,
            accelerator=accelerator,
            resume_checkpoint=args.resume_checkpoint,
        )

        if accelerator.is_local_main_process:
            print("\n" + "=" * 60)
            print("GRPO TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Total steps: {result.get('global_step', 'N/A')}")
            print(f"Final checkpoint: {result.get('final_checkpoint_path', 'N/A')}")
            if result.get('log_path'):
                print(f"Training log: {result['log_path']}")
            print("=" * 60)

    except Exception as e:
        print(f"\n[Process {accelerator.process_index}] GRPO TRAINING FAILED!")
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
