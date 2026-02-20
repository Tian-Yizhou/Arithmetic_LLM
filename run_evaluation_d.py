#!/usr/bin/env python3
"""Distributed command-line interface for model evaluation.

Launch with: accelerate launch run_evaluation_d.py --model-path ... --tokenizer-path ...
"""

import argparse
from accelerate import Accelerator
from evaluator_d import ModelEvaluator


def main():
    """Evaluate trained model from command line with distributed support."""
    accelerator = Accelerator()
    is_main = accelerator.is_local_main_process

    parser = argparse.ArgumentParser(
        description="Evaluate trained arithmetic LLM model (distributed)"
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        help="Path to base model checkpoint when evaluating a LoRA adapter"
    )

    # Evaluation configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of test expressions to generate (default: 1000)"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth of test expressions (default: 5)"
    )

    parser.add_argument(
        "--num-range",
        type=int,
        nargs=2,
        default=[1, 20],
        metavar=("MIN", "MAX"),
        help="Range of numbers in test expressions (default: 1 20)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )

    parser.add_argument(
        "--max-gen-length",
        type=int,
        default=512,
        help="Maximum generation length in tokens (default: 512)"
    )

    args = parser.parse_args()

    # Display configuration (main process only)
    if is_main:
        print("\n" + "=" * 60)
        print("MODEL EVALUATION (DISTRIBUTED)")
        print("=" * 60)
        print(f"\nModel: {args.model_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Device: {accelerator.device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print("\nEvaluation Configuration:")
        print(f"  Test samples: {args.num_samples}")
        print(f"  Max depth: {args.max_depth}")
        print(f"  Number range: {args.num_range[0]} to {args.num_range[1]}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Max generation length: {args.max_gen_length}")
        print(f"  Output directory: {args.output_dir}")
        print("=" * 60 + "\n")

    # Create evaluator
    try:
        if is_main:
            print("Loading model and tokenizer...")

        evaluator = ModelEvaluator(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            base_checkpoint_path=args.base_checkpoint,
            accelerator=accelerator
        )

        if is_main:
            print("Model loaded successfully!\n")
            print("Starting evaluation...")

        # Run evaluation
        metrics = evaluator.evaluate(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            num_range=tuple(args.num_range),
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_gen_length=args.max_gen_length
        )

        accelerator.wait_for_everyone()

        # Display results (main process only)
        if is_main:
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"\nTotal Samples: {metrics['total_samples']}")
            print(f"Correct Samples: {metrics['correct_samples']}")
            print(f"Parseable Samples: {metrics['parseable_samples']}")
            print(f"\nExact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
            print(f"Parse Success Rate: {metrics['parse_success_rate']:.2f}%")
            print(f"Avg Generation Length: {metrics['avg_generation_length']:.2f} tokens")
            print("=" * 60)

            # Provide interpretation
            print("\nInterpretation:")
            if metrics['exact_match_accuracy'] >= 80:
                print("  Excellent performance!")
            elif metrics['exact_match_accuracy'] >= 60:
                print("  Good performance")
            elif metrics['exact_match_accuracy'] >= 40:
                print("  ~ Moderate performance - consider more training")
            else:
                print("  Poor performance - model needs more training or debugging")

            if metrics['parse_success_rate'] < 90:
                print("  ! Low parse success rate - model may need better instruction tuning")

            print("=" * 60 + "\n")

    except Exception as e:
        if is_main:
            print("\n" + "=" * 60)
            print("EVALUATION FAILED!")
            print("=" * 60)
            print(f"Error [Process {accelerator.process_index}]: {str(e)}")
            print("=" * 60)
        raise


if __name__ == "__main__":
    main()
