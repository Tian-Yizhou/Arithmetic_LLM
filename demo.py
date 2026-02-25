#!/usr/bin/env python3
"""Demo script: load a trained model and evaluate it on 10 example expressions.

Usage:
    python demo.py \
        --model-path models/instruction_XXXXXXXX/best_model.pt \
        --tokenizer-path data/tokenizer \
        --output demo_results.txt
"""

import argparse
import torch
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.generator import ExpressionGenerator
from model.transformer_model import ArithmeticTransformer
from evaluation.evaluator import eval_expression


def load_model(model_path, tokenizer, device):
    """Load model from checkpoint and return it in eval mode."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get("model_config") or checkpoint.get("config", {})
    model = ArithmeticTransformer(
        vocab_size=len(tokenizer.token2id),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 8),
        num_layers=config.get("num_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 1024),
        dropout=config.get("dropout", 0.1),
        max_seq_length=config.get("max_seq_length", 512),
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device).eval()
    return model


def generate_solution(model, tokenizer, expression, device, max_length=256):
    """Run inference on a single expression and return generated text."""
    prompt = f"Evaluate: {expression} <think>"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Remove trailing EOS so the model continues generating
    eos_id = tokenizer.token2id.get("<eos>")
    if eos_id is not None and input_ids and input_ids[-1] == eos_id:
        input_ids = input_ids[:-1]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            eos_token_id=eos_id,
        )

    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)


def extract_predicted_answer(text):
    """Extract the integer after 'Final Result:' from generated text."""
    import re

    error_match = re.search(r"Final Result\s*:\s*ERROR\b", text, re.IGNORECASE)
    if error_match:
        return "ERROR"
    match = re.search(r"Final Result\s*:\s*([+-]?\s*\d+)", text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1).replace(" ", ""))
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Arithmetic LLM Demo")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to tokenizer directory")
    parser.add_argument("--output", type=str, default="demo_results.txt",
                        help="Path to save evaluation results (default: demo_results.txt)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, mps, cpu, or auto (default: auto)")
    parser.add_argument("--num-examples", type=int, default=10,
                        help="Number of examples to evaluate (default: 10)")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Maximum depth of generated expressions (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible examples (default: 42)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(args.tokenizer_path)

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, tokenizer, device)
    print(f"Device: {device}\n")

    # Generate example expressions with a fixed seed for reproducibility
    import random
    random.seed(args.seed)
    generator = ExpressionGenerator(max_depth=args.max_depth, num_range=(1, 20), invalid_rate=0.0)
    expressions = [generator.generate() for _ in range(args.num_examples)]

    # Evaluate each expression
    lines = []
    correct = 0

    header = "=" * 70
    lines.append(header)
    lines.append("ARITHMETIC LLM — DEMO EVALUATION")
    lines.append(header)
    lines.append(f"Model      : {args.model_path}")
    lines.append(f"Tokenizer  : {args.tokenizer_path}")
    lines.append(f"Device     : {device}")
    lines.append(f"Examples   : {args.num_examples}")
    lines.append(header)
    lines.append("")

    for i, expr in enumerate(expressions, 1):
        # Ground truth
        ground_truth = eval_expression(expr)
        expected = ground_truth["answer"]

        # Model prediction
        generated_text = generate_solution(model, tokenizer, expr, device)
        predicted = extract_predicted_answer(generated_text)

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        status = "CORRECT" if is_correct else "WRONG"

        lines.append(f"--- Example {i}/{args.num_examples} [{status}] ---")
        lines.append(f"Expression     : {expr}")
        lines.append(f"Expected Answer: {expected}")
        lines.append(f"Model Output   :")
        # Indent generated text for readability
        for gl in generated_text.strip().splitlines():
            lines.append(f"  {gl}")
        lines.append(f"Predicted      : {predicted}")
        lines.append("")

        print(f"[{i}/{args.num_examples}] {expr} = {expected}  |  predicted: {predicted}  [{status}]")

    # Summary
    accuracy = correct / args.num_examples * 100
    lines.append(header)
    lines.append("SUMMARY")
    lines.append(header)
    lines.append(f"Total      : {args.num_examples}")
    lines.append(f"Correct    : {correct}")
    lines.append(f"Accuracy   : {accuracy:.1f}%")
    lines.append(header)

    # Write to file
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nAccuracy: {correct}/{args.num_examples} ({accuracy:.1f}%)")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
