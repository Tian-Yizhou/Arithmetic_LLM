"""CLI for GRPO training."""

import argparse
import os

from configs.grpo_config import GRPOConfig
from training.train_grpo_d import train_grpo_model
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
    
    # [新增 1] 初始化 Accelerator
    # 这一步接管了设备放置和分布式环境检测
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" # 建议显式开启混合精度，特别是对于 GRPO 这种计算量大的任务
    )

    # 验证参数 (通常只打印警告，或者抛出异常)
    # 如果 _validate_args 内部有 print，最好加上 if accelerator.is_local_main_process: 
    # 但由于看不到具体实现，先保留原样，只要它不报错就行
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
    
    # 验证配置
    config.validate()

    # [新增 2] 如果你想打印配置信息，只在主进程打印
    if accelerator.is_local_main_process:
        print(f"GRPO Training Configuration loaded.")
        # print(config) # 可选：打印详细配置

    train_grpo_model(
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
        accelerator=accelerator,  # [修改 3] 关键：传入 accelerator 对象
    )


if __name__ == "__main__":
    main()
