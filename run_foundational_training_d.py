import argparse
import json
import torch
from accelerate import Accelerator  # [新增 1]
from training_config_d import TrainingConfig
from train_foundational_d import train_foundational_model

def main():
    """Train foundational model from command line."""
    
    # [新增 2] 初始化 Accelerator 用于控制日志
    # 注意：这里初始化主要为了 is_local_main_process，
    # 真正的配置会在 train_foundational_model 里再次确认，这是安全的。
    accelerator = Accelerator()

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
    
    # [新增 3] 梯度累积参数
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients (default: 1)")

    # [修改 4] Device 参数通常不再需要，accelerate 会自动处理。
    # 但为了兼容性可以保留，只是在代码里忽略它。
    parser.add_argument("--device", type=str, default="auto", help="Ignored when using accelerate")
    
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
        # [修改 5] 只在主进程打印
        if accelerator.is_local_main_process:
            print(f"Loading training configuration from: {args.config}")
        config = TrainingConfig.from_json(args.config)
    else:
        # [修改 6] 移除 device 判断逻辑，直接使用 TrainingConfig
        # 并传入新增的 gradient_accumulation_steps
        config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            gradient_clip=args.gradient_clip,
            save_every=args.save_every,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # device=...  <-- 不再传递 device
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
    
    # [修改 7] Display configuration - 只在主进程打印
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
        # print(f"  Device: {config.device}") <-- 删除
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
            model_config=model_config
        )
        
        # 成功信息也只在主进程打印
        if accelerator.is_local_main_process:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Final checkpoint: {final_checkpoint}")
            print("=" * 60)
            
    except Exception as e:
        # 错误信息最好都打印出来，方便定位是哪个进程挂了
        print(f"\n[Process {accelerator.process_index}] TRAINING FAILED!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()