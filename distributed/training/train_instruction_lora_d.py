"""LoRA instruction fine-tuning script for arithmetic LLM."""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Dict
from accelerate import Accelerator
import math
import time

from model.transformer_model import ArithmeticTransformer
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.data_loader import create_dataloaders
from configs.training_config_d import TrainingConfig
from configs.lora_config import LoRAConfig
from model.lora_utils import get_parameter_stats
from training.train_foundational_d import (
    get_linear_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    evaluate,
)


def freeze_non_lora_parameters(model: ArithmeticTransformer) -> None:
    """Freeze all parameters except LoRA adapters."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.get_lora_parameters():
        param.requires_grad = True


def create_lora_optimizer(
    model: ArithmeticTransformer,
    config: TrainingConfig
) -> torch.optim.Optimizer:
    """Create optimizer for LoRA parameters only."""
    lora_params = list(model.get_lora_parameters())
    if not lora_params:
        raise ValueError("No LoRA parameters found for optimization")

    return torch.optim.AdamW(
        lora_params,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )



def train_instruction_model_lora(
    instruction_corpus_path: str,
    tokenizer_path: str,
    foundational_checkpoint: str,
    output_dir: str,
    config: TrainingConfig,
    lora_config: Optional[LoRAConfig] = None,
    model_config: Optional[Dict] = None,
    save_merged_model: bool = False,
    accelerator: Accelerator = None  # [修改 1] 新增参数
) -> str:
    """Fine-tune model with LoRA on instruction-formatted data."""

    # [修改 1] 在函数最开始记录开始时间
    start_time = time.time()
    
    # [安全检查] 确保 accelerator 被传入
    if accelerator is None:
        raise ValueError("Accelerator must be passed from main script!")

    # Validate configuration
    # 验证逻辑所有进程都跑一遍没坏处，或者只在主进程跑
    config.validate()

    if lora_config is None:
        lora_config = config.lora_config

    if lora_config is None:
        raise ValueError("LoRA configuration is required for LoRA training")

    lora_config.validate()
    config.lora_config = lora_config

    # [修改 2] 目录创建只在主进程
    if accelerator.is_local_main_process:
        # Create unique output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.join(output_dir, f"instruction_lora_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"LoRA fine-tuning output directory: {output_dir}")
        print(f"Training configuration: {config.to_dict()}")
        print("Loading tokenizer...")

    # Load tokenizer (所有进程都需要)
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    
    if accelerator.is_local_main_process:
        print(f"Tokenizer vocabulary size: {vocab_size}")
        print("Initializing model architecture...")

    # Load checkpoint data
    checkpoint_data = torch.load(foundational_checkpoint, map_location='cpu')
    if model_config is None:
        model_config = checkpoint_data.get('model_config', {
            'vocab_size': vocab_size,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 512
        })
    else:
        model_config['vocab_size'] = vocab_size

    checkpoint_vocab_size = checkpoint_data.get('tokenizer_vocab_size')
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({vocab_size}) does not match checkpoint "
            f"vocab size ({checkpoint_vocab_size})."
        )

    model_config['vocab_size'] = vocab_size
    max_seq_length = model_config.get('max_seq_length', 512)

    # Create dataloaders
    if accelerator.is_local_main_process:
        print("Creating dataloaders...")
        
    train_dataloader, val_dataloader = create_dataloaders(
        corpus_path=instruction_corpus_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=max_seq_length,
        train_split=0.9,
        shuffle=True,
        num_workers=0,
        mode="instruction"
    )
    
    if accelerator.is_local_main_process:
        print(f"Training batches: {len(train_dataloader)}")
        print(f"Validation batches: {len(val_dataloader)}")

    # Initialize model
    model = ArithmeticTransformer(**model_config)

    # Load foundational model weights
    if accelerator.is_local_main_process:
        print(f"Loading foundational model from: {foundational_checkpoint}")
        
    checkpoint_metadata = load_checkpoint(
        checkpoint_path=foundational_checkpoint,
        model=model
        # 确保 load_checkpoint 内部没有 .to(device) 操作
    )
    
    if accelerator.is_local_main_process:
        print(f"Loaded checkpoint from epoch {checkpoint_metadata['epoch']}, "
              f"step {checkpoint_metadata['step']}")

    # Inject LoRA and freeze parameters
    # [注意] 这些操作在 prepare 之前做是安全的，所有进程都会执行
    model.inject_lora(lora_config)
    freeze_non_lora_parameters(model)

    # [修改 3] 移除 model.to(device)
    # model = model.to(config.device) <--- DELETE

    # Count parameters (只在主进程打印)
    if accelerator.is_local_main_process:
        stats = get_parameter_stats(model)
        print(f"Total parameters: {stats['total']:,}")
        print(f"Trainable parameters: {stats['trainable']:,}")
        print(f"Frozen parameters: {stats['frozen']:,}")
        print(f"Trainable percentage: {stats['trainable_pct']:.2f}%")

    # Initialize optimizer
    # 确保 create_lora_optimizer 能处理参数已经被过滤过的情况
    optimizer = create_lora_optimizer(model, config)

    # [修改 4] 第一阶段 Prepare
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # [modified] Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * config.num_epochs
    
    # 动态计算 warmup
    warmup_ratio = 0.05 # 5% 的步数用于热身
    calculated_warmup_steps = int(total_steps * warmup_ratio)

    # 确保至少有一点热身，且不超过 config 的硬性限制（如果有的话）
    real_warmup_steps = calculated_warmup_steps

    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=real_warmup_steps,
        num_training_steps=total_steps
    )
    
    # [修改 6] 第二阶段 Prepare Scheduler
    scheduler = accelerator.prepare(scheduler)

    # Save configuration (主进程)
    if accelerator.is_local_main_process:
        config.to_json(os.path.join(output_dir, 'training_config.json'))
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        with open(os.path.join(output_dir, 'foundational_checkpoint.txt'), 'w') as f:
            f.write(foundational_checkpoint)

        print("\nStarting LoRA instruction fine-tuning...")

    global_step = 0
    best_val_loss = float('inf')
    training_log = []
    prev_val_loss = None
    patience_counter = 0
    early_stopped = False

    for epoch in range(config.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"{'='*60}")

        # [修改 7] 传递 accelerator
        train_loss, global_step = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch + 1,
            global_step=global_step,
            output_dir=output_dir,
            tokenizer_vocab_size=vocab_size,
            accelerator=accelerator # Pass accelerator
        )

        # [修改 8] 传递 accelerator
        if accelerator.is_local_main_process:
            print("\nEvaluating on validation set...")
        val_loss = evaluate(model, val_dataloader, config, accelerator)

        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")

            # Log metrics
            training_log.append({
                'epoch': epoch + 1,
                'step': global_step,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })

            # Save best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # [修改 9] 解包模型 (Unwrap)
                # 只有解包后，save_checkpoint 才能正确处理
                unwrapped_model = accelerator.unwrap_model(model)
                
                best_checkpoint_path = save_checkpoint(
                    model=unwrapped_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    step=global_step,
                    loss=val_loss,
                    config=config,
                    tokenizer_vocab_size=vocab_size,
                    output_dir=output_dir,
                    is_final=False
                )
                
                # 重命名
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                os.rename(best_checkpoint_path, best_model_path)
                print(f"  New best model saved: {best_model_path}")

        # Early stopping check (all processes see the same val_loss from gather)
        if config.early_stopping and prev_val_loss is not None:
            if prev_val_loss == 0:
                improvement_ratio = 0.0 if val_loss == 0 else float('inf')
            else:
                improvement_ratio = (prev_val_loss - val_loss) / abs(prev_val_loss)

            if improvement_ratio < config.early_stopping_epsilon:
                patience_counter += 1
                if accelerator.is_local_main_process:
                    print(f"  Early stopping: insufficient improvement ({improvement_ratio:.6f} < {config.early_stopping_epsilon}), patience {patience_counter}/{config.early_stopping_patience}")
                if patience_counter >= config.early_stopping_patience:
                    if accelerator.is_local_main_process:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                    early_stopped = True
            else:
                patience_counter = 0

        prev_val_loss = val_loss

        # [修改 10] 等待同步
        accelerator.wait_for_everyone()

        if early_stopped:
            break

    # Save final model
    accelerator.wait_for_everyone() # 确保所有进程都跑完了

    adapter_path = "" # 初始化返回值，防止非主进程报错
    
    if accelerator.is_local_main_process:
        print("\nSaving final fine-tuned model checkpoint...")
        
        # [修改 11] 解包模型
        unwrapped_model = accelerator.unwrap_model(model)
        
        final_checkpoint_path = save_checkpoint(
            model=unwrapped_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config.num_epochs,
            step=global_step,
            loss=train_loss,
            config=config,
            tokenizer_vocab_size=vocab_size,
            output_dir=output_dir,
            is_final=True
        )
        print(f"Final model checkpoint saved: {final_checkpoint_path}")

        # Save LoRA adapters separately
        # [修改 12] 调用 LoRA 特有的保存方法
        # 因为我们已经 unwrapped 了，所以可以直接调用 save_lora_adapters
        adapter_path = os.path.join(output_dir, 'lora_adapter.pt')
        if hasattr(unwrapped_model, 'save_lora_adapters'):
            unwrapped_model.save_lora_adapters(adapter_path, base_model_path=foundational_checkpoint)
            print(f"LoRA adapter saved: {adapter_path}")
        else:
            print("Warning: Model does not have 'save_lora_adapters' method.")

        merged_model_path = None
        if save_merged_model:
            print("Merging LoRA weights and saving merged model...")
            if hasattr(unwrapped_model, 'merge_lora_weights'):
                unwrapped_model.merge_lora_weights()
                merged_model_path = os.path.join(output_dir, 'merged_model.pt')
                torch.save(
                    {
                        'model_state_dict': unwrapped_model.state_dict(),
                        'config': config.to_dict(),
                        'tokenizer_vocab_size': vocab_size,
                        'merged': True,
                        'foundational_checkpoint': foundational_checkpoint,
                    },
                    merged_model_path
                )
                print(f"Merged model saved: {merged_model_path}")
            else:
                print("Warning: Model does not have 'merge_lora_weights' method.")

        # [修改 2] 计算总耗时并添加到 summary 中
        end_time = time.time()
        total_seconds = end_time - start_time
        
        # 格式化时间为 "HH:MM:SS"
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        # Save training log
        log_path = os.path.join(output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"Training log saved: {log_path}")

        # Save summary
        summary = {
            'total_epochs': config.num_epochs,
            'total_steps': global_step,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'model_config': model_config,
            'training_config': config.to_dict(),
            'tokenizer_vocab_size': vocab_size,
            'foundational_checkpoint': foundational_checkpoint,
            'lora_adapter_path': adapter_path,
            'merged_model_path': merged_model_path,
            # add training duration to summary
            'training_duration_seconds': total_seconds,
            'training_duration_formatted': formatted_time
        }
        
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved: {summary_path}")
        
        # print training duration
        print(f"Total training time: {formatted_time}")

        print("\n" + "="*60)
        print("LoRA instruction fine-tuning completed successfully!")
        print(f"Output directory: {output_dir}")
        print("="*60)

    return adapter_path


