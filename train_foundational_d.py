"""Foundational model training script for arithmetic LLM.

This module implements the training pipeline for the foundational transformer model
on arithmetic expressions and evaluations.
"""

import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict, Tuple
import math

from transformer_model import ArithmeticTransformer
from arithmetic_tokenizer import ArithmeticBPETokenizer
from data_loader import create_dataloaders
from training_config_d import TrainingConfig
# import accelerate to distribute training
from accelerate import Accelerator


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with linear warmup and decay.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch number for resuming training
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(
    model: ArithmeticTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    step: int,
    loss: float,
    config: TrainingConfig,
    tokenizer_vocab_size: int,
    output_dir: str,
    is_final: bool = False
) -> str:
    """Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch number
        step: Current training step
        loss: Current training loss
        config: Training configuration
        tokenizer_vocab_size: Size of tokenizer vocabulary
        output_dir: Directory to save checkpoint
        is_final: Whether this is the final checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': config.to_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'nhead': model.nhead,
            'num_layers': model.num_layers,
            'dim_feedforward': model.dim_feedforward,
            'dropout': model.dropout,
            'max_seq_length': model.max_seq_length,
        },
        'tokenizer_vocab_size': tokenizer_vocab_size,
    }
    
    if is_final:
        checkpoint_path = os.path.join(output_dir, 'final_model.pt')
    else:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{step}.pt')
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: ArithmeticTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
) -> Dict:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0.0),
        'config': checkpoint.get('config', {}),
        'tokenizer_vocab_size': checkpoint.get('tokenizer_vocab_size', 0),
    }


def train_epoch(
    model: ArithmeticTransformer,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    output_dir: str,
    tokenizer_vocab_size: int,
    accelerator: Accelerator
) -> Tuple[float, int]:
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if accelerator.is_local_main_process:
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    else:
        progress_bar = train_dataloader
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
        # [关键修改 1] 使用 accumulate 包装整个前向+后向过程
        # 这会自动处理梯度累积：只有达到 accumulation_steps 时才会真正同步梯度
        with accelerator.accumulate(model):
            # Prepare inputs and targets
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:] 
            input_attention_mask = attention_mask[:, :-1]
            
            # Forward pass
            logits = model(inputs, input_attention_mask)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100 
            )
            
            # Backward pass
            # 注意：在 accumulate 下，zero_grad 应该放在循环开头或此处
            
            accelerator.backward(loss)
            
            # [关键修改 2] 使用 accelerator 专用的梯度裁剪
            # 它会自动处理多卡间的梯度同步
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # [关键修改 3] 更新优化器和调度器
            # 在 accumulate 环境下，step() 只有在梯度同步时（即累积完成时）才会真正执行
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
                
            optimizer.zero_grad()

        # [关键修改 4] global_step 的逻辑对齐
        # 只有当模型真正更新了权重（即一个 update step 完成）时，才增加 global_step
        if accelerator.sync_gradients:
            global_step += 1

        # 统计逻辑保持不变（按 batch 统计 loss）
        total_loss += loss.detach()
        num_batches += 1
        
        # 更新进度条
        if batch_idx % 10 == 0 and accelerator.is_local_main_process:
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{(total_loss / num_batches).item():.4f}",
                'lr': f"{current_lr:.2e}",
                'step': global_step  # 显示真正的 update step
            })
        
        # 保存 Checkpoint 逻辑
        # 现在的 global_step 是基于更新次数的，与你 scheduler 的 total_steps 完全对齐
        if global_step > 0 and global_step % config.save_every == 0 and accelerator.sync_gradients:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_checkpoint(
                    model=unwrapped_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    config=config,
                    tokenizer_vocab_size=tokenizer_vocab_size,
                    output_dir=output_dir,
                    is_final=False
                )
            accelerator.wait_for_everyone()
    
    avg_loss = (total_loss / num_batches).item() if num_batches > 0 else 0.0
    return avg_loss, global_step


def evaluate(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    accelerator: Accelerator  # [modified 1] add accelerator
) -> float:
    """Evaluate model on validation set."""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_dataloader:
            # [modified 2] delete .to(device)
            
            # Prepare inputs and targets
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            input_attention_mask = attention_mask[:, :-1]
            
            # Forward pass
            logits = model(inputs, input_attention_mask)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
            
            # [modified 3] gather loss from all GPUs to compute global average loss
            # loss is scaler, we need to reshape to gather
            all_losses = accelerator.gather(loss.reshape(1))
            
            # use average loss of current batch to compute global loss
            avg_batch_loss = all_losses.mean().item()
            
            total_loss += avg_batch_loss
            num_batches += 1
    
    # 计算所有 batch 的平均值
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def train_foundational_model(
    corpus_path: str,
    tokenizer_path: str,
    output_dir: str,
    config: TrainingConfig,
    model_config: Optional[Dict] = None
) -> str:
    """Train foundational model on arithmetic corpus.
    
    Args:
        corpus_path: Path to training corpus
        tokenizer_path: Path to trained tokenizer
        output_dir: Directory to save checkpoints and logs
        config: Training configuration
        model_config: Optional model architecture configuration
        
    Returns:
        Path to final model checkpoint
    """
    # [modified] Initialize accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps, # 如果你的 config 有这个参数
        mixed_precision="fp16" # 或者 "bf16"，也可以在 accelerate config 中配置
    )

    # [modified] only main process print logs
    if accelerator.is_local_main_process:
        # Validate configuration
        config.validate()
        
        # Create unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.join(output_dir, f"foundational_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Training output directory: {output_dir}")
        print(f"Configuration: {config.to_dict()}")
        print("Loading tokenizer...")

    # [注意] 所有进程都需要知道 output_dir，但它是在主进程生成的。
    # 简单的做法是：去掉 timestamp，或者让 timestamp 固定。
    # 为了严谨，这里应该广播 output_dir，但为了代码简单，建议在多卡训练时固定 output_dir 的路径，或者容忍非主进程不知道正确路径（只要它们不保存文件）。
    # 这里我们假设非主进程不需要知道 output_dir 用于保存。

    # Load tokenizer to all processes
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    
    if accelerator.is_local_main_process:
        print(f"Tokenizer vocabulary size: {vocab_size}")
        print("Initializing model configuration...")

    if model_config is None:
        model_config = {
            'vocab_size': vocab_size,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 512
        }
    else:
        model_config['vocab_size'] = vocab_size

    max_seq_length = model_config.get('max_seq_length', 512)

    # Create dataloaders
    if accelerator.is_local_main_process:
        print("Creating dataloaders...")
        
    train_dataloader, val_dataloader = create_dataloaders(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=max_seq_length,
        train_split=0.9,
        shuffle=True,
        num_workers=0,
        mode="foundational"
    )
    
    # Initialize model
    if accelerator.is_local_main_process:
        print("Initializing model...")
    
    model = ArithmeticTransformer(**model_config)
    
    # [modified] delete model.to(config.device)
    # accelerator.prepare handle devices
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    
    # [modified] Prepare model, optimizer, dataloaders with accelerator before calculating total steps and initializing scheduler
    # this will wrap the model with DDP or FSDP, and also handle moving model and data to the correct device in distributed training
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

    # [modified] prepare Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=real_warmup_steps,
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    print(f"DEBUG: Steps per Epoch = {len(train_dataloader)}")
    print(f"DEBUG: Total Steps sent to scheduler = {total_steps}")

    # save config and model config to output_dir (only main process to avoid conflicts)
    if accelerator.is_local_main_process:
        config.to_json(os.path.join(output_dir, 'training_config.json'))
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        print("\nStarting training...")

    global_step = 0
    best_val_loss = float('inf')
    training_log = []
    
    for epoch in range(config.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"{'='*60}")
        
        # [modified] add accelerator
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
            accelerator=accelerator  # input accelerator
        )
        
        # [modified] input accelerator to evaluate function to gather validation loss across GPUs
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
            
            # Save best model (only on main process to avoid conflicts)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                
                best_checkpoint_path = save_checkpoint(
                    model=unwrapped_model, # 使用 unwrapped model
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
                
                # Rename to best_model.pt
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                os.rename(best_checkpoint_path, best_model_path)
                print(f"  New best model saved: {best_model_path}")

        # wait all processes to finish and saving before next epoch
        accelerator.wait_for_everyone()
    
    # Save final model
    # we need wait and unwrap model before saving final checkpoint
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
        print("\nSaving final model...")
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
        print(f"Final model saved: {final_checkpoint_path}")
        
        # Save training log
        log_path = os.path.join(output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # Save summary
        summary = {
            'total_epochs': config.num_epochs,
            'total_steps': global_step,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'model_config': model_config,
            'training_config': config.to_dict(),
            'tokenizer_vocab_size': vocab_size
        }
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        return final_checkpoint_path
    else:
        return "" # return empty string for non-main processes, they don't save the model


