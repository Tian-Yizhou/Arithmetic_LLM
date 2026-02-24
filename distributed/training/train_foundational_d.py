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

from model.transformer_model import ArithmeticTransformer
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.data_loader import create_dataloaders
from configs.training_config_d import TrainingConfig
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


def _merge_lora_state_dict(state_dict: dict, checkpoint_config: dict) -> dict:
    """Merge LoRA weights into base weights, producing a plain state dict.

    When a checkpoint was saved from a LoRA-adapted model, the state dict
    contains keys like 'layers.0.self_attention.q_proj.base_layer.weight'
    and 'layers.0.self_attention.q_proj.lora_A/lora_B'. This function
    merges them into plain keys like 'layers.0.self_attention.q_proj.weight'
    so the checkpoint can be loaded into a plain ArithmeticTransformer.
    """
    lora_config = checkpoint_config.get('lora_config', {})
    if isinstance(lora_config, dict):
        alpha = lora_config.get('alpha', 16.0)
        rank = lora_config.get('rank', 8)
    else:
        alpha = 16.0
        rank = 8
    scaling = alpha / rank

    merged = {}
    processed_prefixes = set()

    for key in state_dict:
        if '.base_layer.' in key:
            prefix, suffix = key.split('.base_layer.')
            plain_key = f"{prefix}.{suffix}"

            if suffix == 'weight' and prefix not in processed_prefixes:
                lora_a_key = f"{prefix}.lora_A"
                lora_b_key = f"{prefix}.lora_B"
                base_weight = state_dict[key]
                if lora_a_key in state_dict and lora_b_key in state_dict:
                    lora_a = state_dict[lora_a_key]
                    lora_b = state_dict[lora_b_key]
                    merged[plain_key] = base_weight + (lora_b @ lora_a) * scaling
                else:
                    merged[plain_key] = base_weight
                processed_prefixes.add(prefix)
            else:
                merged[plain_key] = state_dict[key]
        elif key.endswith('.lora_A') or key.endswith('.lora_B'):
            continue
        else:
            merged[key] = state_dict[key]

    return merged


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

    state_dict = checkpoint['model_state_dict']

    # Detect LoRA checkpoint being loaded into a plain model.
    has_lora_keys = any('.base_layer.' in k for k in state_dict)
    model_expects_lora = any('.base_layer.' in k for k in model.state_dict())

    if has_lora_keys and not model_expects_lora:
        state_dict = _merge_lora_state_dict(state_dict, checkpoint.get('config', {}))

    model.load_state_dict(state_dict)
    
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
        # Use accumulate() to wrap forward+backward: gradients are only
        # synchronized across processes when accumulation_steps is reached.
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
            accelerator.backward(loss)

            # Clip gradients only after they have been synchronized.
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # Under accumulate(), step() only takes effect when gradients
            # are synchronized (i.e. accumulation is complete).
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
                
            optimizer.zero_grad()

        # Increment global_step only when an actual weight update happens
        # (i.e. after gradient accumulation is complete).
        if accelerator.sync_gradients:
            global_step += 1

        # Accumulate loss statistics per batch.
        total_loss += loss.detach()
        num_batches += 1
        
        # Update progress bar.
        if batch_idx % 10 == 0 and accelerator.is_local_main_process:
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{(total_loss / num_batches).item():.4f}",
                'lr': f"{current_lr:.2e}",
                'step': global_step
            })
        
        # Save checkpoint at regular intervals.
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
            
            # Gather per-device losses to compute a global average.
            # Loss is a scalar; reshape to (1,) so gather can concatenate.
            all_losses = accelerator.gather(loss.reshape(1))
            
            # Average across all devices for this batch.
            avg_batch_loss = all_losses.mean().item()
            
            total_loss += avg_batch_loss
            num_batches += 1
    
    # Average loss across all batches.
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def train_foundational_model(
    corpus_path: str,
    tokenizer_path: str,
    output_dir: str,
    config: TrainingConfig,
    model_config: Optional[Dict] = None,
    accelerator: Accelerator = None,
    resume_checkpoint: Optional[str] = None
) -> str:
    """Train foundational model on arithmetic corpus.

    Args:
        corpus_path: Path to training corpus
        tokenizer_path: Path to trained tokenizer
        output_dir: Directory to save checkpoints and logs
        config: Training configuration
        model_config: Optional model architecture configuration
        accelerator: Pre-initialized Accelerator instance
        resume_checkpoint: Optional path to checkpoint file to resume training from

    Returns:
        Path to final model checkpoint
    """
    # Use provided accelerator or create a new one
    if accelerator is None:
        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="fp16"
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

    # Note: output_dir is timestamped and created on the main process only.
    # Non-main processes never write files, so they don't need the exact path.

    # Load tokenizer on all processes.
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
    
    # Device placement is handled by accelerator.prepare() below.
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    

    # Prepare model, optimizer, and dataloaders for distributed training.
    # This wraps the model with DDP/FSDP and moves data to the correct device.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Calculate total training steps (update steps, not micro-batches).
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * config.num_epochs

    # Use 5% of total steps for learning rate warmup.
    warmup_ratio = 0.05
    real_warmup_steps = int(total_steps * warmup_ratio)

    # Initialize and prepare the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=real_warmup_steps,
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_local_main_process:
        print(f"DEBUG: Steps per Epoch = {len(train_dataloader)}")
        print(f"DEBUG: Total Steps sent to scheduler = {total_steps}")

    # save config and model config to output_dir (only main process to avoid conflicts)
    if accelerator.is_local_main_process:
        config.to_json(os.path.join(output_dir, 'training_config.json'))
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        print("\nStarting training...")

    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    training_log = []
    prev_val_loss = None
    patience_counter = 0
    early_stopped = False

    # Resume from checkpoint if provided.
    if resume_checkpoint is not None:
        if accelerator.is_local_main_process:
            print(f"\nResuming training from checkpoint: {resume_checkpoint}")
        unwrapped_model = accelerator.unwrap_model(model)
        resume_metadata = load_checkpoint(
            checkpoint_path=resume_checkpoint,
            model=unwrapped_model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        start_epoch = resume_metadata['epoch']
        global_step = resume_metadata['step']
        if accelerator.is_local_main_process:
            print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    for epoch in range(start_epoch, config.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"{'='*60}")
        
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
            accelerator=accelerator
        )
        
        # Evaluate and gather validation loss across all GPUs.
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
                
                # Unwrap model before saving to get the raw state dict
                # (without DDP/FSDP wrapper prefixes).
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
                
                # Rename to best_model.pt
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

        # Wait for all processes to finish before next epoch.
        accelerator.wait_for_everyone()

        if early_stopped:
            break
    
    # Wait for all processes, then save final model on main process.
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
        return ""


