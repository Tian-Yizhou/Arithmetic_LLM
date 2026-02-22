"""Instruction fine-tuning script for arithmetic LLM.

This module implements the fine-tuning pipeline for the instruction-tuned model
on instruction-formatted arithmetic data.
"""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Dict
from accelerate import Accelerator
import math

from model.transformer_model import ArithmeticTransformer
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.data_loader import create_dataloaders
from configs.training_config_d import TrainingConfig
from training.train_foundational_d import (
    get_linear_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    evaluate
)


def train_instruction_model(
    instruction_corpus_path: str,
    tokenizer_path: str,
    foundational_checkpoint: str,
    output_dir: str,
    config: TrainingConfig,
    model_config: Optional[Dict] = None,
    accelerator: Accelerator = None
) -> str:
    """Fine-tune model with instruction formatting."""

    # Require an Accelerator instance for distributed training.
    if accelerator is None:
        raise ValueError("Accelerator must be passed from main!")

    # File I/O and logging happen on the main process only.
    if accelerator.is_local_main_process:
        # Validate configuration
        config.validate()
        
        # Create unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.join(output_dir, f"instruction_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Fine-tuning output directory: {output_dir}")
        print(f"Configuration: {config.to_dict()}")
        print("Loading tokenizer...")

    # Load tokenizer on all processes.
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    
    if accelerator.is_local_main_process:
        print(f"Tokenizer vocabulary size: {vocab_size}")
        print("Initializing model architecture...")

    # Load checkpoint metadata to get config.
    # map_location='cpu' prevents all processes from crowding GPU 0.
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

    # Validate tokenizer vocab size
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
    
    # Load weights before prepare() — Accelerate will move the model
    # to the correct device automatically.
    if accelerator.is_local_main_process:
        print(f"Loading foundational model from: {foundational_checkpoint}")
        
    checkpoint_metadata = load_checkpoint(
        checkpoint_path=foundational_checkpoint,
        model=model
    )
    
    if accelerator.is_local_main_process:
        print(f"Loaded checkpoint from epoch {checkpoint_metadata['epoch']}, "
              f"step {checkpoint_metadata['step']}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Prepare model, optimizer, and dataloaders for distributed training.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    # Calculate total training steps (update steps, not micro-batches).
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * config.num_epochs

    # Use 5% of total steps for learning rate warmup.
    warmup_ratio = 0.05
    real_warmup_steps = int(total_steps * warmup_ratio)
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=real_warmup_steps,
        num_training_steps=total_steps
    )
    
    # Prepare scheduler for distributed training.
    scheduler = accelerator.prepare(scheduler)
    
    # Save configuration (main process only).
    if accelerator.is_local_main_process:
        config.to_json(os.path.join(output_dir, 'training_config.json'))
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        with open(os.path.join(output_dir, 'foundational_checkpoint.txt'), 'w') as f:
            f.write(foundational_checkpoint)
        
        print("\nStarting instruction fine-tuning...")
    
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
        
        # Train one epoch.
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

        # Evaluate on validation set.
        if accelerator.is_local_main_process:
            print("\nEvaluating on validation set...")
        val_loss = evaluate(model, val_dataloader, config, accelerator)
        
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            
            training_log.append({
                'epoch': epoch + 1,
                'step': global_step,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Unwrap model to get raw state dict for saving.
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
    
    # Save final model
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
        print("\nSaving final fine-tuned model...")
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
        
        # Save logs
        log_path = os.path.join(output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        summary = {
            'total_epochs': config.num_epochs,
            'total_steps': global_step,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'model_config': model_config,
            'training_config': config.to_dict(),
            'tokenizer_vocab_size': vocab_size,
            'foundational_checkpoint': foundational_checkpoint
        }
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n" + "="*60)
        print("Instruction fine-tuning completed successfully!")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        return final_checkpoint_path
    
    return ""


