"""Instruction fine-tuning script for arithmetic LLM.

This module implements the fine-tuning pipeline for the instruction-tuned model
on instruction-formatted arithmetic data.
"""

import os
import torch
from typing import Optional, Dict
from accelerate import Accelerator

from model.transformer_model import ArithmeticTransformer
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.data_loader import create_dataloaders
from configs.training_config import TrainingConfig
from training.common import load_checkpoint, path_sha256
from training.supervised_loop import describe_data, run_supervised_training, write_summary


def train_instruction_model(
    instruction_corpus_path: str,
    tokenizer_path: str,
    foundational_checkpoint: str,
    output_dir: str,
    config: TrainingConfig,
    model_config: Optional[Dict] = None,
    accelerator: Accelerator = None,
    resume_checkpoint: Optional[str] = None
) -> str:
    """Fine-tune model with instruction formatting."""

    # Require an Accelerator instance for distributed training.
    if accelerator is None:
        raise ValueError("Accelerator must be passed from main!")

    config.validate()

    if accelerator.is_main_process:
        print("Loading tokenizer...")

    # Load tokenizer on all processes.
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        print("Creating dataloaders...")
        
    train_dataloader, val_dataloader = create_dataloaders(
        corpus_path=instruction_corpus_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=max_seq_length,
        train_split=0.9,
        shuffle=True,
        num_workers=0,
        mode="instruction",
        seed=config.seed
    )
    data_info = describe_data(instruction_corpus_path, tokenizer_path, "instruction", config,
                              train_dataloader, val_dataloader)
    data_info['foundational_checkpoint_sha256'] = path_sha256(foundational_checkpoint)
    
    if accelerator.is_main_process:
        print(f"Training batches: {len(train_dataloader)}")
        print(f"Validation batches: {len(val_dataloader)}")
    
    # Initialize model
    model = ArithmeticTransformer(**model_config)
    
    # Load weights before prepare() — Accelerate will move the model
    # to the correct device automatically.
    if accelerator.is_main_process:
        print(f"Loading foundational model from: {foundational_checkpoint}")
        
    checkpoint_metadata = load_checkpoint(
        checkpoint_path=foundational_checkpoint,
        model=model
    )
    
    if accelerator.is_main_process:
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
    
    def write_run_files(run_dir: str) -> None:
        with open(os.path.join(run_dir, 'foundational_checkpoint.txt'), 'w') as f:
            f.write(foundational_checkpoint)

    if accelerator.is_main_process:
        print("\nStarting instruction fine-tuning...")

    result = run_supervised_training(
        stage="instruction",
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        accelerator=accelerator,
        output_parent=output_dir,
        data_info=data_info,
        tokenizer_vocab_size=vocab_size,
        model_config=model_config,
        resume_checkpoint=resume_checkpoint,
        write_run_files=write_run_files,
    )
    summary = result['summary']
    summary['foundational_checkpoint'] = foundational_checkpoint
    write_summary(accelerator, result['output_dir'], summary)

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Instruction fine-tuning completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        print("="*60)

    return result['final_checkpoint_path']
