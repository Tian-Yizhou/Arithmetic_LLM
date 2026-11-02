"""Foundational model training script for arithmetic LLM.

This module implements the training pipeline for the foundational transformer model
on arithmetic expressions and evaluations.
"""

import torch
from typing import Optional, Dict

from model.transformer_model import ArithmeticTransformer
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.data_loader import create_dataloaders
from configs.training_config import TrainingConfig
# import accelerate to distribute training
from accelerate import Accelerator

# Re-exported for modules that historically imported these helpers from here.
from training.common import (  # noqa: F401
    _merge_lora_state_dict,
    get_linear_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
)
from training.supervised_loop import (  # noqa: F401
    describe_data,
    evaluate,
    run_supervised_training,
    train_epoch,
    write_summary,
)


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
        output_dir: Directory in which the run directory is created
        config: Training configuration
        model_config: Optional model architecture configuration
        accelerator: Accelerator built by the launcher (see build_accelerator)
        resume_checkpoint: Optional path to checkpoint file to resume training from

    Returns:
        Path to final model checkpoint
    """
    # Require an Accelerator instance, so there is exactly one per run and it
    # is built from the resolved configuration.
    if accelerator is None:
        raise ValueError("Accelerator must be passed from main script!")

    config.validate()

    if accelerator.is_main_process:
        print("Loading tokenizer...")

    # Load tokenizer on all processes.
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)

    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        print("Creating dataloaders...")

    train_dataloader, val_dataloader = create_dataloaders(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=max_seq_length,
        train_split=0.9,
        shuffle=True,
        num_workers=0,
        mode="foundational",
        seed=config.seed
    )
    data_info = describe_data(corpus_path, tokenizer_path, "foundational", config,
                              train_dataloader, val_dataloader)

    # Initialize model
    if accelerator.is_main_process:
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
    # This wraps the model with DDP and moves data to the correct device.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process:
        print("\nStarting training...")

    result = run_supervised_training(
        stage="foundational",
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
    )
    write_summary(accelerator, result['output_dir'], result['summary'])

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        print("="*60)

    return result['final_checkpoint_path']
