"""LoRA instruction fine-tuning script for arithmetic LLM."""

import os
import torch
from typing import Optional, Dict
from accelerate import Accelerator
import time

from model.transformer_model import ArithmeticTransformer
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from data.data_loader import create_dataloaders
from configs.training_config import TrainingConfig
from configs.lora_config import LoRAConfig
from model.lora_utils import get_parameter_stats
from training.common import atomic_torch_save, describe_model, load_checkpoint, path_sha256
from training.supervised_loop import describe_data, run_supervised_training, write_summary


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
    accelerator: Accelerator = None,
    resume_checkpoint: Optional[str] = None
) -> str:
    """Fine-tune model with LoRA on instruction-formatted data."""

    start_time = time.time()

    # Require an Accelerator instance for distributed training.
    if accelerator is None:
        raise ValueError("Accelerator must be passed from main script!")

    # Validate configuration.
    config.validate()

    if lora_config is None:
        lora_config = config.lora_config

    if lora_config is None:
        raise ValueError("LoRA configuration is required for LoRA training")

    lora_config.validate()
    config.lora_config = lora_config

    if accelerator.is_main_process:
        print("Loading tokenizer...")

    # Load tokenizer on all processes.
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    
    if accelerator.is_main_process:
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

    # Load foundational model weights
    if accelerator.is_main_process:
        print(f"Loading foundational model from: {foundational_checkpoint}")
        
    checkpoint_metadata = load_checkpoint(
        checkpoint_path=foundational_checkpoint,
        model=model
    )
    
    if accelerator.is_main_process:
        print(f"Loaded checkpoint from epoch {checkpoint_metadata['epoch']}, "
              f"step {checkpoint_metadata['step']}")

    # Inject LoRA adapters and freeze base parameters before prepare().
    model.inject_lora(lora_config)
    freeze_non_lora_parameters(model)

    # Display parameter statistics (main process only).
    if accelerator.is_main_process:
        stats = get_parameter_stats(model)
        print(f"Total parameters: {stats['total']:,}")
        print(f"Trainable parameters: {stats['trainable']:,}")
        print(f"Frozen parameters: {stats['frozen']:,}")
        print(f"Trainable percentage: {stats['trainable_pct']:.2f}%")

    # Initialize optimizer for LoRA parameters only.
    optimizer = create_lora_optimizer(model, config)

    # Prepare model, optimizer, and dataloaders for distributed training.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    def write_run_files(run_dir: str) -> None:
        with open(os.path.join(run_dir, 'foundational_checkpoint.txt'), 'w') as f:
            f.write(foundational_checkpoint)

    if accelerator.is_main_process:
        print("\nStarting LoRA instruction fine-tuning...")

    result = run_supervised_training(
        stage="instruction_lora",
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
    output_dir = result['output_dir']

    adapter_path = ""
    merged_model_path = None
    
    if accelerator.is_main_process:
        # Unwrap model for saving.
        unwrapped_model = accelerator.unwrap_model(model)

        # Save LoRA adapters separately.
        adapter_path = os.path.join(output_dir, 'lora_adapter.pt')
        if hasattr(unwrapped_model, 'save_lora_adapters'):
            unwrapped_model.save_lora_adapters(adapter_path, base_model_path=foundational_checkpoint)
            print(f"LoRA adapter saved: {adapter_path}")
        else:
            print("Warning: Model does not have 'save_lora_adapters' method.")

        if save_merged_model:
            print("Merging LoRA weights and saving merged model...")
            if hasattr(unwrapped_model, 'merge_lora_weights'):
                unwrapped_model.merge_lora_weights()
                merged_model_path = os.path.join(output_dir, 'merged_model.pt')
                atomic_torch_save(
                    {
                        'model_state_dict': unwrapped_model.state_dict(),
                        'config': config.to_dict(),
                        'model_config': describe_model(unwrapped_model),
                        'tokenizer_vocab_size': vocab_size,
                        'merged': True,
                        'foundational_checkpoint': foundational_checkpoint,
                    },
                    merged_model_path
                )
                print(f"Merged model saved: {merged_model_path}")
            else:
                print("Warning: Model does not have 'merge_lora_weights' method.")

    # Compute total training duration (of this invocation, if resumed).
    total_seconds = time.time() - start_time
    
    # Format duration as HH:MM:SS.
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    summary = result['summary']
    summary.update({
        'foundational_checkpoint': foundational_checkpoint,
        'lora_adapter_path': adapter_path,
        'merged_model_path': merged_model_path,
        # Training duration.
        'training_duration_seconds': total_seconds,
        'training_duration_formatted': formatted_time
    })
    write_summary(accelerator, output_dir, summary)

    if accelerator.is_main_process:
        print(f"Training summary saved: {os.path.join(output_dir, 'training_summary.json')}")
        # Print training duration.
        print(f"Total training time: {formatted_time}")

        print("\n" + "="*60)
        print("LoRA instruction fine-tuning completed successfully!")
        print(f"Output directory: {output_dir}")
        print("="*60)

    accelerator.wait_for_everyone()
    return adapter_path
