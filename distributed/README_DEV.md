# Development Log of Arithmetic LLM

**Project:** Arithmetic LLM
**Author:** Yizhou Tian
**Status:** Completed & Tested

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Comparison](#architecture-comparison)
3. [Refactoring Challenges with Accelerate](#refactoring-challenges-with-accelerate)
4. [Feature Enhancement: Early Stopping](#feature-enhancement-early-stopping)
5. [Performance & Advantages](#performance--advantages)
6. [Code Walkthrough: Before vs. After](#code-walkthrough-before-vs-after)
7. [Lessons Learned](#lessons-learned)

---

## Overview

This folder contains the **distributed training** version of the Arithmetic LLM project, refactored from a single-GPU research prototype into a production-ready, multi-GPU training pipeline using the [HuggingFace Accelerate](https://huggingface.co/docs/accelerate) framework.

The refactoring touched every stage of the training pipeline — foundational pre-training, instruction fine-tuning, LoRA parameter-efficient fine-tuning, and Group Relative Policy Optimization (GRPO) — while preserving full backward compatibility with single-GPU execution.

### What Changed

| Metric | Single-GPU (`single_gpu/`) | Distributed (`distributed/`) |
|---|---|---|
| File organization | 40+ files in a flat directory | Modular package layout (`model/`, `training/`, `data/`, `configs/`, `evaluation/`, `tools/`, `tests/`) |
| GPU support | Single device only | N GPUs via `accelerate launch` |
| Mixed precision | Not supported | FP16 automatic mixed precision |
| Gradient accumulation | Not supported | Configurable via `Accelerator` context manager |
| Early stopping | Not implemented | Synchronized across all distributed processes |
| Device management | Manual `.to(device)` calls | Fully delegated to `Accelerator` |
| Evaluation | Sequential on one GPU | Test set sharded across processes, metrics reduced with `accelerator.reduce()` |

---

## Architecture Comparison

### Single-GPU Version — Flat Structure

```
single_gpu/
├── transformer_model.py
├── lora_layer.py
├── lora_utils.py
├── arithmetic_tokenizer.py
├── data_loader.py
├── generator.py
├── corpus_generator.py
├── training_config.py
├── lora_config.py
├── grpo_config.py
├── train_foundational.py
├── train_instruction.py
├── train_instruction_lora.py
├── train_grpo.py
├── grpo_trainer.py
├── evaluator.py
├── arithmetic_verifier.py
├── run_foundational_training.py
├── run_instruction_training.py
├── run_instruction_training_lora.py
├── run_grpo_training.py
├── run_evaluation.py
├── merge_lora_adapter.py
├── run_exp.sh
└── ... (40+ files, all in one directory)
```

### Distributed Version — Modular Package Layout

```
distributed/
├── configs/                          # Configuration dataclasses
│   ├── training_config_d.py          # Extended with gradient accumulation & early stopping
│   ├── grpo_config.py
│   └── lora_config.py
├── model/                            # Model architecture (unchanged)
│   ├── transformer_model.py
│   ├── lora_layer.py
│   ├── lora_utils.py
│   └── merge_lora_adapter.py
├── data/                             # Data generation & loading
│   ├── arithmetic_tokenizer.py
│   ├── data_loader.py
│   ├── generator.py
│   ├── corpus_generator.py
│   └── generate_*.py
├── training/                         # Accelerate-aware training modules
│   ├── train_foundational_d.py
│   ├── train_instruction_d.py
│   ├── train_instruction_lora_d.py
│   ├── train_grpo_d.py
│   └── grpo_trainer_d.py
├── evaluation/                       # Distributed evaluation
│   ├── evaluator.py
│   └── evaluator_d.py               # Sharded inference + metric reduction
├── tools/                            # Debugging & diagnostic utilities
│   ├── diagnose_speed.py
│   ├── interactive_solver.py
│   └── check_sequence_lengths.py
├── tests/
│   ├── run_evaluator_tests.py
│   └── test_eos_truncation.py
├── run_*_d.py                        # CLI entry points (accelerate launch)
├── run_exp_d.sh                      # End-to-end pipeline script
└── demo.py / demo.sh                 # Quick demonstration
```

This modular layout follows Python packaging conventions, making the codebase navigable for teams and CI/CD integration.

---

## Refactoring Challenges with Accelerate

### 1. Recalculating Update Steps for Distributed Training

One of the most subtle — and error-prone — aspects of the migration was correctly computing the number of **optimizer update steps** when the data is sharded across multiple GPUs and gradient accumulation is enabled.

**The problem:** In single-GPU training, the learning rate scheduler is initialized with `total_steps = len(dataloader) * num_epochs`. In a distributed setting, `accelerator.prepare(dataloader)` automatically shards the dataloader, so each process sees `len(dataloader) / num_processes` batches per epoch. Additionally, with gradient accumulation, a single optimizer update occurs every `gradient_accumulation_steps` batches. Getting this arithmetic wrong causes the learning rate schedule to either decay prematurely (underfitting) or never finish warming up (unstable early training).

**The solution:** The scheduler must be initialized *after* `accelerator.prepare()` wraps the dataloader, so that `len(train_dataloader)` reflects the per-process batch count:

```python
# train_foundational_d.py, lines 404-427

# Step 1: Prepare model, optimizer, and dataloaders FIRST
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

# Step 2: Now len(train_dataloader) = total_batches / num_processes
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / accelerator.gradient_accumulation_steps
)
total_steps = num_update_steps_per_epoch * config.num_epochs

# Step 3: Dynamic warmup (5% of total steps)
warmup_ratio = 0.05
real_warmup_steps = int(total_steps * warmup_ratio)

# Step 4: Initialize scheduler with correct step counts, then prepare it
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=real_warmup_steps,
    num_training_steps=total_steps
)
scheduler = accelerator.prepare(scheduler)
```

**Why this ordering matters:** If the scheduler is created *before* `prepare()`, `len(train_dataloader)` would reflect the full dataset size, inflating `total_steps` by a factor of `num_processes`. The learning rate would decay far too slowly, effectively keeping it near its peak for the entire training run.

### 2. Gradient Accumulation with `accelerator.accumulate()`

The single-GPU version had no gradient accumulation — every batch triggered an optimizer step. Introducing it required restructuring the training loop to distinguish between "forward-backward steps" (every batch) and "update steps" (every `gradient_accumulation_steps` batches).

**Before (single-GPU):**
```python
for input_ids, attention_mask, labels in train_dataloader:
    loss = compute_loss(model, input_ids, labels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    scheduler.step()
    global_step += 1
```

**After (distributed):**
```python
for input_ids, attention_mask, labels in train_dataloader:
    with accelerator.accumulate(model):
        loss = compute_loss(model, input_ids, labels)
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        if accelerator.sync_gradients:
            scheduler.step()

        optimizer.zero_grad()

    if accelerator.sync_gradients:
        global_step += 1
```

**Key design decisions:**

- **`accelerator.accumulate(model)`** automatically delays gradient synchronization across processes until the accumulation boundary. This avoids expensive all-reduce operations on every micro-batch.
- **`accelerator.sync_gradients`** is a boolean flag that is `True` only when an actual weight update occurs. Gradient clipping and scheduler stepping are gated behind this flag to prevent premature decay.
- **`global_step` tracks update steps**, not batch steps — this ensures checkpoint intervals and logging are consistent regardless of the accumulation setting.

### 3. Distributed Validation Loss: `accelerator.gather()`

In single-GPU evaluation, the validation loss is computed directly from all batches. In distributed mode, each process only sees a shard of the validation set. To compute a globally accurate validation loss, per-batch losses must be gathered across all processes:

```python
# evaluator in train_foundational_d.py, lines 283-291
def evaluate(model, val_dataloader, config, accelerator):
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_dataloader:
            loss = nn.functional.cross_entropy(...)

            # Gather loss scalars from all GPUs into a single tensor
            all_losses = accelerator.gather(loss.reshape(1))
            avg_batch_loss = all_losses.mean().item()

            total_loss += avg_batch_loss
```

Without this gather step, each process would compute a validation loss only on its data shard, leading to noisy and potentially divergent early stopping decisions.

### 4. Checkpoint Unwrapping

When `accelerator.prepare()` wraps a model for Distributed Data Parallel (DDP), it adds wrapper layers around the original `nn.Module`. Saving this wrapped model directly would produce checkpoints that are incompatible with single-GPU inference and would include DDP-specific metadata.

```python
# train_foundational_d.py, lines 231-245
accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(model=unwrapped_model, ...)
accelerator.wait_for_everyone()
```

This pattern ensures:
- Only the main process writes to disk (avoiding race conditions)
- All processes wait at a barrier before and after (avoiding premature reads)
- The saved checkpoint contains the raw `ArithmeticTransformer` state dict, compatible with both single-GPU and distributed loading

### 5. GRPO Training: Manual Data Sharding

Unlike standard supervised training that uses PyTorch `DataLoader` (which Accelerate can auto-shard), GRPO operates on raw prompt-answer pairs and generates candidates on-the-fly. This required manual data sharding:

```python
# train_grpo_d.py, lines 122-126
pairs = pairs[accelerator.process_index::accelerator.num_processes]
```

Each process gets a strided slice of the data, ensuring no overlap. The GRPO trainer was also refactored to:

- Replace `loss.backward()` with `accelerator.backward(scaled_loss)` for correct gradient scaling under mixed precision
- Replace `torch.nn.utils.clip_grad_norm_()` with `accelerator.clip_grad_norm_()` for synchronized clipping
- Replace manual `GradScaler` with `accelerator.autocast()` for cleaner mixed-precision inference

### 6. Device Management: Removing Manual `.to()` Calls

The single-GPU version had manual device placement scattered throughout the code:

```python
# Single-GPU: explicit device management
input_ids = input_ids.to(config.device)
attention_mask = attention_mask.to(config.device)
labels = labels.to(config.device)
model = model.to(config.device)
```

In the distributed version, **all `.to(device)` calls were removed**. The `device` field was also removed from `TrainingConfig`. Accelerate handles device placement transparently through `prepare()`, which makes the code both cleaner and hardware-agnostic — the same script runs on CPU, single GPU, or multi-GPU without any code changes.

---

## Feature Enhancement: Early Stopping

A robust early stopping mechanism was implemented to prevent overfitting and save computational resources. This is especially valuable in distributed settings where GPU-hours are expensive.

### Implementation

```python
# train_foundational_d.py, lines 514-538
if config.early_stopping and prev_val_loss is not None:
    improvement_ratio = (prev_val_loss - val_loss) / abs(prev_val_loss)

    if improvement_ratio < config.early_stopping_epsilon:
        patience_counter += 1
        if patience_counter >= config.early_stopping_patience:
            early_stopped = True
    else:
        patience_counter = 0

accelerator.wait_for_everyone()
if early_stopped:
    break
```

### Distributed Synchronization

Early stopping in a distributed environment is non-trivial — all processes must agree on whether to stop. A naive implementation where each process independently evaluates its local shard could lead to **deadlocks** (one process stops, others wait at a collective operation) or **inconsistent state** (processes disagree on when to save).

This was solved by leveraging the fact that `accelerator.gather()` returns the same aggregated validation loss to every process. Since all processes see identical `val_loss` values, the early stopping decision is inherently synchronized without requiring an explicit broadcast:

1. `evaluate()` gathers loss from all processes via `accelerator.gather()` — every process gets the same global average
2. The early stopping logic runs identically on all processes (same `val_loss`, same `patience_counter`, same decision)
3. `accelerator.wait_for_everyone()` ensures all processes reach the epoch boundary before any process proceeds
4. `if early_stopped: break` executes simultaneously on all processes

### Configuration

Early stopping is controlled by three hyperparameters, added to `TrainingConfig`:

| Parameter | Default | Description |
|---|---|---|
| `early_stopping` | `False` | Enable/disable the mechanism |
| `early_stopping_patience` | `3` | Number of epochs to tolerate no improvement |
| `early_stopping_epsilon` | `1e-4` | Minimum relative improvement threshold |

The improvement is measured as a **relative ratio** rather than an absolute difference, making the threshold scale-invariant across different loss magnitudes.

---

## Performance & Advantages

### Training Throughput

By distributing the workload across N GPUs, the effective training throughput scales near-linearly:

- **Data parallelism:** Each GPU processes `batch_size` samples, yielding a global batch of `batch_size * N` per step
- **FP16 mixed precision:** Reduces memory footprint per GPU by ~40%, enabling larger per-device batch sizes and faster matrix operations on Tensor Cores
- **Gradient accumulation:** Simulates even larger effective batches without additional GPU memory: `effective_batch = batch_size * N * gradient_accumulation_steps`

### Effective Batch Size Scaling

| Configuration | Per-GPU Batch | GPUs | Grad Accum Steps | Effective Global Batch |
|---|---|---|---|---|
| Single-GPU baseline | 16 | 1 | 1 | 16 |
| Distributed (2 GPUs) | 16 | 2 | 1 | 32 |
| Distributed (4 GPUs) | 16 | 4 | 2 | 128 |
| Distributed (8 GPUs) | 16 | 8 | 4 | 512 |

Larger effective batch sizes lead to more stable gradient estimates, enabling higher learning rates and faster convergence.

### Code Quality Improvements

| Aspect | Before | After |
|---|---|---|
| **Directory structure** | Flat (40+ files in root) | Modular packages with `__init__.py` |
| **Device management** | 15+ manual `.to(device)` calls | Zero — fully handled by Accelerate |
| **Mixed precision** | Not available | Single flag: `mixed_precision="fp16"` |
| **Gradient accumulation** | Not available | Context manager: `accelerator.accumulate()` |
| **Logging** | Unrestricted `print()` | Guarded by `accelerator.is_local_main_process` |
| **Checkpoint safety** | Direct `torch.save()` | Barrier + unwrap + main-process-only save |
| **Early stopping** | Not implemented | Distributed-aware with patience & threshold |
| **Launch mechanism** | `python script.py` | `accelerate launch script.py` (auto-detects hardware) |

### Backward Compatibility

The distributed version is fully backward-compatible with single-GPU execution. Running `accelerate launch` with a single process behaves identically to the original `python` invocation — no code changes required.

---

## Code Walkthrough: Before vs. After

### Training Loop

**Single-GPU** (`single_gpu/train_foundational.py`):
```python
# Every batch triggers an optimizer step
for input_ids, attention_mask, labels in train_dataloader:
    input_ids = input_ids.to(config.device)      # Manual device transfer
    attention_mask = attention_mask.to(config.device)
    labels = labels.to(config.device)

    logits = model(inputs, input_attention_mask)
    loss = cross_entropy(logits, targets, ignore_index=-100)

    optimizer.zero_grad()
    loss.backward()                               # Standard backward
    torch.nn.utils.clip_grad_norm_(...)           # Standard clipping
    optimizer.step()
    scheduler.step()                              # Steps every batch
    global_step += 1                              # Increments every batch
```

**Distributed** (`distributed/training/train_foundational_d.py`):
```python
# Optimizer steps only at accumulation boundaries
for input_ids, attention_mask, labels in train_dataloader:
    # No .to(device) — Accelerate handles placement

    with accelerator.accumulate(model):           # Gradient sync control
        logits = model(inputs, input_attention_mask)
        loss = cross_entropy(logits, targets, ignore_index=-100)

        accelerator.backward(loss)                # Distributed backward

        if accelerator.sync_gradients:            # Only at update steps
            accelerator.clip_grad_norm_(...)       # Synchronized clipping

        optimizer.step()
        if accelerator.sync_gradients:
            scheduler.step()                      # Steps per update, not per batch

        optimizer.zero_grad()

    if accelerator.sync_gradients:
        global_step += 1                          # Tracks true update steps
```

### Evaluation

**Single-GPU** (`single_gpu/train_foundational.py`):
```python
def evaluate(model, val_dataloader, config):
    for input_ids, attention_mask, labels in val_dataloader:
        input_ids = input_ids.to(config.device)
        loss = cross_entropy(logits, targets)
        total_loss += loss.item()                 # Local loss only
```

**Distributed** (`distributed/training/train_foundational_d.py`):
```python
def evaluate(model, val_dataloader, config, accelerator):
    for input_ids, attention_mask, labels in val_dataloader:
        loss = cross_entropy(logits, targets)
        all_losses = accelerator.gather(loss.reshape(1))  # Gather from all GPUs
        avg_batch_loss = all_losses.mean().item()          # Global average
        total_loss += avg_batch_loss
```

### Distributed Evaluation (Test-Time Inference)

**Single-GPU** (`single_gpu/evaluator.py`):
```python
# All samples evaluated sequentially on one GPU
for expression in test_expressions:
    result = model.generate(prompt)
    # ... check correctness
```

**Distributed** (`distributed/evaluation/evaluator_d.py`):
```python
# Shard test set across processes
local_expressions = test_expressions[
    accelerator.process_index::accelerator.num_processes
]

# Each process evaluates its shard
for expression in local_expressions:
    result = model.generate(prompt)

# Aggregate metrics via all-reduce
correct_tensor = torch.tensor([correct], device=accelerator.device)
correct = int(accelerator.reduce(correct_tensor, reduction="sum").item())
```

### Pipeline Execution

**Single-GPU** (`single_gpu/run_exp.sh`):
```bash
python run_foundational_training.py \
  --batch-size 16 --learning-rate 1e-4 --num-epochs 5
```

**Distributed** (`distributed/run_exp_d.sh`):
```bash
accelerate launch run_foundational_training_d.py \
  --batch-size 16 --learning-rate 4e-4 --num-epochs 7 \
  --early-stopping --early-stopping-patience 2
```

---

## Lessons Learned

### 1. Scheduler Initialization Order Is Critical

The most subtle bug encountered during the refactoring: initializing the learning rate scheduler *before* `accelerator.prepare()` caused the total step count to be inflated by `num_processes`. The learning rate barely decayed during training, leading to unstable loss curves. The fix was to always initialize the scheduler after `prepare()` wraps the dataloader.

**Takeaway:** In distributed training, the sequencing of initialization calls is as important as the calls themselves.

### 2. `sync_gradients` Is the Key Abstraction

Accelerate's `sync_gradients` flag cleanly separates "micro-batch" steps from "update" steps. Gating the scheduler, gradient clipping, global step counter, and checkpoint saves behind this flag eliminated an entire class of bugs related to gradient accumulation.

**Takeaway:** When adopting a framework, lean into its abstractions rather than reimplementing the logic manually.

### 3. Gather Before You Decide

Early stopping decisions must be based on globally aggregated metrics, not local shards. A process that happens to have "easy" validation samples would stop too early, while one with "hard" samples would train indefinitely. Using `accelerator.gather()` to synchronize validation loss before the early stopping check ensures all processes make the same decision at the same time.

**Takeaway:** Any decision that affects the training loop control flow (early stopping, checkpointing, dynamic hyperparameter adjustments) must operate on gathered metrics.

### 4. Always Unwrap Before Saving

DDP-wrapped models have a `module.` prefix on every parameter name. Saving without unwrapping creates checkpoints that fail to load in non-distributed contexts. The `accelerator.unwrap_model()` + `wait_for_everyone()` + main-process-only save pattern became a reusable idiom across all training scripts.

**Takeaway:** Treat checkpoint compatibility as a first-class concern — models should be loadable in any deployment context.

### 5. Flat Directories Don't Scale

The original 40+ files in a single directory made it difficult to reason about module dependencies and onboard new contributors. Reorganizing into `model/`, `training/`, `data/`, `configs/`, `evaluation/`, and `tools/` packages with `__init__.py` files clarified the architecture and enabled proper Python imports.

**Takeaway:** Code organization is not cosmetic — it directly impacts maintainability and the speed of iteration.

### 6. Mixed Precision Is Nearly Free

Enabling FP16 mixed precision via `Accelerator(mixed_precision="fp16")` required zero changes to the training logic while reducing GPU memory usage by ~40%. Accelerate handles loss scaling, gradient unscaling, and numeric stability internally.

**Takeaway:** For transformer-based models, FP16 mixed precision should be the default unless numeric stability issues are observed.

---

## Running the Distributed Pipeline

### Prerequisites

```bash
pip install accelerate torch
accelerate config  # Configure multi-GPU settings interactively
```

### Full Training Pipeline

```bash
# Launch the end-to-end pipeline (training + evaluation)
bash run_exp_d.sh
```

### Individual Steps

```bash
# Foundational pre-training
accelerate launch run_foundational_training_d.py \
  --corpus-path data/foundational_corpus.txt \
  --tokenizer-path data/tokenizer \
  --output-dir models/ \
  --batch-size 16 --learning-rate 4e-4 --num-epochs 7 \
  --early-stopping --early-stopping-patience 2

# Instruction fine-tuning
accelerate launch run_instruction_training_d.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_*/best_model.pt \
  --batch-size 8 --learning-rate 2e-5 --num-epochs 20 \
  --early-stopping --early-stopping-patience 2

# LoRA fine-tuning
accelerate launch run_instruction_training_lora_d.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_*/best_model.pt \
  --lora-rank 32 --lora-alpha 64 --lora-target-modules attention,feedforward \
  --batch-size 8 --learning-rate 1e-4 --save-merged-model

# GRPO reinforcement learning
accelerate launch run_grpo_training_d.py \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/instruction_*/best_model.pt \
  --output-dir models/grpo \
  --num-candidates 4 --temperature 0.8 --kl-penalty-coef 0.05

# Distributed evaluation
accelerate launch run_evaluation_d.py \
  --model-path models/instruction_*/best_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000 --batch-size 1
```



**Checkpoint Resumption Support for Distributed Training**

The original distributed training codebase saved comprehensive checkpoint files containing model weights, optimizer state dictionaries, scheduler state dictionaries, epoch numbers, and global step counts. However, none of the training pipelines actually utilized the saved optimizer, scheduler, or progress metadata when restarting. Every training run began from epoch 0 and global step 0 regardless of whether a checkpoint was provided. The `load_checkpoint()` function in `train_foundational_d.py` already accepted optional optimizer and scheduler parameters, but these were never passed by any caller — checkpoints were only used to initialize model weights for the next stage (e.g., loading a foundational model before instruction fine-tuning).

To enable training resumption, a `--resume-checkpoint` CLI argument was added to all four training entry points: `run_foundational_training_d.py`, `run_instruction_training_d.py`, r`un_instruction_training_lora_d.py`, and `run_grpo_training_d.py`. Each entry point passes this path through to its corresponding training function.

In these three epoch-based training functions (`train_foundational_d.py`, `train_instruction_d.py`, `train_instruction_lora_d.py`), the resume logic loads the checkpoint after the model, optimizer, and scheduler have been initialized and prepared by Accelerate. It calls `load_checkpoint()` with all three components — model, optimizer, and scheduler — restoring not just the weights but also the optimizer's momentum buffers and the scheduler's learning rate position. The saved epoch and global step are extracted from the checkpoint metadata, and the epoch loop changes from `range(config.num_epochs)` to `range(start_epoch, config.num_epochs)`, so completed epochs are skipped entirely.

For the GRPO pipeline, which has a different architecture using GRPO Trainer, the resume logic calls `trainer.load_checkpoint()` to restore the policy model, reference model, optimizer, and scheduler states. The saved epoch and step values are passed to `trainer.train()` as `start_epoch` and `start_step` parameters. The `train()` method was modified to accept these parameters, initializing `global_step` from `start_step` instead of 0 and starting the epoch loop from `start_epoch`. After loading the checkpoint, the optimizer and scheduler are re-initialized for the remaining steps to ensure the learning rate schedule is correctly aligned with the remaining training duration.

These changes allow any interrupted or completed training run to be resumed from any saved checkpoint by simply adding `--resume-checkpoint <path>` to the launch command. The full training state — including optimizer momentum, learning rate schedule position, and progress counters — is restored, so resumed training produces results equivalent to an uninterrupted run.
