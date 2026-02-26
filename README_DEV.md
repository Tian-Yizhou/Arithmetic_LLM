# Development Log

**Author:** Yizhou Tian

---

## Table of Contents

1. [Summary of Contributions](#summary-of-contributions)
2. [From Single-GPU to Distributed: Framework-Level Refactoring](#from-single-gpu-to-distributed-framework-level-refactoring)
3. [Feature Additions](#feature-additions)
4. [Technical Challenges and Solutions](#technical-challenges-and-solutions)
5. [Code Quality and Architecture](#code-quality-and-architecture)
6. [Key Takeaways](#key-takeaways)

---

## Summary of Contributions

The original version of this project (`single_gpu/`) was a working but single-GPU research prototype with 40+ files in a flat directory. I refactored it into a production-grade, multi-GPU distributed training system using [HuggingFace Accelerate](https://huggingface.co/docs/accelerate), restructured the codebase into a modular Python package layout, and added several features required for real-world training workloads.

### At a Glance

| Dimension | Before (`single_gpu/`) | After (root) |
|---|---|---|
| GPU support | Single device only | Multi-GPU via `accelerate launch` |
| Mixed precision | Not supported | FP16 automatic mixed precision |
| Gradient accumulation | Not supported | Configurable via `Accelerator` context manager |
| Early stopping | Not implemented | Distributed-synchronized, configurable patience and threshold |
| Checkpoint resumption | Not supported | Full state restore (model + optimizer + scheduler + epoch/step) |
| Device management | Manual `.to(device)` scattered across codebase | Fully delegated to Accelerate; device field removed from config |
| Evaluation | Sequential on one GPU | Sharded across processes, metrics aggregated via `accelerator.gather()` |
| Project structure | 40+ files in a flat directory | Modular packages (`model/`, `training/`, `data/`, `configs/`, `evaluation/`, `tools/`, `tests/`) |
| Launch mechanism | `python script.py` | `accelerate launch script.py` (auto-detects hardware) |

---

## From Single-GPU to Distributed: Framework-Level Refactoring

The migration to Accelerate was not a surface-level wrapper -- it required rethinking every part of the training pipeline. The following sections describe the core changes applied across all four training stages (foundational pre-training, instruction fine-tuning, LoRA fine-tuning, and GRPO reinforcement learning).

### Device Management

The single-GPU version had manual device placement calls scattered across every training and evaluation function:

```python
# Before: manual .to() in every function
input_ids = input_ids.to(config.device)
attention_mask = attention_mask.to(config.device)
labels = labels.to(config.device)
model = model.to(config.device)
```

I removed **all** manual `.to(device)` calls and deleted the `device` field from `TrainingConfig` entirely. Accelerate handles placement transparently through `prepare()`, making the code hardware-agnostic -- the same script runs on CPU, single GPU, or multi-GPU without modification.

### Training Loop Restructuring

The single-GPU training loop assumed every forward-backward pass triggers an optimizer step. Introducing distributed training with gradient accumulation required restructuring the loop to distinguish micro-batch steps from true update steps:

**Before (single-GPU):**
```python
for input_ids, attention_mask, labels in train_dataloader:
    input_ids = input_ids.to(config.device)
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

Key design decisions:

- **`accelerator.accumulate(model)`** delays gradient synchronization (all-reduce) across processes until the accumulation boundary, avoiding expensive communication on every micro-batch.
- **`accelerator.sync_gradients`** gates gradient clipping, scheduler stepping, and step counting so they only fire on actual weight updates.
- **`global_step` tracks update steps**, not batch steps, ensuring consistent checkpoint intervals and logging regardless of accumulation settings.

### Validation Loss Aggregation

In single-GPU evaluation, validation loss is computed from all batches locally. In distributed mode, each process only sees a shard of the validation set. I used `accelerator.gather()` to collect per-batch losses across all processes and compute a globally accurate average:

```python
def evaluate(model, val_dataloader, config, accelerator):
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_dataloader:
            loss = nn.functional.cross_entropy(...)
            all_losses = accelerator.gather(loss.reshape(1))
            avg_batch_loss = all_losses.mean().item()
            total_loss += avg_batch_loss
```

Without this, each process would report a different validation loss based on its data shard, leading to noisy metrics and unreliable early stopping decisions.

### Checkpoint Save/Load with DDP Unwrapping

When `accelerator.prepare()` wraps a model for Distributed Data Parallel, it adds wrapper layers. Saving the wrapped model directly produces checkpoints incompatible with single-GPU inference. I implemented a save pattern used across all training scripts:

```python
accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(model=unwrapped_model, ...)
accelerator.wait_for_everyone()
```

This ensures:
- Only the main process writes to disk (no race conditions or duplicate files)
- All processes synchronize before and after (no premature reads)
- Saved checkpoints contain raw model weights, compatible with any deployment context

### GRPO: Manual Data Sharding

Unlike supervised training where Accelerate auto-shards `DataLoader`, GRPO operates on raw prompt-answer pairs and generates candidates on-the-fly. I implemented manual data sharding with strided slicing:

```python
# Each process gets a non-overlapping slice
pairs = pairs[accelerator.process_index::accelerator.num_processes]
```

The GRPO trainer was also refactored to replace:
- `loss.backward()` → `accelerator.backward(scaled_loss)` for correct gradient scaling under mixed precision
- `torch.nn.utils.clip_grad_norm_()` → `accelerator.clip_grad_norm_()` for synchronized clipping
- Manual `GradScaler` → Accelerate's built-in mixed precision management

### Distributed Evaluation (Test-Time Inference)

The single-GPU evaluator ran all test samples sequentially. I built a distributed evaluator that shards the test set across processes and aggregates results:

```python
# Each process evaluates its shard
local_expressions = test_expressions[
    accelerator.process_index::accelerator.num_processes
]
for expression in local_expressions:
    result = model.generate(prompt)

# Aggregate via all-reduce
correct_tensor = torch.tensor([correct], device=accelerator.device)
correct = int(accelerator.reduce(correct_tensor, reduction="sum").item())
```

### Effective Batch Size Scaling

Distributed training combined with gradient accumulation enables flexible batch size scaling without code changes:

| Configuration | Per-GPU Batch | GPUs | Grad Accum Steps | Effective Global Batch |
|---|---|---|---|---|
| Single-GPU baseline | 16 | 1 | 1 | 16 |
| Distributed (2 GPUs) | 16 | 2 | 1 | 32 |
| Distributed (4 GPUs) | 16 | 4 | 2 | 128 |
| Distributed (8 GPUs) | 16 | 8 | 4 | 512 |

---

## Feature Additions

Beyond the distributed migration, I implemented several features that the original single-GPU version lacked.

### Early Stopping

I added a distributed-aware early stopping mechanism across all four training pipelines. This prevents overfitting and saves GPU-hours on expensive multi-GPU runs.

```python
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

The improvement is measured as a **relative ratio** rather than an absolute difference, making the threshold scale-invariant across different loss magnitudes.

Configuration via `TrainingConfig`:

| Parameter | Default | Description |
|---|---|---|
| `early_stopping` | `False` | Enable/disable |
| `early_stopping_patience` | `3` | Epochs to tolerate no improvement |
| `early_stopping_epsilon` | `1e-4` | Minimum relative improvement threshold |

### Checkpoint Resumption

The original codebase saved model weights, optimizer states, and scheduler states in checkpoints, but **never actually restored the optimizer or scheduler** when restarting. Every run began from epoch 0 and step 0 regardless of checkpoint.

I added `--resume-checkpoint` support to all four training entry points. The resume logic:

1. Loads the checkpoint **after** `accelerator.prepare()` wraps the model, optimizer, and scheduler
2. Restores model weights, optimizer momentum buffers, and scheduler learning rate position
3. Extracts saved epoch and step counters from checkpoint metadata
4. Adjusts the epoch loop from `range(config.num_epochs)` to `range(start_epoch, config.num_epochs)`

For GRPO (which uses a different trainer architecture), the resume logic also re-initializes the optimizer and scheduler for the remaining steps, ensuring the learning rate schedule aligns correctly with the remaining training duration.

This means any interrupted training run can be resumed with:

```bash
accelerate launch run_foundational_training.py \
  --corpus-path data/foundational_corpus.txt \
  --tokenizer-path data/tokenizer \
  --resume-checkpoint models/foundational_YYYYMMDD_HHMMSS/checkpoint_step_5000.pt
```

### LoRA Checkpoint Merging for Foundational Resumption

I implemented `_merge_lora_state_dict()` in `training/train_foundational.py`, which merges LoRA adapter weights (A and B matrices) back into the corresponding base weight matrices. This enables resuming foundational-style training from a LoRA-adapted checkpoint without requiring the LoRA infrastructure at load time.

### Training Duration Tracking

Added wall-clock timing to LoRA training with formatted output (HH:MM:SS) and metadata fields (`training_duration_seconds`, `training_duration_formatted`) in the training summary JSON.

---

## Technical Challenges and Solutions

### Challenge 1: Learning Rate Scheduler Miscalculation

**Problem:** The most subtle bug I encountered. In the single-GPU version, the scheduler is initialized with `total_steps = len(dataloader) * num_epochs`. After `accelerator.prepare()` shards the dataloader, `len(dataloader)` reflects per-process batches. If the scheduler is created **before** `prepare()`, `total_steps` is inflated by `num_processes`, causing the learning rate to barely decay.

**Symptom:** Unstable loss curves -- the learning rate stayed near its peak for the entire run.

**Solution:** Always initialize the scheduler **after** `prepare()`, and compute warmup as a ratio (5%) of the corrected total steps:

```python
# Step 1: prepare() shards the dataloader
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

# Step 2: len(train_dataloader) is now per-process
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / accelerator.gradient_accumulation_steps
)
total_steps = num_update_steps_per_epoch * config.num_epochs

# Step 3: ratio-based warmup
warmup_steps = int(total_steps * 0.05)

# Step 4: create and prepare scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scheduler = accelerator.prepare(scheduler)
```

### Challenge 2: Distributed Early Stopping Synchronization

**Problem:** In distributed training, all processes must agree on whether to stop. A naive implementation where each process evaluates its local shard independently could lead to **deadlocks** (one process stops while others wait at a collective operation) or **inconsistent state** (processes disagree on when to save).

**Solution:** I leveraged the fact that `accelerator.gather()` returns the **same** aggregated validation loss to every process. Since all processes see identical `val_loss` values, the early stopping decision is inherently synchronized without an explicit broadcast:

1. `evaluate()` gathers losses via `accelerator.gather()` -- every process gets the same global average
2. Early stopping logic runs identically on all processes (same input → same decision)
3. `accelerator.wait_for_everyone()` ensures all processes reach the epoch boundary
4. `if early_stopped: break` executes simultaneously on all processes

### Challenge 3: GRPO Mixed Precision Migration

**Problem:** The single-GPU GRPO trainer managed mixed precision manually with a `GradScaler`, including explicit `scale()`, `unscale_()`, and `update()` calls. This manual management conflicted with Accelerate's built-in mixed precision handling.

**Solution:** I removed the entire manual `GradScaler` setup and replaced it with Accelerate's unified API:
- `scaler.scale(loss).backward()` → `accelerator.backward(loss)`
- `scaler.unscale_(optimizer)` + manual `clip_grad_norm_` → `accelerator.clip_grad_norm_()`
- `scaler.step(optimizer)` + `scaler.update()` → `optimizer.step()`

This eliminated ~15 lines of error-prone scaling logic per training step.

### Challenge 4: Checkpoint Compatibility Across Contexts

**Problem:** DDP-wrapped models prepend `module.` to every parameter name. Saving without unwrapping creates checkpoints that fail to load in single-GPU or inference contexts.

**Solution:** Established a checkpoint save pattern used consistently across all training scripts: barrier → unwrap → main-process-only save → barrier. This guarantees checkpoints are always portable.

### Challenge 5: Process-Safe File I/O

**Problem:** When multiple processes execute `os.makedirs()`, `json.dump()`, or `torch.save()` simultaneously, file corruption or race conditions can occur.

**Solution:** All file I/O is gated behind `accelerator.is_local_main_process`, with `wait_for_everyone()` barriers before and after to ensure no process reads a file that hasn't been fully written yet.

---

## Code Architecture

### Project Restructuring

The original flat directory with 40+ files was reorganized into a modular Python package structure:

```
single_gpu/                          →    Arithmetic_LLM/
├── transformer_model.py                  ├── model/
├── lora_layer.py                         │   ├── transformer_model.py
├── lora_utils.py                         │   ├── lora_layer.py
├── merge_lora_adapter.py                 │   ├── lora_utils.py
├── arithmetic_tokenizer.py               │   └── merge_lora_adapter.py
├── data_loader.py                        ├── data/
├── generator.py                          │   ├── arithmetic_tokenizer.py
├── training_config.py                    │   ├── data_loader.py
├── train_foundational.py                 │   └── generator.py
├── train_instruction.py                  ├── configs/
├── evaluator.py                          │   └── training_config.py
├── run_foundational_training.py          ├── training/
├── run_evaluation.py                     │   ├── train_foundational.py
└── ... (40+ files total)                 │   └── train_instruction.py
                                          ├── evaluation/
                                          │   └── model_evaluator.py
                                          ├── tools/
                                          ├── tests/
                                          ├── run_foundational_training.py
                                          └── run_evaluation.py
```

Each package has an `__init__.py` with explicit exports, making imports predictable and enabling IDE navigation. The flat layout made it difficult to reason about module dependencies; the package layout makes the architecture immediately visible.

### Configuration Cleanup

- Removed the `device` field and its validation logic from `TrainingConfig` (Accelerate handles it)
- Added `gradient_accumulation_steps`, `mixed_precision`, and early stopping fields
- Updated `from_json()` to filter deprecated fields, so old config files don't cause errors
- Updated `to_dict()` to handle LoRA config serialization

### Robust Method Guards

In the LoRA training script, I added `hasattr()` guards before calling LoRA-specific methods like `save_lora_adapter()` and `merge_and_save()`, preventing crashes if the model object is unexpectedly unwrapped or the LoRA layers are not attached.

### Backward Compatibility

The distributed version is fully backward-compatible with single-GPU execution. Running `accelerate launch` with a single process behaves identically to the original `python` invocation -- no code changes required.

---

## Key Takeaways

1. **Scheduler initialization order matters.** In distributed training, the learning rate scheduler must be created *after* `accelerator.prepare()` wraps the dataloader. Getting this wrong silently inflates the total step count, producing loss curves that look reasonable but converge poorly.

2. **Lean into framework abstractions.** Accelerate's `sync_gradients` flag cleanly separates micro-batch steps from update steps. Gating the scheduler, gradient clipping, and step counter behind this single flag eliminated an entire class of gradient accumulation bugs.

3. **Gather before you decide.** Any decision that affects training loop control flow (early stopping, dynamic hyperparameters) must operate on globally aggregated metrics, not local shards. One process seeing "easy" validation samples would stop prematurely while another trains indefinitely.

4. **Checkpoint portability is a first-class concern.** Always unwrap DDP models before saving. Checkpoints should load cleanly in any context -- single-GPU inference, multi-GPU training, or a different machine entirely.

5. **Guard file I/O in distributed settings.** Multiple processes writing to the same file path causes silent corruption. All writes should be gated behind `is_local_main_process` with synchronization barriers.

6. **Mixed precision is nearly free.** Enabling FP16 via `Accelerator(mixed_precision="fp16")` required zero changes to training logic while reducing GPU memory by ~40%. For transformer models, this should be the default.
