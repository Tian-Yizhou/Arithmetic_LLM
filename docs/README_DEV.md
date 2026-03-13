# Development Log

**Author:** Yizhou Tian

---

## Overview

The original version of this project (`single_gpu/`) was a working single-GPU research prototype with 40+ files in a flat directory. I refactored it into a production-grade, multi-GPU distributed training system using [HuggingFace Accelerate](https://huggingface.co/docs/accelerate), restructured the codebase into a modular Python package layout, and added production features (early stopping, checkpoint resumption, mixed precision) across all four training stages.

| Dimension | Before (`single_gpu/`) | After |
|---|---|---|
| GPU support | Single device only | Multi-GPU via `accelerate launch` |
| Mixed precision | Not supported | FP16 automatic (~40% memory reduction) |
| Gradient accumulation | Not supported | Configurable via `Accelerator` context manager |
| Early stopping | Not implemented | Distributed-synchronized, scale-invariant |
| Checkpoint resumption | Weights only | Full state restore (model + optimizer + scheduler + epoch/step) |
| Device management | Manual `.to(device)` everywhere | Fully delegated to Accelerate |
| Evaluation | Sequential on one GPU | Sharded across processes, metrics aggregated via `gather()` |
| Project layout | 40+ files in flat directory | Modular packages (`model/`, `training/`, `data/`, `configs/`, `evaluation/`) |

The rest of this document covers the technical challenges I encountered during this refactoring and the solutions I designed.

---

## Table of Contents

1. [Learning Rate Scheduler Initialization Ordering](#1-learning-rate-scheduler-initialization-ordering)
2. [Distributed Early Stopping Synchronization](#2-distributed-early-stopping-synchronization)
3. [GRPO Mixed Precision Migration](#3-grpo-mixed-precision-migration)
4. [Distributed Checkpoint Resumption](#4-distributed-checkpoint-resumption)
5. [LoRA State Dict Compatibility in Evaluation](#5-lora-state-dict-compatibility-in-evaluation)
6. [Distributed Checkpoint Portability and File I/O Safety](#6-distributed-checkpoint-portability-and-file-io-safety)
7. [GRPO Manual Data Sharding](#7-grpo-manual-data-sharding)
8. [Training Loop Restructuring for Gradient Accumulation](#8-training-loop-restructuring-for-gradient-accumulation)

---

## 1. Learning Rate Scheduler Initialization Ordering

**Problem**

In the single-GPU version, the learning rate scheduler is initialized with:

```python
total_steps = len(dataloader) * num_epochs
```

When `accelerator.prepare()` shards the dataloader across N GPUs, `len(dataloader)` shrinks by a factor of N. If the scheduler is created **before** `prepare()`, it uses the un-sharded (full) dataloader length, inflating `total_steps` by N. The learning rate schedule then decays N times too slowly -- effectively keeping the learning rate near its peak for the entire run.

**Why It Was Hard to Catch**

The training loss still decreased, just with more noise and slower convergence. The loss curves looked plausible -- not broken, just subtly worse. There was no crash, no error message, and no obvious sign that the learning rate was wrong. I identified the issue by inspecting the scheduler's internal state mid-training and noticing the step count was 4x higher than expected on a 4-GPU setup.

**Solution**

Always create the scheduler **after** `accelerator.prepare()`, and use a ratio-based warmup so it scales correctly regardless of GPU count or dataset size:

```python
# 1. prepare() shards the dataloader
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

# 2. len(train_dataloader) is now per-process -- compute update steps correctly
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / accelerator.gradient_accumulation_steps
)
total_steps = num_update_steps_per_epoch * config.num_epochs
warmup_steps = int(total_steps * 0.05)  # ratio-based, not a fixed count

# 3. create and prepare scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scheduler = accelerator.prepare(scheduler)
```

**Takeaway:** In distributed training, initialization ordering is a correctness requirement, not a style choice. This class of bug produces silent numerical errors that degrade model quality without any traceback.

---

## 2. Distributed Early Stopping Synchronization

**Problem**

Early stopping requires a global decision: "should all processes stop training?" Each process only sees a shard of the validation set. A naive implementation where each process independently decides based on its local shard creates two failure modes:

- **Deadlock**: Process 0's shard has low loss, so it decides to stop. Processes 1-3 continue to the next `accelerator.gather()` call, but Process 0 never reaches it. The entire job hangs indefinitely.
- **Inconsistent state**: Processes disagree on which epoch produced the "best" model. Some save a checkpoint while others skip it, corrupting the training state.

**Solution**

I leveraged the fact that `accelerator.gather()` returns the **same** aggregated tensor to every process. Since the validation loss is gathered before the early stopping check, all processes see identical `val_loss` values. The same code with the same input produces the same decision on every process -- no explicit broadcast or voting needed:

```python
# evaluate() internally calls accelerator.gather() on per-batch losses
# → every process gets the same global average val_loss
val_loss = evaluate(model, val_dataloader, config, accelerator)

# Same val_loss → same improvement_ratio → same decision on all processes
improvement_ratio = (prev_val_loss - val_loss) / abs(prev_val_loss)
if improvement_ratio < config.early_stopping_epsilon:
    patience_counter += 1
    if patience_counter >= config.early_stopping_patience:
        early_stopped = True

accelerator.wait_for_everyone()
if early_stopped:
    break  # All processes break simultaneously
```

I also used a **relative improvement ratio** rather than an absolute threshold. This makes the early stopping criterion scale-invariant: the same `epsilon` value works whether the loss is 0.5 or 50.0, which matters when different training stages operate at different loss scales.

**Takeaway:** Any decision that affects training loop control flow (stop/continue, save/skip) must operate on globally aggregated metrics. The synchronization here is implicit through data, not through explicit coordination messages -- a simpler and less error-prone pattern.

---

## 3. GRPO Mixed Precision Migration

**Problem**

The single-GPU GRPO trainer managed mixed precision manually with a `torch.cuda.amp.GradScaler`, including explicit `scale()`, `unscale_()`, and `update()` calls:

```python
# Before: manual mixed precision management (15+ lines per training step)
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    log_probs, kl_div = self.compute_log_probs(...)
    loss = self.compute_grpo_loss(log_probs, advantages, kl_div)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

This manual management conflicted with Accelerate's built-in mixed precision: Accelerate wraps the model and optimizer during `prepare()`, and expects to control loss scaling itself. Having two competing scaling mechanisms caused gradient values to be double-scaled or inconsistently unscaled, producing NaN losses.

**Solution**

Removed the entire manual `GradScaler` setup and replaced it with Accelerate's unified API:

```python
# After: Accelerate handles everything
loss = self.compute_grpo_loss(log_probs, advantages, kl_div)
accelerator.backward(loss)
accelerator.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
```

The mapping:
| Manual (before) | Accelerate (after) |
|---|---|
| `scaler.scale(loss).backward()` | `accelerator.backward(loss)` |
| `scaler.unscale_(optimizer)` + `clip_grad_norm_()` | `accelerator.clip_grad_norm_()` |
| `scaler.step(optimizer)` + `scaler.update()` | `optimizer.step()` |

This eliminated ~15 lines of error-prone scaling logic per training step, and also made the GRPO trainer work correctly in both FP16 and FP32 modes without code changes.

**Takeaway:** When migrating to a framework that owns a concern (mixed precision, gradient syncing), fully delegate rather than mixing manual and automated management. Partial delegation creates subtle conflicts that are hard to diagnose.

---

## 4. Distributed Checkpoint Resumption

**Problem (Part 1: The original code didn't truly resume)**

The single-GPU codebase saved comprehensive checkpoint files containing model weights, optimizer state dicts, scheduler state dicts, epoch numbers, and global step counts. However, none of the training scripts actually **restored** the optimizer, scheduler, or progress metadata when restarting. The `load_checkpoint()` function accepted optional optimizer and scheduler parameters, but no caller ever passed them. Every training run started from epoch 0 and step 0, with a freshly initialized optimizer and scheduler.

This meant that if a long training run was interrupted (crash, timeout, disk-full), all progress was lost. Restarting loaded the model weights but reset the optimizer's momentum buffers to zero, reset the learning rate to its initial value, and replayed all completed epochs from scratch.

**Problem (Part 2: Distributed resumption is more complex)**

Even after implementing resume logic, distributed training introduces additional constraints:

- The checkpoint must be loaded **after** `accelerator.prepare()` wraps the model, optimizer, and scheduler. Loading before `prepare()` means the optimizer's internal parameter references don't match the DDP-wrapped model parameters.
- The scheduler must account for the **sharded** dataloader length (post-`prepare()`), not the original length. If `total_steps` is computed from the un-sharded dataloader, the learning rate schedule is wrong on resume -- the same initialization ordering bug from [Challenge 1](#1-learning-rate-scheduler-initialization-ordering).
- The epoch loop must start from the saved epoch, not 0.

**Solution**

Added `--resume-checkpoint` to all four training entry points. The resume logic follows a strict sequence:

```python
# 1. Initialize model, optimizer normally
model = ArithmeticTransformer(**model_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# 2. Let Accelerate wrap everything for distributed training
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

# 3. Compute correct total_steps using the sharded dataloader length
total_steps = math.ceil(len(train_dataloader) / grad_accum_steps) * config.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scheduler = accelerator.prepare(scheduler)

# 4. NOW load the checkpoint -- after prepare(), so state is applied
#    to the wrapped model/optimizer/scheduler
if resume_checkpoint:
    checkpoint = load_checkpoint(
        resume_checkpoint,
        model=accelerator.unwrap_model(model),
        optimizer=optimizer,      # Restores momentum buffers
        scheduler=scheduler       # Restores LR position
    )
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['step']

# 5. Skip completed epochs
for epoch in range(start_epoch, config.num_epochs):
    ...
```

For GRPO (which uses a different trainer architecture), the resume logic also re-initializes the optimizer and scheduler for the remaining steps, ensuring the learning rate schedule aligns correctly with the remaining training duration.

**Takeaway:** True checkpoint resumption means restoring the **full training state**, not just model weights. In distributed settings, the load must happen at the right point in the initialization sequence. Without proper resumption, interrupted multi-day training runs on expensive GPU clusters become a complete waste of compute.

---

## 5. LoRA State Dict Compatibility in Evaluation

**Problem**

The evaluation pipeline loaded checkpoints into a plain `ArithmeticTransformer` model. This worked for foundational and instruction models, but failed for LoRA-trained models with a size mismatch error:

```
RuntimeError: Error(s) in loading state_dict:
  size mismatch for layers.0.self_attn.q_proj.weight:
    copying a param with shape [16, 256] from checkpoint,
    expected shape [256, 256]
```

The root cause: the LoRA training script called `save_checkpoint()` while LoRA layers were still injected. This produced state dict keys like:

```
q_proj.base_layer.weight    # [256, 256] -- the frozen base weights
q_proj.lora_A               # [16, 256]  -- LoRA down-projection
q_proj.lora_B               # [256, 16]  -- LoRA up-projection
```

Instead of the plain `q_proj.weight` that `ArithmeticTransformer` expects.

**Debugging Process**

This took multiple rounds of investigation:

1. Initially suspected `model_config` was missing from the checkpoint, causing wrong model dimensions -- but it was present.
2. Suspected stale `__pycache__` was loading old code -- added debug prints, confirmed the correct code path was hit.
3. Finally examined the actual state dict keys and found the LoRA-wrapped structure.

**Solution**

I built `_merge_lora_state_dict()` in `training/train_foundational.py` to merge LoRA weights back into base weights at load time:

```python
# W_merged = W_base + (B @ A) * (alpha / rank)
merged = base_weight + (lora_B @ lora_A) * (alpha / rank)
```

And added automatic LoRA detection in the evaluator (`evaluation/model_evaluator.py`):

```python
state_dict = checkpoint.get('model_state_dict', checkpoint)
has_lora_keys = any('.base_layer.' in k for k in state_dict)
if has_lora_keys:
    from training.train_foundational import _merge_lora_state_dict
    config_dict = checkpoint.get('config', {})
    state_dict = _merge_lora_state_dict(state_dict, config_dict)
self.model.load_state_dict(state_dict)
```

I also discovered a secondary bug: the instruction LoRA training script's merged model save omitted `model_config` (the architecture parameters like `d_model`, `nhead`, etc.), causing the evaluator to fall back to default dimensions. Fixed by including `model_config` in all checkpoint saves.

A third, unrelated bug compounded the issue: the shell pipeline used `ls -td models/instruction_* | head -1` to find the latest instruction model, but this glob also matched `models/instruction_lora_*`. Since LoRA training runs after instruction training, the LoRA directory was always newest. Fixed with `grep -v lora`:

```bash
INSTRUCTION_DIR=$(ls -td models/instruction_* | grep -v lora | head -n 1)
```

**Takeaway:** Three independent bugs (LoRA-wrapped state dict keys, missing `model_config` in merged save, shell glob collision) produced the same symptom. Systematic debugging -- checking each assumption one at a time -- was essential to untangling them.

---

## 6. Distributed Checkpoint Portability and File I/O Safety

**Problem**

Two related issues with saving checkpoints in distributed mode:

1. **Portability**: `accelerator.prepare()` wraps the model in DDP, which prepends `module.` to every parameter name. Saving the wrapped model directly produces checkpoints that fail to load in single-GPU or inference contexts.
2. **Race conditions**: When multiple processes execute `os.makedirs()`, `json.dump()`, or `torch.save()` simultaneously, files can be corrupted or partially written.

**Solution**

I established a checkpoint save pattern used consistently across all training scripts:

```python
# 1. Barrier: all processes finish the epoch before any starts saving
accelerator.wait_for_everyone()

if accelerator.is_local_main_process:
    # 2. Unwrap: strip DDP wrapper to get plain model weights
    unwrapped_model = accelerator.unwrap_model(model)

    # 3. Save: only main process writes to disk
    save_checkpoint(model=unwrapped_model, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch, step=global_step)

# 4. Barrier: all processes wait until save is complete
accelerator.wait_for_everyone()
```

This guarantees:
- Checkpoints contain raw model weights, loadable in any context (single GPU, multi-GPU, or CPU inference)
- No race conditions or duplicate writes
- No process reads a file that hasn't been fully written yet

---

## 7. GRPO Manual Data Sharding

**Problem**

Standard supervised training uses a `DataLoader`, which Accelerate auto-shards across processes. But GRPO operates differently: it iterates over raw prompt-answer pairs and generates multiple candidate responses per prompt on-the-fly. There is no `DataLoader` for Accelerate to shard.

Without sharding, every GPU processes the same prompts, doing N times the work with no benefit.

**Solution**

Implemented manual strided slicing so each process gets a non-overlapping subset:

```python
pairs = pairs[accelerator.process_index::accelerator.num_processes]
```

With 4 GPUs, process 0 gets indices [0, 4, 8, ...], process 1 gets [1, 5, 9, ...], etc. This distributes the work evenly and ensures each prompt is processed exactly once across all GPUs.

---

## 8. Training Loop Restructuring for Gradient Accumulation

**Problem**

The single-GPU training loop assumed every forward-backward pass triggers an optimizer step:

```python
# Before: single-GPU -- every batch updates weights
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

Introducing gradient accumulation means multiple micro-batches contribute to a single optimizer step. Operations like gradient clipping, scheduler stepping, and step counting must only fire on the actual update boundary, not on every micro-batch. Getting this wrong causes:
- Gradient clipping on partially accumulated gradients (weakening the clip)
- Scheduler advancing N times too fast (learning rate decays prematurely)
- Step counter inflated by the accumulation factor

**Solution**

Restructured the loop to use `accelerator.accumulate()` and `sync_gradients`:

```python
# After: distributed with gradient accumulation
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

- `accelerator.accumulate(model)` delays all-reduce communication until the accumulation boundary
- `accelerator.sync_gradients` is `True` only on the final micro-batch of each accumulation cycle, gating clip/schedule/count operations

This pattern allows flexible batch size scaling without code changes:

| Per-GPU Batch | GPUs | Grad Accum Steps | Effective Batch |
|---|---|---|---|
| 16 | 1 | 1 | 16 |
| 16 | 2 | 1 | 32 |
| 16 | 4 | 2 | 128 |
| 16 | 8 | 4 | 512 |

**Takeaway:** Gradient accumulation is not just "do N forward passes before stepping." Every training loop operation must be classified as either per-micro-batch (loss computation, backward) or per-update (clipping, scheduling, counting), and gated accordingly.
