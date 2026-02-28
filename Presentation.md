# Arithmetic LLM -- Presentation Outline

**Duration:** 18 minutes

---

## Slide 1: Title (30 seconds)

**Arithmetic LLM: Building and Scaling a Chain-of-Thought Reasoning Model from Scratch**

Yizhou Tian

---

## Slide 2: What Is This Project? (1 minute)

- A transformer language model built **entirely from scratch** -- tokenizer, model, data pipeline, training loop
- Task: evaluate arithmetic expressions with **step-by-step chain-of-thought reasoning**
- Input/output example:

```
Input:  Evaluate: 8 + ((19 + 17) - (15 - 11))

Output:
  Step 1: 19 + 17 = 36
  Step 2: 15 - 11 = 4
  Step 3: 36 - 4 = 32
  Step 4: 8 + 32 = 40
  Final Result: 40
```

---

## Slide 3: Training Pipeline Overview (1.5 minutes)

Four progressive stages, each building on the last:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. Foundational | Next-token prediction | Learn arithmetic syntax and token patterns |
| 2. Instruction Fine-tuning | Supervised fine-tuning (full-parameter) | Learn to follow prompts and produce structured reasoning |
| 3. LoRA Fine-tuning | Parameter-efficient fine-tuning | Explore efficient adaptation with fewer trainable parameters |
| 4. GRPO | Reinforcement learning with verifiable rewards | Optimize for correctness using reward signals |

- Talking point: Each stage's output becomes the next stage's starting checkpoint

---

## Slide 4: My Contribution -- The Big Picture (2.5 minutes)

Started with a working **single-GPU research prototype** (40+ files in a flat directory). I:

1. **Refactored** the entire codebase for multi-GPU distributed training using HuggingFace Accelerate
2. **Restructured** the project into a modular Python package layout
3. **Added** production features: early stopping, checkpoint resumption, mixed precision
4. **Trained and evaluated** all four model stages end-to-end

| Dimension | Before | After |
|-----------|--------|-------|
| GPU support | Single GPU only | Multi-GPU distributed |
| Mixed precision | Not supported | FP16 (40% memory reduction) |
| Gradient accumulation | Not supported | Configurable |
| Early stopping | Not implemented | Distributed-synchronized |
| Checkpoint resumption | Not supported | Full state restore |
| Codebase | 40+ files, flat directory | Modular packages |

---

## Slide 5: Framework-Level Refactoring (3 minutes)

### What "distributed training" actually required:

Not just adding `accelerate launch` -- every component needed rethinking:

- **Training loop**: Restructured to separate micro-batch steps from optimizer update steps using `accelerator.accumulate()` and `sync_gradients`
- **Device management**: Removed all manual `.to(device)` calls; Accelerate handles placement transparently
- **Validation**: Each GPU sees a different data shard -- used `accelerator.gather()` to aggregate losses globally
- **Checkpointing**: DDP wraps the model with extra layers -- must `unwrap_model()` before saving, gated behind `is_local_main_process` with synchronization barriers
- **GRPO data sharding**: Standard DataLoader auto-shards, but GRPO generates candidates on-the-fly -- implemented manual strided slicing

**Key code comparison** (before vs. after):

```python
# Before: every batch triggers an optimizer step
loss.backward()
optimizer.step()
scheduler.step()

# After: gradient accumulation + distributed sync
with accelerator.accumulate(model):
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(...)
    optimizer.step()
    if accelerator.sync_gradients:
        scheduler.step()
```

---

## Slide 6: Feature Additions (2 minutes)

### Early Stopping

- Monitors validation loss improvement ratio (relative, not absolute -- scale-invariant)
- In distributed setting: all processes must agree on when to stop
- Solution: `accelerator.gather()` returns the same global loss to every process, so the decision is inherently synchronized -- no explicit broadcast needed

### Checkpoint Resumption

- Original code saved optimizer/scheduler states but **never restored them** -- every restart began from epoch 0
- Added `--resume-checkpoint` to all four training scripts
- Restores: model weights + optimizer momentum + scheduler LR position + epoch/step counters

### Mixed Precision

- Enabled FP16 via `Accelerator(mixed_precision="fp16")`
- Zero code changes to training logic; ~40% GPU memory reduction

---

## Slide 7: Technical Challenges (2 minutes)

### Challenge 1: Learning Rate Scheduler Bug

- **Problem**: In single-GPU training, the scheduler is initialized with `total_steps = len(dataloader) * num_epochs`. When `accelerator.prepare()` shards the dataloader across N GPUs, `len(dataloader)` shrinks by a factor of N. If the scheduler is created **before** `prepare()`, it uses the un-sharded (full) dataloader length, inflating `total_steps` by N. The learning rate schedule then decays N times too slowly -- effectively keeping the learning rate near its peak for the entire run.
- **Symptom**: Training loss decreased but was noisy and unstable. The model appeared to learn but converged to a suboptimal point. The bug was hard to catch because the loss curves looked plausible -- just not as good as expected.
- **Fix**: Always create the scheduler **after** `accelerator.prepare()` wraps the dataloader. At that point, `len(dataloader)` reflects the per-process batch count. Additionally, I switched from a fixed warmup step count `1000` to a ratio-based warmup (5% of total steps) so the warmup scales correctly regardless of dataset size or GPU count:

```python
# 1. prepare() first -- this shards the dataloader
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(...)

# 2. Now len(train_dataloader) is per-process, compute update steps correctly
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / accelerator.gradient_accumulation_steps
)
total_steps = num_update_steps_per_epoch * config.num_epochs
warmup_steps = int(total_steps * 0.05)

# 3. Create scheduler with correct counts, then prepare it
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scheduler = accelerator.prepare(scheduler)
```

- **Takeaway**: In distributed training, the sequencing of initialization calls is as important as the calls themselves. This class of bug produces silent numerical errors.

### Challenge 2: Early Stopping Deadlocks

- **Problem**: Early stopping requires a global decision: "should all processes stop training?" In a distributed setting, each process only sees a part of validation set. A naive implementation where each process independently computes validation loss on its local shard and decides whether to stop can lead to two failure modes:
  - **Deadlock**: Process 0 decides to stop (its shard had low loss), but Processes 1-3 continue. At the next `accelerator.gather()` or `wait_for_everyone()`, Process 0 hangs because the others never reach that synchronization point.
  - **Inconsistent state**: Processes disagree on which epoch produced the "best" model, leading to some processes saving a checkpoint while others don't, corrupting the training state.
- **Fix**: I leveraged the fact that `accelerator.gather()` returns the **same** aggregated tensor to every process. Since the validation loss is gathered before the early stopping check, all processes see identical `val_loss` values. The early stopping logic (same code, same input) produces the same decision on every process -- no explicit broadcast or voting needed:

```python
# evaluate() internally calls accelerator.gather() on per-batch losses
# → every process gets the same global average val_loss
val_loss = evaluate(model, val_dataloader, config, accelerator)

# Same val_loss on all processes → same improvement_ratio → same decision
improvement_ratio = (prev_val_loss - val_loss) / abs(prev_val_loss)
if improvement_ratio < config.early_stopping_epsilon:
    patience_counter += 1
    if patience_counter >= config.early_stopping_patience:
        early_stopped = True

accelerator.wait_for_everyone()  # All agree before proceeding
if early_stopped:
    break  # All break simultaneously
```

- **Takeaway**: Any decision that affects training loop control flow (stop/continue, save/skip) must operate on globally aggregated metrics, not local shards. The synchronization is implicit through data, not through explicit coordination messages.

### Challenge 3: GRPO Mixed Precision Conflict

- **Problem**: The original single-GPU GRPO trainer managed mixed precision manually using PyTorch's `GradScaler`. This involved 4 separate API calls per training step:
  1. `scaler.scale(loss).backward()` -- scale loss to prevent FP16 underflow
  2. `scaler.unscale_(optimizer)` -- unscale gradients before clipping
  3. `scaler.step(optimizer)` -- conditionally step (skips if gradients contain NaN/Inf)
  4. `scaler.update()` -- adjust the loss scale factor

  When migrating to Accelerate, these calls conflicted with Accelerate's built-in mixed precision handling. Accelerate internally manages its own scaler, so calling both the manual scaler and Accelerate's methods caused double-scaling (gradients scaled twice) and incorrect gradient values.
- **Fix**: Removed the entire manual `GradScaler` setup and replaced all four calls with Accelerate's unified API:

```python
# Before: ~15 lines of manual mixed precision management
if self.use_mixed_precision:
    self._scaler.scale(scaled_loss).backward()
    self._scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm)
    self._scaler.step(self.optimizer)
    self._scaler.update()
else:
    scaled_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm)
    self.optimizer.step()

# After: 3 lines, Accelerate handles everything
self.accelerator.backward(scaled_loss)
self.accelerator.clip_grad_norm_(self.policy_model.parameters(), max_norm)
self.optimizer.step()
```

- **Takeaway**: When adopting a framework, commit fully to its abstractions. Mixing manual management with framework-managed behavior creates subtle conflicts that are hard to diagnose.

### Challenge 4: Distributed Checkpoint Resumption

- **Problem (part 1 -- the original code didn't truly resume)**: The single-GPU codebase already saved comprehensive checkpoint files containing model weights, optimizer state dicts, scheduler state dicts, epoch numbers, and global step counts. However, none of the training scripts actually **restored** the optimizer, scheduler, or progress metadata when restarting. The `load_checkpoint()` function accepted optional optimizer and scheduler parameters, but no caller ever passed them. Every training run started from epoch 0 and step 0, with a freshly initialized optimizer and scheduler. The checkpoint was only used to initialize model weights for the next pipeline stage (e.g., loading a foundational model before instruction fine-tuning). This meant that if a long training run was interrupted -- due to a crash, timeout, or disk-full error -- all progress was lost. Restarting loaded the model weights but reset the optimizer's momentum buffers to zero, reset the learning rate to its initial value, and replayed all completed epochs from scratch.
- **Fix**: Added `--resume-checkpoint` to all four training entry points. The resume logic follows a careful sequence:

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

  Full state restored on resume:
  - Model weights
  - Optimizer momentum buffers
  - Scheduler learning rate position (without this, the LR restarts from its initial value, causing a sudden spike)
  - Epoch and global step counters (without these, completed epochs are replayed)

- **Takeaway**: True checkpoint resumption means restoring the **full training state**, not just model weights. In distributed settings, the load must happen at the right point in the initialization sequence, and the scheduler must account for the sharded dataloader size. Without proper resumption, interrupted multi-day training runs on expensive GPU clusters become a complete waste of compute.

---

## Slide 8: Results -- Performance Progression (2 minutes)

| Model | Accuracy | Parse Rate | Improvement |
|-------|----------|------------|-------------|
| Foundational (base) | 0.0% | 0.0% | -- |
| Instruction Fine-tuned | 41.4% | 88.8% | +41.4% |
| LoRA Fine-tuned | 43.4% | 83.8% | +2.0% |
| **GRPO (RL-optimized)** | **72.7%** | **76.6%** | **+29.3%** |

- Talking points:
  - Instruction fine-tuning is the critical step -- takes the model from random output to structured reasoning
  - LoRA achieves comparable results with far fewer trainable parameters
  - GRPO provides the largest single accuracy jump (+29.3%) through reward-based optimization
  - Trade-off: GRPO has lower parse rate but much higher accuracy among parsed outputs

### Accuracy by Expression Complexity

| Complexity | Example | GRPO Accuracy |
|------------|---------|---------------|
| Single number | `5` | 100% |
| Simple operation | `6 + 9` | ~95% |
| Moderate nesting | `(a + b) - c` | ~85% |
| Deep nesting (4+ levels) | `((a + b) - (c - d)) + e` | ~40-50% |

---

## Slide 9: Example -- Chain-of-Thought Reasoning (1 minute)

Show a real model output:

```
Expression: 8 + ((((19 + 17) - (15 - 11)) - ((6 - 5) + (17 - 4))) - 7)
Ground Truth: 19

Model Output:
  Step 1: 19 + 17 = 36
  Step 2: 15 - 11 = 4
  Step 3: 36 - 4 = 32
  Step 4: 6 - 5 = 1
  Step 5: 17 - 4 = 13
  Step 6: 1 + 13 = 14
  Step 7: 32 - 14 = 18
  Step 8: 18 - 7 = 11
  Step 9: 8 + 11 = 19

Predicted: 19  (Correct)
```

- Talking point: The model learned to decompose a 9-step nested expression into sequential operations -- this is emergent chain-of-thought behavior, not hard-coded

---

## Slide 10: Project Architecture (1 minute)

```
Arithmetic_LLM/
├── configs/          # Training & LoRA configurations
├── data/             # Tokenizer, data generation, data loading
├── model/            # Transformer architecture, LoRA layers
├── training/         # All training loops (4 stages)
├── evaluation/       # Distributed evaluation & verification
├── tools/            # Interactive solver, diagnostics
├── tests/            # Unit tests
├── run_*.py          # CLI entry points
├── run_exp.sh        # Full pipeline script
└── single_gpu/       # Archived original version
```

- Talking point: From 40+ files in one directory to a clean, modular package structure. Each package has explicit `__init__.py` exports. The entire pipeline runs with one command: `bash run_exp.sh`

---

## Slide 11: Summary and Takeaways (1 minute)

### What I Built
- A complete LLM training system from scratch with four progressive training stages
- Distributed training support across multiple GPUs with production-grade features

### Key Technical Contributions
- Multi-GPU distributed training with HuggingFace Accelerate
- Distributed-synchronized early stopping and checkpoint resumption
- FP16 mixed precision with 40% memory reduction
- Clean, modular codebase architecture

### Key Result
- **72.7% accuracy** on arithmetic evaluation with step-by-step reasoning
- Progressive improvement through foundational -> instruction -> LoRA -> GRPO pipeline

### Lessons Learned
- Initialization order matters in distributed systems (scheduler bug)
- Global consensus is critical for distributed control flow (early stopping)
- Checkpoint portability should be a first-class design concern

---

## Slide 12: Q&A

Questions?

- GitHub: github.com/Tian-Yizhou/Arithmetic_LLM
