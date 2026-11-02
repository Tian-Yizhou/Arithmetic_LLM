# Development Log

**Author:** Yizhou Tian

---

## Overview

The original version of this project (`single_gpu/`) was a working single-GPU research prototype with 40+ files in a flat directory. I refactored it into a multi-GPU distributed training system using [HuggingFace Accelerate](https://huggingface.co/docs/accelerate), restructured the codebase into a modular Python package layout, and added early stopping, checkpoint resumption and configurable mixed precision across all four training stages.

Before deleting `single_gpu/`, I audited the refactor against the original code. The audit found defects in the distributed version; they are fixed and the fixes are covered by tests. The sections below describe the final design, and say where a first implementation was wrong. The full record (each defect, its evidence, the fix, and how it was verified) is in [Audit_and_Fix_Plan.md](Audit_and_Fix_Plan.md).

| Dimension | Before (`single_gpu/`) | After |
|---|---|---|
| GPU support | Single device only | Multi-GPU via `accelerate launch` (single-device and DDP) |
| Mixed precision | GRPO had a `GradScaler` code path, but it was never enabled: everything trained in fp32 | `--mixed-precision fp16` (default), `bf16` or `no` in every stage, recorded in the run config |
| Gradient accumulation | Not supported in supervised stages | Configurable in all stages |
| Early stopping | Not implemented | All four stages, the same decision on every process |
| Checkpoint resumption | Checkpoints saved but never restored | Exact resume: weights, optimizer, scheduler, fp16 scaler, per-process RNG, data position, early-stopping state |
| Reproducibility | Unseeded data split and order | One seed controls the split, the batch order and initialization |
| Device management | Manual `.to(device)` everywhere | Fully delegated to Accelerate |
| Evaluation | Sequential on one GPU | Sharded across processes, metrics aggregated |
| Tests | None for training | CPU unit, end-to-end and two-process tests (`tests/`) |
| Project layout | 40+ files in flat directory | Modular packages (`model/`, `training/`, `data/`, `configs/`, `evaluation/`) |

The shared parts of the supervised stages live in `training/supervised_loop.py` (epoch loop, validation, early stopping, checkpointing, resume) and `training/common.py` (Accelerator construction, schedule sizing, checkpoint format, RNG handling).

---

## Table of Contents

1. [Learning-Rate Schedules with a Prepared Scheduler](#1-learning-rate-schedules-with-a-prepared-scheduler)
2. [Distributed Early Stopping Synchronization](#2-distributed-early-stopping-synchronization)
3. [GRPO Mixed Precision Migration](#3-grpo-mixed-precision-migration)
4. [Exact Checkpoint Resumption](#4-exact-checkpoint-resumption)
5. [LoRA State Dict Compatibility in Evaluation](#5-lora-state-dict-compatibility-in-evaluation)
6. [Distributed Checkpoint Portability and File I/O Safety](#6-distributed-checkpoint-portability-and-file-io-safety)
7. [GRPO Data Sharding and Validation](#7-grpo-data-sharding-and-validation)
8. [Training Loop Restructuring for Gradient Accumulation](#8-training-loop-restructuring-for-gradient-accumulation)
9. [Other Fixes](#9-other-fixes)
10. [Testing the Distributed Behaviour](#10-testing-the-distributed-behaviour)

---

## 1. Learning-Rate Schedules with a Prepared Scheduler

**Problem**

The single-GPU version sizes the schedule with `total_steps = len(dataloader) * num_epochs`. After `accelerator.prepare()`, the training dataloader is sharded: on N GPUs, `len(train_dataloader)` is the per-process batch count, about 1/N of the original. Each process performs exactly that many optimizer updates, so computing `total_steps` from the prepared dataloader gives the right number of updates.

My first implementation did that and then **also** passed the scheduler to `accelerator.prepare()`. Accelerate's `AcceleratedScheduler` assumes the opposite convention: that the schedule was sized for a *single* process. By default (`split_batches=False`) it therefore calls the wrapped scheduler's `step()` `num_processes` times per call. Combined, the schedule ran N times too fast: on 4 GPUs, warmup ended after a quarter of its intended length and the learning rate reached 0 after 25% of training and stayed there. The "4× step count" I had seen while debugging was this multiplier at work, not a sign of the original ordering bug.

**Why It Was Hard to Catch**

Nothing crashes, and the loss still decreases for a while. All of my saved runs happened to be single-process, where the multiplier is 1, so they were unaffected.

**Solution**

Keep the schedule in real optimizer updates per process, and do not prepare the scheduler:

```python
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
total_steps = num_update_steps_per_epoch * config.num_epochs
warmup_steps = resolve_warmup_steps(total_steps, config.warmup_steps, config.warmup_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)  # not prepared

...
optimizer.step()
if accelerator.sync_gradients:
    # The Accelerate wrapper would also skip the scheduler when the fp16
    # GradScaler skipped the update (gradient overflow); do that here.
    if optimizer.step_was_skipped:
        skipped_updates += 1
    else:
        scheduler.step()
```

The alternative (size the schedule for one process and let Accelerate step it N times) also works, but then logs and checkpoints would count scheduler steps in a different unit from optimizer updates. `global_step` counts update attempts (accumulation boundaries); the number of fp16-skipped updates is logged separately, so a run with skipped updates can legitimately end with a learning rate slightly above 0.

`--warmup-ratio` (default 0.05) sets the warmup as a fraction of all updates; `--warmup-steps` overrides it with an exact count (including 0). It used to be accepted but ignored.

**Verification:** a real two-process run (torchrun, gloo) with 14 updates per epoch performs 28 updates in 2 epochs and its learning rate reaches 0 exactly at the last one (`tests/test_distributed.py`).

**Takeaway:** In distributed training, check which convention every wrapper assumes. Two individually reasonable choices (per-process step counts, and a wrapper that multiplies by the process count) combined into a silent error.

---

## 2. Distributed Early Stopping Synchronization

**Problem**

Early stopping requires a global decision: "should all processes stop training?" Each process only sees a shard of the validation set. A naive implementation where each process independently decides based on its local shard creates two failure modes:

- **Deadlock**: Process 0's shard has low loss, so it decides to stop. Processes 1-3 continue to the next collective call, but Process 0 never reaches it. The entire job hangs indefinitely.
- **Inconsistent state**: Processes disagree on which epoch produced the "best" model.

**Solution**

Every process computes the same validation number. `evaluate()` gathers each validation sample's summed token loss and token count with `accelerator.gather_for_metrics()`, which also drops the samples a prepared dataloader repeats to give every process the same number of batches. The result is the mean loss per predicted token over the unique validation samples, identical on every process. The same code with the same input produces the same decision everywhere, with no explicit broadcast or voting:

```python
val_loss = evaluate(model, val_dataloader, accelerator)   # identical on all processes

improvement_ratio = (prev_val_loss - val_loss) / abs(prev_val_loss)
if improvement_ratio < config.early_stopping_epsilon:
    patience_counter += 1
    if patience_counter >= config.early_stopping_patience:
        early_stopped = True       # every process sets it in the same epoch
```

I used a **relative improvement ratio** rather than an absolute threshold, so the same `epsilon` works whether the loss is 0.5 or 50.0.

GRPO uses the same scheme on its validation reward rate, which is summed across processes with `accelerator.reduce()`. GRPO validation was never actually wired up in my first version (see [§7](#7-grpo-data-sharding-and-validation)); it now runs every `--eval-every` updates and at the end of every epoch.

`final_model.pt` and `training_summary.json` record the number of epochs actually completed (not the configured number) and whether early stopping ended the run.

**Takeaway:** Any decision that affects training loop control flow (stop/continue, save/skip) must operate on globally aggregated metrics.

---

## 3. GRPO Mixed Precision Migration

**Background**

The single-GPU GRPO trainer contained a manual mixed-precision path (`torch.amp.GradScaler`, explicit `scale()`, `unscale_()`, `step()`, `update()`), but it was controlled by a `use_mixed_precision` flag that defaulted to `False` and that nothing set. In practice the original GRPO trained in fp32.

The refactor removed the manual scaler and delegated mixed precision to Accelerate:

| Manual (before) | Accelerate (after) |
|---|---|
| `scaler.scale(loss).backward()` | `accelerator.backward(loss)` |
| `scaler.unscale_(optimizer)` + `clip_grad_norm_()` | `accelerator.clip_grad_norm_()` |
| `scaler.step(optimizer)` + `scaler.update()` | `optimizer.step()` |

**Problem found in the audit: gradients divided by the loss scale twice**

My first version created and prepared an optimizer in `GRPOTrainer.__init__`, then created and prepared a second one (`reset_optimizer_and_scheduler()`) once the number of steps was known, and a third on resume. Accelerate keeps every optimizer it has prepared, and `accelerator.clip_grad_norm_()` unscales the fp16 gradients once *per prepared optimizer*. With two optimizers over the same parameters, gradients were divided by the loss scale twice (65,536× too small at the initial scale; the exact factor is the current scale); after a resume, three times. AdamW is largely scale-invariant, but at this magnitude its `eps` dominates for small gradients, damping their updates while weight decay stays at full strength.

**Solution**

Create and prepare exactly one optimizer:

```python
trainer = GRPOTrainer(...)                       # prepares the policy model only
trainer.setup_optimizer_and_scheduler(total_steps, warmup_steps)   # called once
if resume_checkpoint:
    trainer.load_checkpoint(resume_checkpoint)   # loads state into the same objects
```

`setup_optimizer_and_scheduler()` refuses to run twice. As in the supervised stages, the scheduler is not prepared and is not advanced when the GradScaler skips an update.

**Verification:** with one prepared optimizer, a GRPO step's gradient norm on Apple MPS is 22.4251 in fp16 and 22.4260 in fp32. A test asserts that exactly one optimizer is prepared, both on a fresh run and after resume.

**Takeaway:** When delegating a concern to a framework, learn its bookkeeping too. Here the framework's state (the list of prepared optimizers) turned an innocent "re-create the optimizer" into a numerical bug.

---

## 4. Exact Checkpoint Resumption

**Problem (Part 1: the original code didn't resume)**

The single-GPU code saved checkpoints with model weights, optimizer state, scheduler state, epoch and step, but no script restored anything except weights. Restarting from a checkpoint kept the weights, but reset the optimizer's moment estimates, restarted the learning-rate schedule and replayed all epochs from the beginning.

**Problem (Part 2: what exact resumption needs)**

My first version restored model, optimizer and scheduler and continued at the saved epoch. The audit showed that this was not enough:

- **Mid-epoch checkpoints skipped data.** A periodic checkpoint stored the epoch in progress, but resume treated it as "epochs completed" and started the next epoch, so the rest of the interrupted epoch was never trained (a 108-update test run ended after 84).
- **The data order was not reproducible.** The train/validation split and the shuffling were unseeded, so a resumed run saw different batches, and on several GPUs each process could even compute a different train/validation split.
- **Other state started over:** fp16 scaler, RNG (dropout), best validation loss, early-stopping counters, training log, and the output directory.
- **GRPO** reset the optimizer and scheduler right after loading them (losing the Adam state and restarting warmup), and loaded the GRPO weights into the *reference* model, moving the KL anchor.

**Solution**

Checkpoints (format version 2) record the exact position and all state needed to continue:

| Field | Meaning |
|---|---|
| `epoch`, `batches_done` | 0-based epoch to continue, and batches of it already trained (0 for end-of-epoch checkpoints) |
| `step` | Optimizer update attempts so far |
| `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` | Adam moments, LR position, fp16 loss scale and growth tracker |
| `training_state` | Best metric, previous metric, patience counter, early-stop flag, training log, running loss of the current epoch |
| `rng_states` | One RNG snapshot per process (Python, NumPy, torch CPU and device) |
| `run_info` | Resolved schedule (world size, batch size, accumulation, total and warmup steps) and data identity (corpus and tokenizer hashes, split sizes, seed; for GRPO also the SFT checkpoint's hash) |

On resume:

```python
train_dataloader.set_epoch(epoch)       # seedable sampler: order = f(seed, epoch)
loader = accelerator.skip_first_batches(train_dataloader, batches_done)
loader.set_epoch(epoch)                 # the new loader's epoch counter starts at 0
```

Each process restores its own RNG snapshot after all setup work. The run continues in the checkpoint's own directory, unless that directory already contains later progress, in which case it writes to a new sibling directory instead of overwriting newer files. A resume whose schedule, data, or (for GRPO) SFT checkpoint differs from the checkpointed run is rejected with an explanation, because the restored scheduler and data position would no longer line up. Checkpoints from before this format still load: the position is derived from the update count and the run is written to a new directory, with a warning that the data order, RNG and early-stopping state cannot be recovered.

For GRPO, the reference model is always the SFT model from `--sft-checkpoint`, and is never overwritten on resume.

Two details mattered for exactness:
- Each `DataLoader` gets its own `torch.Generator`. Otherwise creating an iterator draws a seed from the global RNG, so a resumed run (which creates the iterator at a different point) had a shifted dropout stream.
- NumPy's RNG state is stored as a tensor plus plain numbers, so checkpoints stay loadable with `torch.load`'s default `weights_only=True`.

As for when to load: the optimizer and scheduler must exist before their state can be loaded into them, and the scheduler must be built with the same schedule as the original run. Wrapping the model with DDP does not by itself invalidate optimizer parameter references (DDP uses the same parameter tensors), so the ordering is about having the right objects, not about `prepare()` as such.

**Verification:** resuming from a mid-epoch checkpoint produces **bit-identical** final weights, Adam state and per-epoch losses compared with an uninterrupted run. This holds for the supervised stages and for GRPO, on one process and on two (`tests/test_training_runs.py`, `tests/test_distributed.py`).

**Takeaway:** "Resume" is only well-defined relative to an uninterrupted run. Testing that equivalence directly (same weights, not just "it continued") found three problems that inspecting the code had not.

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

I built `_merge_lora_state_dict()` (now in `training/common.py`) to merge LoRA weights back into base weights at load time:

```python
# W_merged = W_base + (B @ A) * (alpha / rank)
merged = base_weight + (lora_B @ lora_A) * (alpha / rank)
```

And added automatic LoRA detection in the evaluator (`evaluation/model_evaluator.py`):

```python
state_dict = checkpoint.get('model_state_dict', checkpoint)
has_lora_keys = any('.base_layer.' in k for k in state_dict)
if has_lora_keys:
    from training.common import _merge_lora_state_dict
    config_dict = checkpoint.get('config', {})
    state_dict = _merge_lora_state_dict(state_dict, config_dict)
self.model.load_state_dict(state_dict)
```

On a real LoRA checkpoint, the merged plain model's logits match the live LoRA model's within 3e-5.

I also discovered a secondary bug: the instruction LoRA training script's merged model save omitted `model_config` (the architecture parameters like `d_model`, `nhead`, etc.), causing the evaluator to fall back to default dimensions. Fixed by including `model_config` in all checkpoint saves.

A third, unrelated bug compounded the issue: the shell pipeline used `ls -td models/instruction_* | head -1` to find the latest instruction model, but this glob also matched `models/instruction_lora_*`. Since LoRA training runs after instruction training, the LoRA directory was always newest. Fixed with `grep -v lora`:

```bash
INSTRUCTION_DIR=$(ls -td models/instruction_* | grep -v lora | head -n 1)
```

Merging is for *loading weights* (evaluation, or as the starting point of a new run). Resuming *training* of a different kind from a LoRA checkpoint (e.g. full fine-tuning) is not possible: the LoRA optimizer state covers only the adapter parameters, and the loader now says so explicitly.

**Takeaway:** Three independent bugs (LoRA-wrapped state dict keys, missing `model_config` in merged save, shell glob collision) produced the same symptom. Systematic debugging -- checking each assumption one at a time -- was essential to untangling them.

---

## 6. Distributed Checkpoint Portability and File I/O Safety

**Problem**

Three related issues with saving checkpoints in distributed mode:

1. **Portability**: `accelerator.prepare()` wraps the model in DDP, which prepends `module.` to every parameter name. Saving the wrapped model directly produces checkpoints that fail to load in single-GPU or inference contexts.
2. **Race conditions**: When multiple processes execute `os.makedirs()`, `json.dump()`, or `torch.save()` simultaneously, files can be corrupted or partially written.
3. **Name collisions** (found in the audit): the best model was saved as `checkpoint_step_{N}.pt` and then renamed to `best_model.pt`. When a periodic save landed on the same step, the periodic checkpoint was overwritten and then renamed away, so resuming from it failed with `FileNotFoundError`.

**Solution**

One save routine used by every stage:

```python
rng_states = gather_rng_states(accelerator)        # collective: every process contributes
if accelerator.is_main_process:                     # exactly one writer for the whole job
    save_checkpoint(model=accelerator.unwrap_model(model), ..., filename="best_model.pt")
    #   -> torch.save(..., path + ".tmp"); os.replace(path + ".tmp", path)
accelerator.wait_for_everyone()
```

This guarantees:
- Checkpoints contain raw model weights, loadable in any context (single GPU, multi-GPU, or CPU inference).
- One writer per job: `is_main_process` rather than `is_local_main_process`, which would allow one writer per machine on a multi-node run.
- Atomic writes: a crash mid-write never leaves a truncated file under the final name.
- `best_model.pt`, `final_model.pt` and `checkpoint_step_{N}.pt` each have their own name, so no save destroys another.

The guarantee is for single-device and DDP training. FSDP and DeepSpeed shard the model and optimizer state and need their own checkpointing, so the launchers refuse to start under those backends instead of writing incomplete checkpoints.

---

## 7. GRPO Data Sharding and Validation

**Problem**

Standard supervised training uses a `DataLoader`, which Accelerate auto-shards across processes. GRPO iterates over raw prompt-answer pairs and generates multiple candidate responses per prompt on the fly, so there is no `DataLoader` for Accelerate to shard. My first version used strided slicing, `pairs[rank::num_processes]`. The audit found three problems with it:

- When the number of prompts is not divisible by the number of processes, the shares differ in size. One process then runs an extra backward pass, waits forever for its partners in the DDP all-reduce, and the job hangs.
- In `generated` data mode, each process generated its own random prompts, so the "shards" came from different datasets.
- No validation prompts were ever passed to the trainer, so GRPO validation, best-model saving and early stopping never ran (in the original code either), even though `run_exp.sh` enabled `--early-stopping`.

**Solution**

```python
pairs = build_pairs_on_main_process_and_broadcast()        # identical list on every process
random.Random(seed).shuffle(pairs)
val, train = pairs[:val_size], pairs[val_size:]            # --val-size (default 256)
train = train[:len(train) - len(train) % num_processes]    # trim *each* part, after the split
val = val[:len(val) - len(val) % num_processes]
local_train, local_val = train[rank::num_processes], val[rank::num_processes]
```

Too-small datasets are rejected with a clear error instead of producing empty shares. Validation runs every `--eval-every` updates and at the end of each epoch, so short runs also get a best model. Validation samples with a fixed per-process seed inside an isolated RNG context, which makes the metric repeatable and leaves the training random stream untouched (exact resume depends on that). The frozen reference model is kept in eval mode, so the KL penalty compares against a fixed distribution without dropout noise.

The frozen reference model is not passed to `accelerator.prepare()`. It has nothing for DDP to synchronize, and on some backends (multi-CPU) Accelerate wraps every prepared model in DDP, which fails for a model without trainable parameters.

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

Introducing gradient accumulation means multiple micro-batches contribute to a single optimizer step. Operations like gradient clipping, scheduler stepping, and step counting must only fire on the actual update boundary, not on every micro-batch.

**Solution (supervised stages)**

Restructured the loop to use `accelerator.accumulate()` and `sync_gradients`:

```python
for input_ids, attention_mask, labels in train_dataloader:
    with accelerator.accumulate(model):
        loss = compute_loss(model, input_ids, labels)
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        if accelerator.sync_gradients and not optimizer.step_was_skipped:
            scheduler.step()

        optimizer.zero_grad()

    if accelerator.sync_gradients:
        global_step += 1
```

- `accelerator.accumulate(model)` delays all-reduce communication until the accumulation boundary, and forces a boundary at the end of the dataloader, so the last, shorter group of an epoch is also applied.
- `accelerator.sync_gradients` is `True` only on the final micro-batch of each accumulation cycle, gating clip/schedule/count operations.

**Solution (GRPO)**

GRPO iterates over a list of prompt batches, so it groups batches itself:

- Each batch's loss is multiplied by `1 / group_size`, so an update is the mean over its group. The last group of an epoch may be shorter and is still applied (the first version dropped it, so 3 batches × 2 epochs at accumulation 2 gave 2 updates instead of 4).
- The Accelerator is created with `gradient_accumulation_steps=1`, because `accelerator.backward()` divides the loss by that value. The first version divided by the accumulation steps both manually and inside `backward()`, which changed the gradients and the clipping threshold.
- All but the last batch of a group run under `accelerator.no_sync()`, so DDP all-reduces once per update.

This pattern allows flexible batch size scaling without code changes:

| Per-GPU Batch | GPUs | Grad Accum Steps | Effective Batch |
|---|---|---|---|
| 16 | 1 | 1 | 16 |
| 16 | 2 | 1 | 32 |
| 16 | 4 | 2 | 128 |
| 16 | 8 | 4 | 512 |

**Takeaway:** Gradient accumulation is not just "do N forward passes before stepping." Every training loop operation must be classified as either per-micro-batch (loss computation, backward) or per-update (clipping, scheduling, counting), and gated accordingly.

---

## 9. Other Fixes

- **Configuration precedence.** Each launcher resolves its settings before creating the one `Accelerator`: explicit command-line flags override the `--config` JSON file, which overrides the stage defaults. Previously the full-instruction and LoRA launchers created the Accelerator from the command-line accumulation value before reading `--config`, the foundational launcher created a second, unused Accelerator, and LoRA settings in a JSON file were silently replaced by command-line defaults.
- **Recorded precision.** `training_config.json` used to say `"mixed_precision": "no"` while every run used fp16; the recorded value is now the one passed to the Accelerator.
- **`from_json` compatibility.** Old JSON configs with removed fields (e.g. `device`) still load, and a nested `lora_config` is rebuilt as a `LoRAConfig` again.
- **GRPO sequence-length overflow.** Re-encoding a generated sequence adds BOS/EOS tokens, so a sequence of `max_seq_length` generated tokens became 513 tokens and overflowed the position embeddings. Generation is now capped at `max_seq_length - 2` and re-encoded sequences are truncated.
- **Tokenizer/checkpoint vocabulary size.** The GRPO loader and the evaluator keep the checkpoint's embedding size when the tokenizer is smaller, and refuse a larger tokenizer. Equal sizes do not prove identical token ids, so the tokenizer the model was trained with must still be used; the training stages store a hash of the tokenizer and check it on resume.
- **Reproducible evaluation.** `run_evaluation.py` generates its test set with a fixed seed (42), so every process and every run evaluates the same expressions.

---

## 10. Testing the Distributed Behaviour

Every defect above was silent: nothing crashed. So the fixes come with tests that check behaviour against an independent expectation, not just that the code runs. They need no GPU and no prepared data:

```bash
python -m unittest discover -s tests -p "test_*.py" -t .
```

- `tests/test_config_and_utils.py`: configuration precedence, warmup resolution, scheduler behaviour on skipped updates, RNG round trip through a `weights_only` checkpoint, GRPO data splitting.
- `tests/test_grpo_trainer.py`: exactly one prepared optimizer, restored Adam state, unchanged reference model, accumulation groups including the last short one, repeatable validation.
- `tests/test_training_runs.py`: real command-line runs on a tiny model. They check schedule length, checkpoint survival, bit-identical mid-epoch resume (supervised and GRPO), rejection of a changed schedule or a different SFT checkpoint, and early-stopping metadata.
- `tests/test_distributed.py`: real two-process runs (`torchrun`, gloo backend). They check that the schedule is sized per process, that two-process resume is bit-identical, and that GRPO doesn't hang on a prompt count that does not divide evenly.

A note for macOS: with the gloo backend, `torch.distributed.barrier()` needs `PYTORCH_ENABLE_MPS_FALLBACK=1` (the tests set it), and the worker processes can abort in a C++ destructor *after* finishing. This also happens with a five-line script that only uses `torch.distributed`, so the tests check the run's output there instead of its exit code.
