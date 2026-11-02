# Audit and Fix Record

This document records the problems found while verifying the improvements made on top of the original single-GPU project, how each was fixed, and how each fix was verified. The audit was done before deleting `single_gpu/`, so every change could be compared with the original code.

- **Audited commit:** `a97022f`
- **Baseline (original project):** the non-`_d` files at commit `be814c0`. `single_gpu/` is identical to them apart from four small later edits (`grpo_config.py`, a progress bar in `grpo_trainer.py`, `run_exp.sh`, `test.sh`).
- **Independent review:** the first version of this plan was reviewed by a second assistant (Codex). Its record is [Audit_Plan_Review.md](Audit_Plan_Review.md). It confirmed the seven numbered defects, refined several fixes, and found further defects (R1–R8 below). All of them are incorporated here.
- **Date:** 2026-10-30
- **Status:** all fixes are implemented and verified on CPU (one and two processes) and Apple MPS (fp16). Validation on CUDA GPUs and re-benchmarking the distributed pipeline are still open (see [Remaining work](#remaining-work)).

## How the audit was done

1. Every original file was compared with its counterpart in the restructured project after removing comments and docstrings (both versions were re-printed from their Python AST), so comment-only commits neither hid nor inflated any change. The model, tokenizer, data pipeline and LoRA code are functionally unchanged; all functional changes are in `configs/`, `training/`, the `run_*.py` entry points and `evaluation/model_evaluator.py`.
2. Each functional change was read against the source of Accelerate 1.12.0 and PyTorch 2.10.0, the pinned versions.
3. Behaviour was tested by running the real entry points on a tiny model (d_model 32, 1 layer): on Apple MPS with native fp16, and on CPU in fp32, where results are deterministic and an interrupted-and-resumed run can be compared bit for bit with an uninterrupted one.
4. Multi-process behaviour was tested with real worker processes: `torchrun --nproc_per_node 2` with `ACCELERATE_USE_CPU=true` (gloo backend). Each run reports `Processes: 2`. (`accelerate launch --num_processes 2 --cpu`, proposed in the first version of this plan, starts only one process; see [Corrections](#corrections-to-the-first-version-of-this-plan).)
5. The checks are automated in `tests/` (27 tests, about 30 seconds on CPU):

   ```bash
   python -m unittest discover -s tests -p "test_*.py" -t .
   ```

## Summary

| ID | Problem | Severity | Status | Verified by |
|---|---|---|---|---|
| 1 | GRPO gradients divided by the fp16 loss scale once per extra prepared optimizer | High | Fixed | `test_grpo_trainer`, MPS fp16/fp32 comparison |
| 2 | On N GPUs the learning-rate schedule ends after 1/N of training | High | Fixed | `test_distributed` |
| 3 | GRPO resume discards the restored optimizer and moves the reference model | High | Fixed | `test_grpo_trainer`, `test_training_runs` |
| 4 | GRPO validation, best model and early stopping never run; uneven shares can hang DDP | Medium | Fixed | `test_training_runs`, `test_distributed` |
| 5 | Mid-epoch resume skips the rest of the epoch; resume state incomplete | Medium | Fixed | `test_training_runs`, `test_distributed` |
| 6 | Periodic checkpoints destroyed by the best-model rename | Medium | Fixed | `test_training_runs` |
| 7 | `--warmup-steps` accepted but ignored | Low | Fixed | `test_config_and_utils` |
| R1 | Scheduler must not advance when fp16 skips an update | Medium | Fixed | `test_config_and_utils` |
| R2 | GRPO drops the last, shorter accumulation group of every epoch | Medium | Fixed | `test_grpo_trainer` |
| R3 | Unseeded train/validation split differs between processes on fresh runs | High | Fixed | `test_training_runs`, `test_distributed` |
| R4 | Configuration resolved after the Accelerator is built; JSON LoRA settings overwritten | Medium | Fixed | `test_config_and_utils` |
| R5 | Checkpoint writes: one writer per *machine*, no barriers in GRPO, sharded backends unhandled | Low | Fixed | Code review, two-process runs |
| R6 | Evaluator overwrites the checkpoint's vocabulary size with the tokenizer's | Low | Fixed | Code review, evaluation of new checkpoints |
| R7 | Validation loss is a mean of batch means and counts repeated tail samples | Low | Fixed | Code review |
| R8 | GRPO `generated` mode builds different prompt lists on different processes | Medium | Fixed | Code review |
| I1 | DataLoader iterator draws from the global RNG, so resume was not exact | Low | Fixed | Bit-identical resume tests |
| I2 | Frozen GRPO reference model gets wrapped in DDP on multi-CPU | Medium | Fixed | `test_distributed` |
| M1 | Mixed precision hard-coded to fp16; config records `"no"` | Low | Fixed | Run configs |
| M2 | Final checkpoint and summary record configured, not completed, epochs | Low | Fixed | `test_training_runs` |
| M3 | Foundational launcher builds a second, unused Accelerator | Low | Fixed | Code review |
| M4 | GRPO divides the loss by the accumulation steps twice | Medium | Fixed | `test_grpo_trainer` |
| M5 | `TrainingConfig.from_json` no longer rebuilds `LoRAConfig` | Low | Fixed | `test_config_and_utils` |

**High:** training silently does the wrong thing. **Medium:** a feature does not work as documented, or wrong results in some configurations. **Low:** misleading, or affects robustness only. R = found by the independent review; I = found while implementing and testing the fixes. M4 was raised from Low: the review showed it changes the gradients, not just a log value.

### Where the code now lives

| File | Role |
|---|---|
| `training/common.py` | Accelerator construction, schedule sizing, warm-up, checkpoint format v2, per-process RNG state, data/checkpoint identity, run directories |
| `training/supervised_loop.py` | Epoch loop, validation, early stopping, checkpointing and exact resume shared by the three supervised stages |
| `training/train_foundational.py`, `train_instruction.py`, `train_instruction_lora.py` | Build the model and optimizer for each stage, then call the shared loop |
| `training/grpo_trainer.py`, `training/train_grpo.py` | GRPO trainer and its data preparation |
| `configs/cli.py` | Training flags shared by the supervised launchers, and their precedence |

---

## 1. GRPO gradients divided by the fp16 loss scale

**Problem.** `GRPOTrainer.__init__` created and prepared an optimizer; `train_grpo_model` then called `reset_optimizer_and_scheduler()`, which created and prepared a second one over the same parameters, and a resume prepared a third. Accelerate keeps every prepared optimizer, and `accelerator.clip_grad_norm_()` unscales the fp16 gradients once per prepared optimizer (its source marks this with a TODO). With two optimizers the gradients were divided by the loss scale twice. The extra factor is the *current* scale: 65,536 at the start, smaller once the scaler has backed off. AdamW is largely scale-invariant, but at this magnitude its `eps=1e-8` dominates for parameters with small gradients, damping their updates while weight decay is not damped.

The original GRPO trainer did not have this problem: its own `GradScaler` path unscaled a single optimizer, and it was never enabled (`use_mixed_precision` defaulted to `False`), so the original GRPO trained in fp32.

**Evidence.** On MPS, the clipped gradient norm was 0.621 with one prepared optimizer and 9.48e-6 with two (exactly 65,536×). The review reproduced this on CPU with the unscale path forced: 0.625 became 9.54e-6 with two optimizers and 1.46e-10 with three. In the real trainer, 2 optimizers were prepared on a fresh run and 3 after resume.

**Fix (implemented).** `GRPOTrainer.__init__` prepares only the policy model. `setup_optimizer_and_scheduler(total_steps, warmup_steps)` creates and prepares the one optimizer, is called once by `train_grpo_model`, and raises if called again. `load_checkpoint()` loads state into the existing objects.

**Verification.**
- `test_grpo_trainer`: exactly one prepared optimizer on a fresh trainer and after resume.
- MPS: one GRPO step gives a gradient norm of 22.4260 in fp32 and 22.4251 in fp16 (loss scale 256). At the default initial scale of 65,536 the same step overflows; the scaler skips that update and backs off, as designed (R1).

## 2. On N GPUs the learning-rate schedule ends after 1/N of training

**Problem.** All stages computed `total_steps` from the dataloader *after* `accelerator.prepare()` (correctly, per process), and then also passed the scheduler to `accelerator.prepare()`. With `split_batches=False` (the default), `AcceleratedScheduler.step()` steps the wrapped scheduler `num_processes` times per call, because it assumes a schedule sized for one process. On 4 GPUs, warmup ended 4× too early and the learning rate reached 0 after 25% of training. GRPO had the same problem. `docs/README_DEV.md` §1 presented this ordering as the fix for a scheduler bug.

**Fix (implemented).** The schedule stays in real optimizer updates per process, and the scheduler is **not** prepared. The plan's first proposal, `step_scheduler_with_optimizer=False`, would also have removed the multiplier, but the review pointed out that it also bypasses the wrapper's check for skipped fp16 updates. The loops therefore step the scheduler themselves with that check (R1), and the wrapper adds nothing. `training/common.py: step_scheduler_if_updated()`.

**Verification.** `test_distributed`: a real two-process run with 14 updates per epoch per process performs 28 updates in 2 epochs, and the scheduler is at position 28 with learning rate 0 at the end.

## 3. GRPO resume discards the restored optimizer and moves the reference model

**Problem.**
- `train_grpo_model` called `trainer.load_checkpoint()`, which restored the optimizer and scheduler, and then `reset_optimizer_and_scheduler(remaining_steps)`, which replaced both: Adam moments lost, a second warm-up, and a third prepared optimizer (issue 1).
- `load_checkpoint()` loaded the GRPO weights into the reference model too, so after a resume the KL penalty anchored the policy to itself instead of to the SFT model. This part was inherited from the original code.

**Evidence.** After resume, the live optimizer held no Adam state. The review reproduced this: 20 Adam state entries restored, then 0 after the reset.

**Fix (implemented).**
- `train_grpo_model` computes the full schedule, calls `setup_optimizer_and_scheduler()` once, then `load_checkpoint()`, and never resets.
- The reference model always comes from `--sft-checkpoint` and is never touched by `load_checkpoint()`. GRPO checkpoints store the SFT checkpoint's path and **SHA-256**. A resume with a different SFT file is rejected, because a path alone does not prove the same weights.
- GRPO resumes at the exact batch position (issue 5).

**Verification.**
- `test_grpo_trainer`: after resume, one prepared optimizer, Adam state present, scheduler at the saved position, policy weights equal to the checkpoint, reference weights equal to the SFT model, and the reference model in eval mode.
- `test_training_runs`: GRPO resumed from a mid-epoch checkpoint ends with **bit-identical** weights to the uninterrupted run, and a resume with a modified SFT checkpoint is rejected. Also bit-identical on two processes.

## 4. GRPO validation, best model and early stopping never run

**Problem.** Validation, best-model saving and early stopping in `GRPOTrainer.train()` only ran `if val_dataloader is not None`, and no validation set was ever passed (in the original code either). `--early-stopping` in `run_exp.sh` therefore had no effect, and no GRPO `best_model.pt` was ever written; the saved GRPO runs contain only `checkpoint_step_*.pt` and `final_model.pt`. Related hazard: prompts were split with `pairs[rank::num_processes]`, so with a prompt count not divisible by the number of processes, one process ran an extra backward pass and the DDP collectives fell out of step.

**Fix (implemented).** `training/train_grpo.py: split_and_shard_pairs()`, with the review's safeguards:
- `--val-size` (default 256) and `--seed`. The prompt list is built once on the main process and broadcast (R8), shuffled with the seed, split into validation and training, and **each part** is trimmed to a multiple of the number of processes after the split. Too-small datasets are rejected with an explanatory error.
- Validation runs every `--eval-every` updates **and** at the end of every epoch, so even short runs get a best model and early-stopping observations.
- Validation sampling uses a fixed per-process seed inside an isolated RNG context: the metric is repeatable, and the training random stream is unchanged (exact resume depends on that).
- The best model is saved directly as `best_model.pt`, after the early-stopping update (issue 6).
- The reference model is kept in eval mode, so the KL penalty is computed against a fixed distribution without dropout. This was listed as an inherited issue in the first version of this plan; the review asked for it as part of a fixed reference.

**Verification.**
- `test_training_runs`: validation runs at updates 4, 8, 9 (end of epoch 1), 12, 16 and 18 with `--eval-every 4`, and `best_model.pt` exists. Early stopping triggers at update 8 with patience 1, and the summary records `early_stopped: true` and 0 completed epochs.
- `test_distributed`: two processes with 60 prompts and 5 held out (neither part divisible by 2) give 54 training and 4 validation prompts, 27 and 2 per process, and finish.
- `test_grpo_trainer`: validation is repeatable and leaves the training RNG untouched.

## 5. Mid-epoch resume skips the rest of the epoch; resume state incomplete

**Problem.** Periodic checkpoints stored the epoch in progress, but resume treated it as "epochs completed" and started the next epoch. Resume also restarted the fp16 scaler, the RNG, the best validation loss, the early-stopping counters and the training log, and wrote to a new output directory.

**Evidence.** In a 4-epoch tiny run with 27 updates per epoch (108 in total), resuming from `checkpoint_step_30.pt` ended at update 84 with a learning rate of 2.3e-4 instead of 0. Resuming from an end-of-epoch checkpoint was already correct.

**Fix (implemented).** Checkpoint format version 2 (`training/common.py`, module docstring), with the review's additions:
- **Position:** `epoch` (0-based epoch to continue), `batches_done` (batches of it already trained; 0 at epoch end) and `step` (update attempts).
- **State:** optimizer, scheduler, **fp16 scaler** (scale and growth tracker), **one RNG snapshot per process** (Python, NumPy, torch CPU and device; restored after all setup), and `training_state`. The last holds the best metric, previous metric, patience counter, early-stop flag, training log and the **running loss of the current epoch**, so a resumed epoch reports the loss over all its batches.
- **Resolved schedule and data identity** (`run_info`): processes, batch size, accumulation, batches and updates per epoch, total and warm-up updates, learning rate; corpus and tokenizer SHA-256, split sizes and seed. A resume that differs is rejected with the list of differences, because a `LambdaLR` state does not carry its schedule function and the data position would not line up.
- **Data order:** `DataLoaderConfiguration(use_seedable_sampler=True, data_seed=seed)`, `set_epoch(epoch)` on every epoch, a dedicated generator for `random_split` (R3) and for each DataLoader (I1). On resume, `accelerator.skip_first_batches()`, followed by `set_epoch(epoch)` on the new loader, whose epoch counter starts at 0.
- **Ordering:** early-stopping counters and the log are updated *before* the best model is saved. End-of-epoch checkpoints are normalized to the next epoch with 0 batches done. Resuming an already finished run (completed or early-stopped) does nothing and no longer references undefined loss variables.
- **Output directory:** the checkpoint's own run directory, unless it already contains later progress (a higher-step checkpoint, or `final_model.pt` when resuming from something else). In that case a new sibling directory `<run>_resumed_step<N>_<timestamp>` is used, so a branch from an older checkpoint never overwrites newer files.
- **Legacy checkpoints** (no `format_version`): weights, optimizer and scheduler are restored; the position is derived from the update count; the run goes to a new directory, with a warning that the data order, RNG and early-stopping state cannot be recovered and the schedule cannot be verified.
- **Serialization:** NumPy's RNG state is stored as a tensor plus primitives, so every checkpoint stays loadable with `torch.load`'s default `weights_only=True`. The review showed that raw NumPy state fails there.

**Verification.**
- `test_training_runs`: resuming from `checkpoint_step_36.pt` (epoch 2, batch 9) gives final weights with a maximum difference of **0.0** from the uninterrupted run and an identical `training_log.json`. A changed number of epochs is rejected.
- `test_distributed`: the same holds for a two-process run resumed mid-epoch.
- Manual runs: the same holds for a supervised resume one batch before the end of an epoch, and for a GRPO resume from the last batch of an epoch, before that epoch's validation. A legacy checkpoint written by the old code resumes to 108 updates and a learning rate of 0.

## 6. Periodic checkpoints destroyed by the best-model rename

**Problem.** The best model was saved as `checkpoint_step_{N}.pt` and then renamed to `best_model.pt`. When a periodic save fell on the same step, the periodic checkpoint was overwritten and renamed away. In GRPO the collision occurs at a coincident save/validation step, which became reachable once validation was wired up (issue 4).

**Evidence.** With `--save-every 27` (the updates per epoch), none of `checkpoint_step_27/54/81/108.pt` survived, and resuming from `checkpoint_step_54.pt` failed with `FileNotFoundError`.

**Fix (implemented).** `save_checkpoint()` takes an explicit `filename`; the best model is written directly as `best_model.pt`. Every checkpoint is written to a temporary file and moved into place with `os.replace()`.

**Verification.** `test_training_runs` uses `--save-every 9` with 27 updates per epoch: all of `checkpoint_step_9 … 81.pt` exist next to `best_model.pt`.

## 7. `--warmup-steps` accepted but ignored

**Problem.** All four entry points accepted `--warmup-steps` (defaults 1000, 500, 500, 100) and the docs described it, but every stage used 5% of the total updates.

**Fix (implemented).** `--warmup-ratio` (default 0.05) and `--warmup-steps` (default unset; overrides the ratio when given; explicit 0 is honoured). `warmup_steps=None` passes validation. A warm-up longer than the whole schedule is rejected. The warm-up actually used is recorded in `run_info.json`, in every checkpoint, and in GRPO's summary.

**Verification.** `test_config_and_utils: test_resolve_warmup_steps`.

## R1. Scheduler must not advance when fp16 skips an update

**Problem.** When gradients overflow, the GradScaler skips the optimizer update. A prepared scheduler notices this; a manually stepped one does not. Setting `step_scheduler_with_optimizer=False` (the first plan) would have advanced the schedule through skipped updates.

**Fix (implemented).** `step_scheduler_if_updated()` advances the scheduler only if `optimizer.step_was_skipped` is false, in all stages. `global_step` counts update attempts; skipped updates are counted separately (`skipped_updates` in the log and summary). A run with skipped updates can therefore legitimately end with a learning rate slightly above 0.

**Verification.** `test_config_and_utils: test_scheduler_not_advanced_on_skipped_update`. The saved February runs match this pattern: each checkpoint's step count is a little ahead of its scheduler position (168,750 vs 168,686 for the foundational run, 18,000 vs 17,997, 45,000 vs 44,985, 8,500 vs 8,496), as expected from occasional skipped updates in single-process fp16 runs.

## R2. GRPO drops the last, shorter accumulation group

**Problem.** `GRPOTrainer.train()` updated only when `(batch_idx + 1) % accum_steps == 0` and zeroed the gradients at the next epoch, so the remainder of every epoch was thrown away (inherited from the original loop). With 3 batches per epoch, accumulation 2 and 2 epochs, it performed 2 updates instead of 4.

**Fix (implemented).** The last group of an epoch is applied. Each batch's loss is multiplied by `1 / group_size`, using the group's actual size, so every update is the mean over its group. All but the last batch of a group run under `accelerator.no_sync()`. Checkpoints are written only at group boundaries.

**Verification.** `test_grpo_trainer: test_last_short_accumulation_group_is_applied`: 4 updates, loss scales `[0.5, 0.5, 1.0]` per epoch, and gradient synchronization only on the last batch of each group.

## R3. Unseeded train/validation split differs between processes

**Problem.** `random_split()` in `data/data_loader.py` was unseeded and ran independently on every process before `prepare()`. Accelerate synchronizes sample *indices*, not the underlying `Subset.indices`, so on several GPUs a sample could be in the training set on one process and in the validation set on another, while other samples were duplicated or never used. This is a fresh-run correctness problem, not only a resume problem.

**Fix (implemented).** `create_dataloaders(..., seed)` uses `torch.Generator().manual_seed(seed)` for the split. The same seed, the corpus hash and the split sizes are part of the data identity checked on resume.

**Verification.** The two-process runs in `test_distributed` train on the same split on both processes and resume bit-identically.

## R4. Configuration resolved after the Accelerator is built

**Problem.** The full-instruction and LoRA launchers built the Accelerator from `args.gradient_accumulation_steps` *before* loading `--config`. A JSON file requesting accumulation 4 was logged as 4 while the Accelerator used 1. The LoRA launcher also replaced any `lora_config` in the JSON with command-line defaults.

**Fix (implemented).** `configs/cli.py` and `TrainingConfig.resolve()`: training flags default to `None`, and the precedence is explicit flag > `--config` JSON > stage default. LoRA settings follow the same rule. Only then is the single Accelerator built (`training/common.py: build_accelerator`), from the resolved values that are also saved.

**Verification.** `test_config_and_utils: test_precedence_cli_over_json_over_defaults`.

## R5. Checkpoint writes and supported backends

**Problem.** Files were written under `is_local_main_process`, which allows one writer per machine on a multi-node job. GRPO's `save_checkpoint()` had no barrier. `unwrap_model()` plus a single state dict is not a correct checkpoint under FSDP or DeepSpeed.

**Fix (implemented).** All writes use `is_main_process`. GRPO saves are collective (every process contributes its RNG state), followed by a barrier. Writes are atomic. `build_accelerator()` refuses FSDP, DeepSpeed and other sharded backends with an explanation; single-device and DDP (multi-GPU, multi-CPU) are supported.

## R6. Evaluator overwrites the checkpoint's vocabulary size

**Problem.** `evaluation/model_evaluator.py` rebuilt the model with `len(tokenizer)` as the vocabulary size, ignoring the checkpoint's own. This is a shape mismatch for checkpoints whose embedding is larger than the tokenizer, a case the GRPO loader explicitly supports.

**Fix (implemented).** The checkpoint's `vocab_size` is kept. A smaller tokenizer gives a warning; a larger one is rejected. Equal sizes do not prove identical token ids, so the docs say to use the training tokenizer, and the training stages check the tokenizer's hash on resume.

**Verification.** `run_evaluation.py` loads the new GRPO, LoRA (`best_model.pt`) and merged checkpoints.

## R7. Validation metric

**Problem.** The validation loss was the average of per-batch mean losses. Batches with different numbers of predicted tokens were weighted equally, and the samples that a prepared dataloader repeats to equalize the processes' batch counts were counted twice.

**Fix (implemented).** `evaluate()` gathers each sample's summed token loss and token count with `accelerator.gather_for_metrics()`, which drops the repeated samples, and returns the mean loss per predicted token. Values are therefore not directly comparable with logs from before this change.

## R8. GRPO `generated` mode builds different prompts per process

**Problem.** In `--data-mode generated`, each process generated its own random expressions and then took its strided share, so the shares came from different datasets.

**Fix (implemented).** Prompts are built once on the main process (generated mode is seeded with `--seed`) and broadcast to all processes. A SHA-256 of the prompt list is part of the data identity checked on resume.

## I1. DataLoader iterator consumed the global RNG

**Found while testing issue 5.** After the fixes above, a mid-epoch resume reached the right step and learning rate but its losses differed slightly from the uninterrupted run. Creating a `DataLoader` iterator draws a seed from the global torch RNG. A resumed run creates the iterator after restoring the RNG, the uninterrupted run before the epoch started, so the dropout streams were shifted.

**Fix.** Each DataLoader gets its own `torch.Generator`. After that, resumed and uninterrupted runs are bit-identical.

## I2. Frozen reference model wrapped in DDP on multi-CPU

**Found by the two-process GRPO test.** The first version of this plan listed "Preparing the frozen GRPO reference model with Accelerate" as verified correct, because Accelerate skips the DDP wrapper for a model without trainable parameters. That check exists only in Accelerate's multi-GPU branch. On multi-CPU it wraps every prepared model, and DDP raises an error for a model with no trainable parameters.

**Fix.** The reference model is not prepared; it is moved to the device and run under `accelerator.autocast()`. Nothing about it needs synchronizing.

## Smaller issues (M1–M5)

| ID | Problem | Fix (implemented) |
|---|---|---|
| M1 | Mixed precision was hard-coded to fp16 in every launcher, while `TrainingConfig.mixed_precision` defaulted to `"no"` and was never read; every saved `training_config.json` says `"no"` although training ran in fp16. | `--mixed-precision {fp16,bf16,no}` (default `fp16`) in all launchers, passed to the Accelerator and recorded in the config. `run_info.json` also records the Accelerator's actual mode. Whether a backend uses native fp16 kernels depends on the device: CPU fp16 has no GradScaler. |
| M2 | After early stopping, `final_model.pt` and `training_summary.json` recorded the configured number of epochs. | Summaries record `epochs_completed`, `num_epochs_configured` and `early_stopped`; GRPO also records `batches_done_in_unfinished_epoch`. `final_model.pt` stores the completed position. Verified: early stop after epoch 2 of 6 gives `epochs_completed: 2` and `epoch: 2` in `final_model.pt`. |
| M3 | `run_foundational_training.py` created an Accelerator but didn't pass it on, so `train_foundational_model` created a second one. | One Accelerator per run, built in the launcher after the configuration is resolved; all four training functions require it. |
| M4 | With accumulation, GRPO divided its loss by the accumulation steps manually and again inside `accelerator.backward()`. This changed the gradients and the clipping threshold (the review measured 0.25 instead of 0.5). | GRPO's Accelerator uses `gradient_accumulation_steps=1` (the trainer rejects anything else), and the trainer scales by the actual group size (R2). |
| M5 | `TrainingConfig.from_json()` no longer converted a nested `lora_config` dict into a `LoRAConfig`. The first plan said no command-line path reaches this; that was wrong, because the supervised launchers accept `--config`, and `to_dict()` then failed. | `from_json()` and `resolve()` rebuild `LoRAConfig`; `to_dict()` serializes it. Verified by a round-trip test. |

---

## Corrections to the first version of this plan

The review found statements in the first version that were wrong or too strong:

- **"A 2-process CPU run (`accelerate launch --num_processes 2 --cpu`)".** That command starts a single process (`world_size=1`). The verification now uses `torchrun` with `ACCELERATE_USE_CPU=true`, and the tests check that the run reports 2 processes.
- **"No saved run is affected … learning rate reaches 0 exactly at the last epoch".** Not as written. The saved February 25 logs end at 1.6e-7 (foundational) and 3.5e-8 (LoRA); the instruction run stopped early after epoch 18 of 20 at 2.1e-6. Their step counts and scheduler positions are consistent with **single-process** runs plus a few fp16-skipped updates (R1), but the logs do not record the number of processes. What they support is narrower: those runs are not affected by the multi-GPU scheduler multiplier (issue 2).
- **"Changes verified correct" did not distinguish how.** The table below now says whether each item was checked by reading the source, by a single-process run, or by a multi-process run. A single-process run cannot show that all processes stop together.
- **"Checkpoints unwrapped from DDP and saved between barriers".** Too broad: GRPO's save had no barrier, and `is_local_main_process` allows one writer per machine (R5).
- **"Preparing the frozen GRPO reference model with Accelerate" (verified correct).** Only true on multi-GPU (I2).
- **M4 severity and M5 reachability.** See the table above.
- **"65,536×" in issue 1.** That is the initial loss scale. The actual factor is the current scale.

## Changes verified correct

These parts of the refactor were correct as written; no code change was needed apart from the refactoring into the shared loop.

| Change | How it was verified |
|---|---|
| Accelerate integration in all four stages (`prepare()`, no manual device moves, logging on the main process) | Source review; all modules import; single-process (MPS, CPU) and two-process (CPU) runs complete |
| Gradient accumulation in the supervised stages via `accumulate()` / `sync_gradients` | Source review against Accelerate's documented pattern; two-process run with accumulation 3 (14 batches → 5 updates per epoch) |
| Supervised early stopping: the same decision on every process | Source review (the decision uses the gathered validation loss, identical on all processes); single-process runs trigger it; the two-process runs keep both processes in step |
| Resume from end-of-epoch checkpoints | Single-process run: per-epoch learning rates identical to the uninterrupted run (now also bit-identical weights) |
| LoRA checkpoint auto-merge on load (`_merge_lora_state_dict`) | On a real LoRA checkpoint, logits of the merged plain model match the live LoRA model within 3e-5 |
| `merged_model.pt` stores `model_config`; `grep -v lora` in `run_exp.sh` | Source review |
| GRPO 513-token overflow fix (generation capped at `max_seq_length - 2`, re-encoded sequences truncated) | Source review |
| Tokenizer/checkpoint vocabulary-size handling **in the GRPO loader** | Source review (the evaluator needed the fix in R6) |
| Distributed evaluation (contiguous sharding, summed counts) and the fixed test-set seed (42) | Source review |

## Documentation changes

- `README.md`: the pipeline diagram now matches `run_exp.sh` (LoRA starts from the foundational model; GRPO from the full instruction model). The results table now uses the documented series (41.4% / 43.4% / 72.7%) with its provenance: original single-GPU code, February 1 runs, evaluation settings. The previous 44.3% / 46.7% figures had no saved evaluation behind them and contradicted the table's own improvement column. Other changes: the unmeasured "~40% memory reduction" is removed, the SymPy claim is removed (rewards come from a regex-based verifier), doc links point into `docs/`, the non-existent `evaluation_results/` folder is described as created on first use, and a Tests section is added.
- `docs/README_DEV.md`: §1 rewritten (the scheduler multiplier and its fix); §3 corrected (the original GRPO ran in fp32; the double-unscale bug and its fix); §4 corrected ("weights could be retained while optimizer and progress state were reset" instead of "all progress was lost"; the DDP/`prepare()` explanation; exact resume); §6–8 updated; new §9 (other fixes, including the previously unrecorded 513-token fix, vocabulary handling, evaluation seed and `from_json` compatibility) and §10 (tests). The overview table no longer claims a memory reduction.
- `docs/Detailed_Usage.md`: new flags (`--warmup-ratio`, `--warmup-steps`, `--mixed-precision`, `--seed`, `--val-size`), configuration precedence, GRPO validation and outputs, the evaluation depth note, and a rewritten §5 on resuming. The claim that full training can resume from a LoRA checkpoint is removed: it raises `ValueError`, and the loader now explains why.
- `docs/Replicate Arithmetic_LLM.md`: the seven `_d.py` file names are replaced (they no longer exist since `075d14b`). The GRPO command gets the `--instruction-corpus` argument it requires, and the LoRA merge command uses `python -m model.merge_lora_adapter`.
- `docs/Project_Summary.md`: a provenance note says its results come from the original single-GPU code.
- `run_exp.sh`: GRPO evaluation uses `best_model.pt` when validation produced one.

## Remaining work

1. **CUDA validation.** All verification here used CPU and Apple MPS. Before quoting results, run the tests' scenarios on the intended GPU setup: fresh and resumed multi-GPU runs, uneven dataset sizes, incomplete accumulation groups, fp16 skipped updates, and early stopping on every process. On GPUs, resumed runs should match closely; exact bit-equality is not guaranteed because of non-deterministic kernels.
2. **Re-benchmark the distributed pipeline.** The README's accuracy figures come from the original code. The distributed version (in particular GRPO, which issue 1 affected) has not been evaluated yet.
3. **Baseline and deletion of `single_gpu/`.** Tag the baseline (`git tag original-version be814c0`). That tree also contains the early `_d` files, so the tag message should say that the non-`_d` files are the baseline. Also keep a reference to the last commit that still contains `single_gpu/` (its later edits are listed above). Then delete the folder and update the references to it in `README.md`, `docs/README_DEV.md` and `docs/Detailed_Usage.md`.

## Out of scope: issues inherited from the original

Recorded for completeness; not changed in this pass.

- `run_evaluation.py` defaults to `--max-depth 5`, while all training data uses depth 4. This caps the accuracy that can be measured, and the usage guide now points it out. On the evaluator's default test set (seed 42, 1,000 expressions, operands 1–20, all valid):
  - 59.1% contain an operator at nesting level 4, which the depth-4 training generator never produces;
  - 25.0% need more than 512 tokens for the prompt plus the correct solution (BOS included, measured with the pipeline's tokenizer in `data/tokenizer`), so they cannot be answered within the context.

  An earlier, less precisely defined count in this audit gave 57% and 27%.
- Evaluation samples at temperature 0.8 rather than decoding greedily, so accuracy has sampling noise.
- GRPO rollouts right-pad the prompts, so for shorter prompts in a batch the first generated token is sampled after padding positions.
- The policy's loss forward pass runs in eval mode (no dropout), because candidate generation switches the model to eval mode and it stays there. This matches the original trainer and is now documented in the code.
- `tests/run_evaluator_tests.py` runs two test files that don't exist. The new tests run with `python -m unittest discover`.

## Contribution record

History anchors for the improvements over the original project (adapted from the review):

| Contribution | Commit | Note |
|---|---|---|
| Original implementation; first distributed (`_d`) files | `be814c0` | The baseline is this commit's non-`_d` files |
| GRPO sequence-length (513-token) fix, ratio-based warm-up in GRPO | `6758c11` | |
| Checkpoint resumption entry points (`--resume-checkpoint`) | `7f52901` | Exact resume completed in this audit |
| LoRA checkpoint merging on load | `7057aab` | |
| GRPO tokenizer/checkpoint vocabulary handling | `5cafad0` | |
| Package organization, early stopping | `8324821`, `0f97a67` | GRPO validation wired in this audit |
| Distributed implementation moved to the root, `_d` suffixes dropped | `075d14b` | |
| LoRA-aware evaluation, `model_config` in merged models | `0f54c14` | |
| Instruction/LoRA path collision in `run_exp.sh` | `a75f5c7` | |
| Documentation restructuring | `a97022f` | |
| Audit fixes, shared training loop, exact resume, tests | (this change) | Not yet committed |
