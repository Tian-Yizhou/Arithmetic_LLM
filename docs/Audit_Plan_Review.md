# Independent review of the audit and fix plan

Reviewed on 2026-10-30 against commit `a97022f`, the working-tree `Audit_and_Fix_Plan.md`, the original non-`_d` files at `be814c0`, and `single_gpu/`.

**Verdict: the seven numbered defects are supported by the code. The plan is a useful starting point, but should be revised before implementation. Several fixes are incomplete, additional distributed-training defects need coverage, and some statements about verification and saved runs are too strong.** The five smaller items also identify real code conditions, although their severity and descriptions need qualification.

This review changes documentation only. It does not implement fixes, alter Claude's plan, create tags, or delete `single_gpu/`.

The conclusions primarily come from source and history review. Before the request to stop execution, small CPU probes used the installed PyTorch 2.10.0 and Accelerate 1.12.0. Their limited results are recorded below. No successful multi-process training or native GPU mixed-precision validation was completed. No further runtime verification was performed after that request. The plan's MPS experiments are its author's reported evidence, not independently reproduced results from this review.

## Assessment of each existing item

Line references below refer to the code at the audited commit.

| Item | Assessment | Code evidence and required refinement |
|---|---|---|
| **1: repeated FP16 unscaling** | Confirmed; proposed optimizer lifecycle is sound. | `training/grpo_trainer.py:75–98` prepares an optimizer; `:161–187` creates another; `training/train_grpo.py:133,150` invokes that reset on fresh/resumed runs. Keep exactly one prepared optimizer. The extra shrinkage depends on the **current** scaler value; 65,536 is the default initial value, not a constant throughout training. |
| **2: scheduler runs N times too fast** | Confirmed under the default non-split-batch DDP configuration; fix incomplete. | `training/train_foundational.py:461–475`, `training/train_instruction.py:145–161`, and `training/train_instruction_lora.py:187–203` size the schedule using per-process update counts and then prepare it. GRPO sizes it from local prompts. Add explicit skipped-update handling when disabling automatic scheduler coupling; see below. |
| **3: GRPO resume loses state/reference** | Confirmed; fix direction is correct. | `training/train_grpo.py:144–150` restores and then replaces optimizer/scheduler state. `training/grpo_trainer.py:795–798` copies policy checkpoint weights into the reference model. Restore into existing objects; retain the original SFT reference. Validate the identity of the SFT checkpoint, not merely its path. |
| **4: GRPO early stopping never runs** | Confirmed; validation split proposal needs safeguards. | `training/train_grpo.py:155–160` passes no validation data; `training/grpo_trainer.py:574` guards all evaluation and early stopping on its presence. Equal rank workloads are necessary. Seed data generation before shuffling, validate split sizes, and ensure short runs actually evaluate. |
| **5: mid-epoch resume skips data** | Confirmed; exact-resume proposal is incomplete. | `training/train_foundational.py:524` passes `epoch + 1` into periodic saves; `:507,512` treats it as the next zero-based epoch. Instruction/LoRA share `train_epoch()` and the same convention. GRPO also has this mismatch. Add scaler state, rank-specific RNG state, checkpoint schema compatibility, and complete progress metadata. |
| **6: periodic checkpoint disappears** | Confirmed; explicit filenames and atomic replacement are appropriate. | `training/train_foundational.py:559–576`, `training/train_instruction.py:241–256`, and `training/train_instruction_lora.py:285–302` overwrite a step checkpoint and rename it. The GRPO equivalent at `training/grpo_trainer.py:582–592` becomes reachable once validation is wired up; its collision occurs at a coincident save/evaluation step, not necessarily an epoch boundary. |
| **7: ignored warmup argument** | Confirmed; precedence proposal is appropriate. | All stages compute `int(total_steps * 0.05)` instead of using the advertised setting. Update configuration validation to accept `warmup_steps=None`, honor explicit zero, and define behavior when warmup exceeds the schedule. GRPO currently has no separate `training_config.json` writer, so adding that record requires explicit implementation. |
| **M1: hard-coded precision** | Confirmed. | All four launchers request `fp16`; supervised `TrainingConfig.mixed_precision` remains `"no"`. Resolve CLI/config precedence before constructing Accelerator, and record the actual configuration. A requested mode alone does not prove a backend used native FP16. |
| **M2: incorrect completed-epoch metadata** | Confirmed. | Final saves and summaries use `config.num_epochs`, e.g. `training/train_foundational.py:616,633` and `training/grpo_trainer.py:669,693`. Distinguish completed epochs from the current epoch and its processed batches, especially when GRPO stops mid-epoch. |
| **M3: two foundational Accelerators** | Confirmed; a cleanup/correctness risk rather than proof training currently fails. | `run_foundational_training.py:12` constructs one but omits it from the call at `:125`; `training/train_foundational.py:375` constructs another. Use one instance after resolving configuration. |
| **M4: double accumulation division** | Confirmed; severity is understated. | `training/grpo_trainer.py:538` supplies `1 / accum_steps`; `:426–429` applies it before `accelerator.backward()`, which also divides. This changes gradients and clipping behavior; it is not merely misleading metadata. Remove the manual division and fix the incomplete-group behavior below. |
| **M5: nested LoRA deserialization** | Confirmed; proposed reconstruction is correct. | `configs/training_config.py:98–99` is a `pass`. The claim that no CLI path reaches it is incorrect: the supervised launchers accept `--config`. For example, foundational/full instruction training can load a JSON containing a LoRA dictionary and then fail in `to_dict()` at `:116`, which calls `asdict()` on that dictionary. The LoRA launcher separately overwrites the loaded adapter settings with CLI defaults; define that precedence too. |

Issue 1 is supported by Accelerate's implementation: clipping delegates to unscaling, and unspecified optimizers cause it to visit every prepared optimizer. The original code already recreated its optimizer before training, but the retained prepared optimizers and resulting repeated unscaling arise in the Accelerate version. [Accelerate 1.12.0 accelerator source](https://github.com/huggingface/accelerate/blob/v1.12.0/src/accelerate/accelerator.py).

## Required changes to the proposed fixes

### 1. Preserve scheduler behavior when FP16 skips an update

Setting `step_scheduler_with_optimizer=False` removes the process-count multiplier, but also bypasses the wrapper's overflow/skip checks. `sync_gradients=True` means an accumulation boundary was reached; it does not prove the optimizer changed weights. After `optimizer.step()`, advance the scheduler only if the update was not skipped. Apply this to all training paths, including GRPO's manually managed update boundaries. [Accelerate 1.12.0 scheduler source](https://github.com/huggingface/accelerate/blob/v1.12.0/src/accelerate/scheduler.py).

For the supervised loop, the intended condition is:

```python
optimizer.step()
if accelerator.sync_gradients and not optimizer.step_was_skipped:
    scheduler.step()
optimizer.zero_grad()
```

Define whether `global_step` counts attempted or successful updates. The current code counts attempts, even when FP16 skips the update. Preserve a separate processed-batch position for resume either way. If skipped updates occur, a finite epoch-based run can legitimately end before the learning rate reaches zero; the proposed verification must distinguish this from an incorrect schedule.

This matters for the actual saved artifacts: the foundational checkpoint records step 168,750 but scheduler `last_epoch` 168,686; the LoRA checkpoint records step 45,000 but scheduler `last_epoch` 44,985. That is consistent with skipped optimizer updates under the current wrapper. It is not evidence of a scheduler that should be forced to advance through overflows.

### 2. Handle GRPO's last incomplete accumulation group

`GRPOTrainer.train()` updates only when `(batch_idx + 1) % accum_steps == 0` (`training/grpo_trainer.py:547`). There is no final flush. The next epoch starts by zeroing gradients (`:527–528`), discarding the remainder.

For three batches per epoch, accumulation two, and two epochs, the current loop performs **two** updates. The proposed `ceil(3 / 2) * 2` schedule assumes **four**. This behavior was inherited from the original loop, but must be addressed because the new scheduler/resume design depends on it.

Choose and implement a policy: preferably update using the last shorter group, with loss normalization appropriate to its actual size. Alternatively, explicitly drop it and calculate/log that choice consistently. Simply wrapping a Python list in `accelerator.accumulate()` does not give it the prepared dataloader's end-of-epoch handling. Include incomplete groups in checkpoint and early-stop semantics.

### 3. Make data identity consistent across ranks on fresh runs

The unseeded `random_split()` at `data/data_loader.py:308–311` runs independently in each process, before `accelerator.prepare()` in every supervised stage. Accelerate can synchronize sample indices without making the different underlying `Subset.indices` arrays identical. A sample can therefore belong to training on one rank and validation on another, while other training samples are duplicated or omitted globally.

The seeded split proposed under issue 5 is necessary for **fresh multi-GPU correctness**, not just resumption. Use a dedicated split generator with the same seed on all ranks. Keep the sampler seed and epoch deterministic as well.

For GRPO's generated-data mode, seed the generation itself or generate once and distribute the pairs. Shuffling independently generated lists with the same seed does not make those lists equal. Save enough data identity information to reject a resume against a different corpus, generated dataset, tokenizer, or split.

### 4. Complete the resume-state design

The proposed epoch/batch position and seeded sampler are appropriate, but “all training state” requires more than the listed fields:

- **FP16 scaler state**, including its current scale and growth tracker.
- **RNG state for each rank**, including the accelerator device's generator, rather than only one process's CPU generator. Restore it after initialization/setup work that consumes randomness.
- **GRPO state**: best reward, previous reward, patience, generated-pair identity/order, and the original SFT reference identity.
- **Current-epoch statistics**, if resumed epoch loss/logs are intended to match a complete epoch. Skipping batches without restoring accumulated loss/counts makes the reported epoch loss cover only the remaining batches.
- **Resolved schedule/configuration**, including total schedule length, effective warmup, accumulation, and world size. A LambdaLR state does not recreate the captured scheduling function; constructing it with different settings changes the continuation.
- **A checkpoint format version** and an explicit legacy policy. Existing checkpoints use a different meaning for `epoch` and have no `batches_done`; do not silently interpret them as the new format or promise exact recovery of state that was never saved.

Save early-stopping counters and the training log after updating them for the evaluation being checkpointed. Currently best-model saves happen before the early-stopping state update. Normalize end-of-epoch checkpoints to the next epoch with zero batches completed. Handle resumes from already-finished runs without referencing undefined `train_loss`/`val_loss` variables.

Accelerate's checkpoint implementation provides a useful reference for saving the scaler and separate RNG files per process. Its state APIs are an option, although the project still needs its own progress, data, and early-stopping metadata. [Accelerate 1.12.0 checkpoint source](https://github.com/huggingface/accelerate/blob/v1.12.0/src/accelerate/checkpointing.py).

There is also a serialization detail: saving raw `numpy.random.get_state()` in the current `.pt` payload causes the existing default `torch.load()` to reject it under the pinned PyTorch version's weights-only loading behavior. Encode the NumPy state using supported primitives/tensors, or deliberately implement an appropriate trusted-state loading path. Update all affected loaders, including inference loaders if training state remains in model checkpoints.

Continuing in the original output directory is a reasonable default, but distinguish continuation from deliberately branching a run from an older checkpoint. Avoid silently replacing an existing later run's logs/best checkpoint with an older branch.

### 5. Guard the new GRPO validation split

The proposal to equalize rank workloads fixes a real synchronization hazard, but specify these boundary cases:

- Split training and validation first, then enforce the chosen rank-sizing policy on each partition. Trimming the combined list alone does not guarantee both partitions are divisible by world size.
- Reject or explicitly handle datasets too small for the requested holdout or number of ranks. A default holdout of 256 can consume an entire tiny dataset; truncation can leave empty validation shares.
- Evaluate at epoch end as well as the configured step cadence, or otherwise ensure short runs have a validation event. A run shorter than `eval_every` would still have no best model or early-stop observation.
- Use controlled evaluation randomness or deterministic decoding when comparing runs. Patience reduces sensitivity to noisy observations but does not make the metric reproducible.

Keep the reference model in evaluation mode as part of establishing a fixed reference, or explicitly retain and disclose the inherited dropout limitation. A saved SFT pathname alone does not guarantee the same weights if the file is replaced.

### 6. Resolve configuration before constructing Accelerator

The full-instruction and LoRA launchers create Accelerator from `args.gradient_accumulation_steps` **before** loading `--config` (`run_instruction_training.py:53–62`; `run_instruction_training_lora.py:62–71`). A JSON requesting accumulation four can therefore be logged as four while the actual Accelerator uses the CLI default one.

This is an additional functional defect beyond M3. Parse and merge configuration first; define explicit CLI overrides; then construct one Accelerator from the resolved values and save those same values. Apply that rule to the proposed precision and warmup settings as well. Resolve nested LoRA settings consistently instead of silently replacing JSON values with argument defaults.

## Statements that should be corrected or narrowed

**The proposed two-process CPU command does not establish a distributed test.** In this environment, `accelerate launch --num_processes 2 --cpu` produced `world_size=1` and `DistributedType.NO`. The pinned launch code selects its simple launcher for that invocation. A future distributed test must use a launcher/configuration that actually creates multiple workers and assert world size in each worker. No such test is claimed as completed here. [Accelerate 1.12.0 launch source](https://github.com/huggingface/accelerate/blob/v1.12.0/src/accelerate/commands/launch.py).

**“No saved run is affected … learning rate reaches 0 exactly” is not supported as written.** The local foundational log ends at approximately `1.60e-7`; LoRA ends at `3.51e-8`; instruction stops after epoch 18 of a configured 20 at approximately `2.11e-6`. The step counts/scheduler positions are consistent with single-process execution and occasional skipped updates, but logs do not explicitly record world size. They do not establish that every distributed change was harmless. Narrow this to the particular scheduler multiplier issue, qualify inferred process count, and preserve the raw evidence.

**The “verified correct” table should distinguish source inspection from distributed execution.** A single-process run cannot establish that all ranks stop together. The gathered validation scalar supports that control-flow argument, but full resume is not verified by matching learning rates alone. The current validation metric is an average of batch means (`training/train_foundational.py:329–346`), not a token-weighted loss over unique validation examples; prepared loaders may also repeat tail samples. Document that metric or aggregate valid-token loss/counts while accounting for padding/duplicates.

**Checkpoint write claims are too broad.** `is_local_main_process` permits one writer per machine, not one writer for the entire distributed job. It is appropriate for local progress bars; shared checkpoint writes generally need `is_main_process`. GRPO's `save_checkpoint()` has no surrounding barriers, despite the table's blanket statement. Scope the current guarantee to the reviewed single-node DDP path. FSDP/DeepSpeed checkpointing is not established merely by calling `unwrap_model()`; either explicitly exclude those backends or implement their state-handling requirements.

**The development log's DDP explanation needs correction.** `docs/README_DEV.md:177` says loading before `prepare()` necessarily invalidates optimizer parameter references. Ordinary DDP wraps the same model parameters; that is not a general requirement. Loading after preparing the supported objects can be a project convention, but explain the actual constraints instead of this claimed universal rule. Also replace “all progress was lost” with the precise statement that weights could be retained while optimizer/progress state was reset.

**Tokenizer compatibility is only partly handled.** The GRPO loader preserves checkpoint architecture when the tokenizer is smaller, but `evaluation/model_evaluator.py:255` still overwrites checkpoint vocabulary size with the tokenizer's length, which can produce a shape mismatch for that same artifact. Vocabulary size alone also does not establish matching token IDs. Narrow the audit's “verified correct” statement to the GRPO loading case, or extend the compatibility design across consumers.

**Results and pipeline provenance need reconciliation for presentation.** The README reports instruction/LoRA accuracy of 44.3%/46.7%, while `docs/Project_Summary.md` reports 41.4%/43.4%; the README improvement column still uses the latter series. Attribute each row to its checkpoint, evaluation settings, sample count, and source version. The summary dates the 72.7% GRPO result to the original project period; it should not be presented as evidence of the distributed refactor's accuracy. Also, `run_exp.sh` starts LoRA from the foundational checkpoint and GRPO from the full instruction checkpoint: the README's sequential instruction → LoRA → GRPO diagram does not describe that script's dependencies.

**Additional documentation cleanup:** README links currently target Markdown files in the repository root even though they live under `docs/`. Correct those links along with the planned `_d.py` filename updates. Keep quantitative statements such as memory savings and the audit's depth/context percentages only when their measurement conditions and reproducible evidence are supplied.

## Contribution record and baseline preservation

The baseline comparison is substantially right: `single_gpu/` differs from the original non-`_d` files in `grpo_config.py`, `grpo_trainer.py`, and `run_exp.sh`, plus the added `test.sh`. The model, tokenizer, data-processing logic, and LoRA implementation are substantially preserved apart from package imports/documentation. The refactor's main contribution is training infrastructure and evaluation integration.

Useful history references for a professor-facing account:

| Contribution | History anchor | Qualification |
|---|---|---|
| Original implementation and initial distributed work | `be814c0` | This commit already contains both original files and `_d` variants. Refer specifically to its non-`_d` files as the baseline. |
| Package organization and early-stop integration | `8324821`, followed by `0f97a67` | The latter enables early stopping in the pipeline script. GRPO's validation path still needs wiring. |
| GRPO context-length protection | `6758c11` | Preserve as a separate improvement rather than bundling it into distributed training. |
| CLI/state restoration for interrupted training | `7f52901` | A real improvement over the original entry points; distinguish implemented restoration from pending exact-resume fixes. |
| LoRA checkpoint merging during model load | `7057aab` | Enables loading wrapped LoRA weights into a plain architecture; full-optimizer resumption across training types is a different operation. |
| GRPO tokenizer/checkpoint size handling | `5cafad0` | Scope the claim to the implemented loader behavior. |
| Move distributed implementation to root and remove `_d` naming | `075d14b` | Organizational improvement; preserve historical file mapping. |
| LoRA-aware evaluation and merged architecture metadata | `0f54c14` | Separately document checkpoint portability for inference. |
| Instruction/LoRA shell-path collision fix | `a75f5c7` | Makes stage selection unambiguous for the existing naming scheme. |
| Documentation/project restructuring | `a97022f` | Update links affected by moving documentation. |

For eventual deletion, a baseline tag is useful, but the tagged whole tree is not a pure original-only checkout because it also contains `_d` files. Document which files constitute the baseline. Preserve a reference to the pre-removal tree as well if the later `single_gpu/` edits should remain easy to inspect. No tag or deletion is performed by this review.

## Evidence already obtained and remaining validation

The limited CPU probes completed before execution was stopped produced:

| Probe | Observation | Scope |
|---|---|---|
| Prepared-optimizer unscaling | A gradient of 0.625 became approximately `9.54e-6` with two optimizers and `1.46e-10` with three. | CPU GradScaler with the unscale branch explicitly enabled; validates the mechanism, not a native GPU training run. |
| GRPO optimizer lifecycle | 20 Adam state entries restored, then zero after the caller's reset; three prepared optimizers; reference weights equal to the resumed policy. | Actual trainer construction/load/reset methods on a tiny CPU model. |
| Scheduler wrapper | With process count simulated as two, an eight-step schedule reached zero after four calls; disabling coupling removed multiplication but advanced even when the optimizer was marked skipped. | Wrapper behavior with simulated distributed state, not DDP execution. |
| GRPO accumulation remainder | Three batches × two epochs at accumulation two yielded two updates rather than four. | Actual `train()` control flow with rollout/reward computation replaced by a lightweight differentiable step. |
| Double loss division | One microbatch's gradient was 0.25 instead of 0.5 at accumulation two. | Isolated backward-scaling check. |
| NumPy RNG serialization | A checkpoint containing raw NumPy RNG state failed the existing default load with `UnpicklingError`. | Pinned PyTorch 2.10.0. |
| Proposed CPU launcher | One process, world size one. | Does not validate multi-process synchronization. |

The full supervised resume experiment in Claude's plan was not independently completed in this review; its indexing defect is confirmed from the call sites. The multi-process attempt did not produce a successful distributed result. Future GPU validation should cover fresh and resumed execution, actual world size, uneven dataset sizes, incomplete accumulation groups, FP16 skipped updates, and early stopping on every rank. Controlled resume comparisons should inspect batch identities, parameters, optimizer/scaler/scheduler state, and early-stop decisions, rather than only final learning rate or update count.

Recommended implementation order: resolve configuration/data determinism; fix GRPO optimizer/reference lifecycle and accumulation; correct scheduler behavior and warmup; wire validation; introduce atomic versioned checkpoints with complete resume state; correct metadata and documentation; validate on the intended GPU setup; then preserve the historical references and remove `single_gpu/`.
