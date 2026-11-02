"""GRPO trainer module."""

from typing import Any, Dict, List, Optional, Tuple

import contextlib
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator

from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from evaluation.arithmetic_verifier import ArithmeticVerifier
from training.common import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    describe_model,
    gather_rng_states,
    get_linear_schedule_with_warmup,
    isolated_rng,
    load_checkpoint,
    step_scheduler_if_updated,
    write_json,
)
from model.transformer_model import ArithmeticTransformer

from configs.grpo_config import GRPOConfig


@dataclass
class GRPOTrainingState:
    """Progress and early-stopping state of a GRPO run, saved in checkpoints.

    Attributes:
        epoch: 0-based epoch in progress (number of completed epochs at the end)
        batches_done: Batches of ``epoch`` already trained
        global_step: Optimizer update attempts
        skipped_updates: Updates the fp16 GradScaler skipped (overflow)
        best_reward_rate: Best validation reward rate so far
        prev_reward_rate: Validation reward rate of the previous evaluation
        patience_counter: Evaluations in a row without sufficient improvement
        early_stopped: Whether early stopping has ended the run
        last_eval_step: global_step of the latest validation (-1: none yet)
        training_log: One entry per training batch
    """
    epoch: int = 0
    batches_done: int = 0
    global_step: int = 0
    skipped_updates: int = 0
    best_reward_rate: Optional[float] = None
    prev_reward_rate: Optional[float] = None
    patience_counter: int = 0
    early_stopped: bool = False
    last_eval_step: int = -1
    training_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "GRPOTrainingState":
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class GRPOTrainer:
    """GRPO trainer for arithmetic LLM."""

    def __init__(
        self,
        config: GRPOConfig,
        sft_checkpoint_path: Optional[str] = None,
        tokenizer: Optional[ArithmeticBPETokenizer] = None,
        tokenizer_path: Optional[str] = None,
        policy_model: Optional[torch.nn.Module] = None,
        reference_model: Optional[torch.nn.Module] = None,
        candidate_sub_batch_size: Optional[int] = None,
        accelerator: Accelerator = None
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Require an Accelerator instance for distributed training.
        if accelerator is None:
            raise ValueError("Accelerator must be provided to GRPOTrainer")
        # GRPO groups batches for gradient accumulation itself (see train()),
        # and accelerator.backward() would divide the loss a second time.
        if accelerator.gradient_accumulation_steps != 1:
            raise ValueError(
                "GRPOTrainer manages gradient accumulation itself: create the "
                "Accelerator with gradient_accumulation_steps=1 and set "
                "GRPOConfig.gradient_accumulation_steps instead."
            )
        self.accelerator = accelerator

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = None
        self.scheduler = None
        self.verifier = ArithmeticVerifier()
        self.candidate_sub_batch_size = candidate_sub_batch_size
        # Mixed precision and its GradScaler are handled by Accelerate.

        if self.tokenizer is None and tokenizer_path is not None:
            self.tokenizer = ArithmeticBPETokenizer()
            self.tokenizer.load(tokenizer_path)

        if self.policy_model is None or self.reference_model is None:
            if sft_checkpoint_path is not None:
                self._load_models_from_checkpoint(sft_checkpoint_path)

        if self.reference_model is not None:
            self._freeze_reference_model()

        # Prepare the policy (DDP wrapping, device placement); the optimizer
        # is created once the number of updates is known
        # (setup_optimizer_and_scheduler). The frozen reference model only
        # needs to be on the device: it has nothing for DDP to synchronize,
        # and on some backends (e.g. multi-CPU) prepare() would wrap it in DDP
        # anyway, which fails for a model without trainable parameters.
        self.policy_model = self.accelerator.prepare(self.policy_model)
        if self.reference_model is not None:
            self.reference_model = self.reference_model.to(self.accelerator.device)

    def _forward_model(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward helper that tolerates models without attention_mask support."""
        if attention_mask is None:
            return model(input_ids)
        try:
            return model(input_ids, attention_mask=attention_mask)
        except TypeError:
            return model(input_ids)

    def _require_generation_components(self) -> None:
        if self.policy_model is None or self.tokenizer is None:
            raise ValueError(
                "policy_model and tokenizer must be provided to generate candidates"
            )

    def _load_models_from_checkpoint(self, checkpoint_path: str) -> None:
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided to load models")

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        vocab_size = len(self.tokenizer.token2id)
        model_config = checkpoint_data.get(
            "model_config",
            {
                "vocab_size": vocab_size,
                "d_model": 256,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 1024,
                "dropout": 0.1,
                "max_seq_length": 512,
            },
        )
        checkpoint_vocab_size = checkpoint_data.get("tokenizer_vocab_size")
        if checkpoint_vocab_size is not None and checkpoint_vocab_size != vocab_size:
            if vocab_size > checkpoint_vocab_size:
                raise ValueError(
                    f"Tokenizer vocab size ({vocab_size}) is larger than checkpoint "
                    f"vocab size ({checkpoint_vocab_size}). Cannot load: "
                    f"embedding weights would be missing for new tokens."
                )
            # Tokenizer is smaller — use checkpoint's vocab_size so the model
            # architecture matches the saved weights. Extra embedding rows are
            # simply unused at inference time.
            if self.accelerator.is_local_main_process:
                print(
                    f"Warning: tokenizer vocab size ({vocab_size}) < checkpoint "
                    f"vocab size ({checkpoint_vocab_size}). Using checkpoint's "
                    f"vocab_size for model architecture."
                )
        self.policy_model = ArithmeticTransformer(**model_config)
        self.reference_model = ArithmeticTransformer(**model_config)

        load_checkpoint(checkpoint_path=checkpoint_path, model=self.policy_model)
        load_checkpoint(checkpoint_path=checkpoint_path, model=self.reference_model)

    def setup_optimizer_and_scheduler(self, total_steps: int, warmup_steps: int) -> None:
        """Create and prepare the optimizer, and create the LR scheduler.

        Call exactly once. Accelerate keeps every optimizer it has prepared,
        and clip_grad_norm_() unscales fp16 gradients once per prepared
        optimizer, so a second optimizer over the same parameters would shrink
        the gradients by the loss scale. Resuming loads state into these
        objects instead of creating new ones.

        The scheduler is not prepared: a prepared scheduler steps
        num_processes times per call, while total_steps here counts this
        process's (= every process's) optimizer updates.
        """
        if self.policy_model is None:
            raise ValueError("policy_model must be initialized")
        if self.optimizer is not None:
            raise RuntimeError("setup_optimizer_and_scheduler() must only be called once")
        
        # Get parameters from the unwrapped model.
        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
        
        optimizer = torch.optim.AdamW(
            unwrapped_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        self.optimizer = self.accelerator.prepare(optimizer)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps)
        )

    def memory_usage_estimate(
        self,
        batch_size: int,
        num_candidates: int,
        max_gen_length: int
    ) -> Dict[str, int]:
        """Return rough memory usage estimates in bytes."""
        if self.policy_model is None:
            return {"parameter_bytes": 0, "activation_bytes": 0, "total_bytes": 0}
        param_bytes = sum(p.numel() * p.element_size() for p in self.policy_model.parameters())
        activation_bytes = batch_size * num_candidates * max_gen_length * 4
        return {
            "parameter_bytes": int(param_bytes),
            "activation_bytes": int(activation_bytes),
            "total_bytes": int(param_bytes + activation_bytes),
        }

    def _freeze_reference_model(self) -> None:
        for param in self.reference_model.parameters():
            param.requires_grad = False
        # A fixed reference: no dropout when computing the KL penalty.
        self.reference_model.eval()

    def compute_group_statistics(
        self,
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute group mean and standard deviation."""
        group_mean = torch.mean(rewards, dim=1)
        group_std = torch.stack(
            [torch.std(row, unbiased=False) for row in rewards.unbind(dim=0)]
        )
        return group_mean, group_std

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages."""
        group_mean, group_std = self.compute_group_statistics(rewards)
        group_mean = group_mean.unsqueeze(1)
        group_std = group_std.unsqueeze(1)
        return (rewards - group_mean) / (group_std + self.config.advantage_epsilon)

    def normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Normalize advantages to zero mean within each group."""
        mean = torch.mean(advantages, dim=1, keepdim=True)
        return advantages - mean

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy gradient loss."""
        if log_probs.shape != advantages.shape:
            raise ValueError(
                "log_probs and advantages must have the same shape, got "
                f"{log_probs.shape} and {advantages.shape}"
            )
        return -torch.mean(advantages * log_probs)

    def compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference logits."""
        if policy_logits.shape != reference_logits.shape:
            raise ValueError(
                "policy_logits and reference_logits must have the same shape, got "
                f"{policy_logits.shape} and {reference_logits.shape}"
            )
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        policy_probs = torch.softmax(policy_logits, dim=-1)
        kl = policy_probs * (policy_log_probs - reference_log_probs)
        kl_sum = torch.sum(kl, dim=-1)
        return torch.mean(kl_sum)

    def compute_total_loss(
        self,
        policy_loss: torch.Tensor,
        kl_divergence: torch.Tensor,
        kl_penalty_coef: Optional[float] = None
    ) -> torch.Tensor:
        """Compute total loss including KL penalty."""
        coef = self.config.kl_penalty_coef if kl_penalty_coef is None else kl_penalty_coef
        policy_loss = policy_loss.float()
        kl_divergence = kl_divergence.float()
        coef_tensor = torch.tensor(
            coef, device=kl_divergence.device, dtype=kl_divergence.dtype
        )
        return policy_loss + coef_tensor * kl_divergence

    def train_step(
        self,
        prompts: List[str],
        ground_truth: List[int],
        do_step: bool = True,
        loss_scale: float = 1.0,
        sync_gradients: bool = True
    ) -> dict:
        """Execute single GRPO training step.

        Args:
            do_step: Run the optimizer step right after the backward pass
            loss_scale: Factor applied to the loss before backward, e.g.
                1 / (number of batches in the accumulation group)
            sync_gradients: False skips the DDP gradient all-reduce for this
                backward (all but the last batch of an accumulation group)
        """
        if len(prompts) != len(ground_truth):
            raise ValueError("prompts and ground_truth must have the same length")

        if self.policy_model is None or self.reference_model is None:
            raise ValueError("policy_model and reference_model must be initialized")
        if self.optimizer is None:
            raise ValueError("optimizer must be initialized for training")

        self._require_generation_components()

        # Note: generate_candidates() leaves the policy in eval mode, so the
        # loss forward pass below also runs without dropout (as in the
        # original trainer); the reference model is always in eval mode.
        num_candidates = self.config.num_candidates
        generated_texts, _ = self.generate_candidates(
            prompts, num_candidates=num_candidates
        )

        device = self.accelerator.device
        bos_id = self.tokenizer.token2id.get("<bos>", 0)
        pad_id = self.tokenizer.token2id.get("<pad>", 0)

        rewards_list: List[float] = []
        log_probs_list: List[torch.Tensor] = []
        kl_list: List[torch.Tensor] = []
        flat_generated_ids: List[List[int]] = []
        flat_prompt_lens: List[int] = []

        for prompt_idx, prompt in enumerate(prompts):
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = 1 + len(prompt_tokens)

            for cand_idx in range(num_candidates):
                text = generated_texts[prompt_idx][cand_idx]
                reward = self.verifier.compute_reward(text, ground_truth[prompt_idx])
                rewards_list.append(float(reward))

                generated_ids = self.tokenizer.encode(text, add_special_tokens=True)
                # Truncate to model's max_seq_length to prevent positional embedding overflow
                unwrapped = self.accelerator.unwrap_model(self.policy_model)
                if hasattr(unwrapped, 'max_seq_length') and len(generated_ids) > unwrapped.max_seq_length:
                    generated_ids = generated_ids[:unwrapped.max_seq_length]
                flat_generated_ids.append(generated_ids)
                flat_prompt_lens.append(prompt_len)

        total_candidates = len(flat_generated_ids)
        sub_batch = self.candidate_sub_batch_size or total_candidates

        positions_cache = {}

        # Without sync, DDP skips the all-reduce; it must cover the forward too.
        sync_context = (
            contextlib.nullcontext() if sync_gradients
            else self.accelerator.no_sync(self.policy_model)
        )
        with sync_context:
            for start in range(0, total_candidates, sub_batch):
                end = min(start + sub_batch, total_candidates)
                chunk_ids = flat_generated_ids[start:end]
                chunk_prompt_lens = flat_prompt_lens[start:end]
                max_len = max(len(ids) for ids in chunk_ids)

                padded_input_ids = []
                attention_masks = []
                pad_lens = []
                for ids in chunk_ids:
                    pad_len = max_len - len(ids)
                    pad_lens.append(pad_len)
                    # Right-pad to keep prompt positions consistent with training.
                    padded_input_ids.append(ids + [pad_id] * pad_len)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)

                input_tensor = torch.tensor(
                    padded_input_ids, dtype=torch.long, device=device
                )
                attention_mask = torch.tensor(
                    attention_masks, dtype=torch.float32, device=device
                )

                # Use accelerator.autocast() for automatic mixed precision.
                with self.accelerator.autocast():
                    policy_logits = self._forward_model(
                        self.policy_model, input_tensor, attention_mask=attention_mask
                    )

                with torch.no_grad():
                    with self.accelerator.autocast():
                        reference_logits = self._forward_model(
                            self.reference_model, input_tensor, attention_mask=attention_mask
                        )

                log_probs = torch.log_softmax(policy_logits, dim=-1)
                targets = input_tensor[:, 1:]
                token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

                valid_mask = attention_mask[:, 1:]
                positions = positions_cache.get(token_log_probs.shape[1])
                if positions is None:
                    positions = torch.arange(
                        token_log_probs.shape[1], device=device
                    )
                    positions_cache[token_log_probs.shape[1]] = positions

                prompt_lens_tensor = torch.tensor(chunk_prompt_lens, device=device)
                # With right-padding, prompt tokens start at index 0.
                start_indices = prompt_lens_tensor - 1
                start_indices = torch.clamp(start_indices, min=0)

                start_mask = positions.unsqueeze(0) >= start_indices.unsqueeze(1)
                token_mask = (valid_mask > 0) & start_mask
                log_prob_sum = (token_log_probs * token_mask).sum(dim=1)

                policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
                reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
                policy_probs = torch.softmax(policy_logits, dim=-1)
                kl = policy_probs * (policy_log_probs - reference_log_probs)
                kl_sum = torch.sum(kl, dim=-1)
                kl_mask = attention_mask
                kl_denom = torch.clamp(kl_mask.sum(dim=1), min=1.0)
                kl_value = (kl_sum * kl_mask).sum(dim=1) / kl_denom

                log_probs_list.extend(log_prob_sum)
                kl_list.extend(kl_value)

            batch_size = len(prompts)
            rewards_tensor = torch.tensor(
                rewards_list, dtype=torch.float32, device=device
            ).view(batch_size, num_candidates)

            advantages = self.compute_advantages(rewards_tensor)
            advantages = self.normalize_advantages(advantages)

            log_probs_tensor = torch.stack(log_probs_list).view(
                batch_size, num_candidates
            )
            policy_loss = self.compute_policy_loss(log_probs_tensor, advantages)

            if kl_list:
                kl_divergence = torch.mean(torch.stack(kl_list))
            else:
                kl_divergence = torch.tensor(0.0, device=device)

            total_loss = self.compute_total_loss(policy_loss, kl_divergence)
            scaled_loss = total_loss * loss_scale

            # Use accelerator.backward() for distributed backward pass (and fp16
            # loss scaling). The Accelerator has gradient_accumulation_steps=1, so
            # it does not divide the loss again.
            self.accelerator.backward(scaled_loss)

        if do_step:
            self._optimizer_step()

        avg_reward = rewards_tensor.mean().item()
        reward_rate = (rewards_tensor > 0.5).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_divergence.item(),
            "total_loss": total_loss.item(),
            "avg_reward": avg_reward,
            "reward_rate": reward_rate,
        }

    def _optimizer_step(self) -> bool:
        """Clip, step the optimizer, advance the LR schedule, zero gradients.

        Returns:
            True if the fp16 GradScaler skipped the update (overflow); the
            scheduler is then not advanced.
        """
        # Clip gradients using Accelerator (this also unscales fp16 grads).
        if self.config.gradient_clip > 0:
            self.accelerator.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.gradient_clip
            )
        self.optimizer.step()
        skipped = step_scheduler_if_updated(self.optimizer, self.scheduler)
        self.optimizer.zero_grad(set_to_none=True)
        return skipped

    @staticmethod
    def _unpack_batch(batch) -> Tuple[List[str], List[int]]:
        if isinstance(batch, dict):
            prompts = batch.get("prompts")
            ground_truth = batch.get("ground_truth")
        else:
            prompts, ground_truth = batch
        if prompts is None or ground_truth is None:
            raise ValueError("batch must contain prompts and ground_truth")
        return prompts, ground_truth

    def train(
        self,
        train_dataloader,
        val_dataloader: Optional[Any] = None,
        output_dir: str = ".",
        state: Optional[GRPOTrainingState] = None,
        run_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run (or resume) the GRPO training loop.

        Args:
            train_dataloader: This process's training batches, a sequence of
                (prompts, ground_truth) tuples or dicts. Every process must
                have the same number of batches, or the DDP collectives hang.
            val_dataloader: This process's validation batches (optional).
                Enables validation, best-model saving and early stopping.
            output_dir: Run directory for checkpoints and logs
            state: Restored state when resuming (see load_checkpoint)
            run_info: Schedule, data and SFT identity stored in checkpoints
        """

        start_time = time.time()

        if self.policy_model is None or self.reference_model is None:
            raise ValueError("policy_model and reference_model must be initialized")
        if self.optimizer is None:
            raise ValueError("call setup_optimizer_and_scheduler() before train()")

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        state = state if state is not None else GRPOTrainingState()
        self._run_info = run_info or {}
        accum_steps = max(1, self.config.gradient_accumulation_steps)
        train_batches = list(train_dataloader)
        num_batches = len(train_batches)
        val_batches = list(val_dataloader) if val_dataloader is not None else []

        if (state.early_stopped or state.epoch >= self.config.num_epochs) and self.accelerator.is_main_process:
            print("\nThis checkpoint has already finished training; nothing left to run.")

        while not state.early_stopped and state.epoch < self.config.num_epochs:
            epoch = state.epoch
            if self.accelerator.is_main_process:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"{'='*60}")

            step_start = None
            accum_metrics = None
            accum_batches = 0

            # A resumed epoch continues after the batches already trained
            # (always the start of an accumulation group).
            batch_indices = range(state.batches_done, num_batches)
            if self.accelerator.is_local_main_process:
                progress_bar = tqdm(
                    batch_indices,
                    initial=state.batches_done,
                    total=num_batches,
                    desc=f"Epoch {epoch + 1}",
                )
            else:
                progress_bar = batch_indices

            for batch_idx in progress_bar:
                prompts, ground_truth = self._unpack_batch(train_batches[batch_idx])

                # Accumulation groups are accum_steps consecutive batches; the
                # last group of an epoch may be shorter and is still applied.
                group_start = batch_idx - batch_idx % accum_steps
                group_size = min(accum_steps, num_batches - group_start)
                is_group_end = batch_idx == group_start + group_size - 1

                if batch_idx == group_start:
                    self.optimizer.zero_grad(set_to_none=True)
                    step_start = time.perf_counter()
                    accum_metrics = None
                    accum_batches = 0

                # Each batch contributes 1/group_size of the update, so a
                # group's gradient is the mean over its batches.
                metrics = self.train_step(
                    prompts=prompts,
                    ground_truth=ground_truth,
                    do_step=False,
                    loss_scale=1.0 / group_size,
                    sync_gradients=is_group_end
                )
                if accum_metrics is None:
                    accum_metrics = {key: 0.0 for key in metrics}
                for key, value in metrics.items():
                    accum_metrics[key] += float(value)
                accum_batches += 1
                state.batches_done = batch_idx + 1

                step_time = None
                if is_group_end:
                    if self._optimizer_step():
                        state.skipped_updates += 1
                    state.global_step += 1
                    step_time = time.perf_counter() - step_start

                learning_rate = self.scheduler.get_last_lr()[0]
                avg_metrics = {k: v / accum_batches for k, v in accum_metrics.items()}

                # Update tqdm progress bar with current metrics
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        'loss': f"{avg_metrics['total_loss']:.4f}",
                        'kl': f"{avg_metrics['kl_divergence']:.4f}",
                        'reward': f"{avg_metrics['avg_reward']:.3f}",
                        'rw_rate': f"{avg_metrics['reward_rate']:.3f}",
                        'lr': f"{learning_rate:.2e}",
                        'step': state.global_step,
                    })

                log_metrics = avg_metrics if is_group_end else metrics
                state.training_log.append({
                    "step": state.global_step,
                    "epoch": epoch + 1,
                    "batch_idx": batch_idx,
                    "metrics": {
                        "policy_loss": log_metrics["policy_loss"],
                        "kl_divergence": log_metrics["kl_divergence"],
                        "total_loss": log_metrics["total_loss"],
                        "avg_reward": log_metrics["avg_reward"],
                        "reward_rate": log_metrics["reward_rate"],
                        "val_reward_rate": None,
                    },
                    "learning_rate": learning_rate,
                    "step_time_s": step_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                if is_group_end:
                    # Validate before the periodic save, so the checkpoint
                    # already contains this evaluation's early-stopping state.
                    if val_batches and state.global_step % self.config.eval_every == 0:
                        self._validate(val_batches, state, output_dir)
                    if state.global_step % self.config.save_every == 0:
                        self.save_checkpoint(
                            output_dir=output_dir,
                            filename=f"checkpoint_step_{state.global_step}.pt",
                            metrics=metrics,
                            state=state
                        )

                if state.early_stopped:
                    break

            if state.early_stopped:
                break

            # Also validate at the end of every epoch (unless the last update
            # was just evaluated), so even runs shorter than eval_every get a
            # best model and early-stopping observations.
            if val_batches and state.last_eval_step != state.global_step:
                self._validate(val_batches, state, output_dir)

            state.epoch = epoch + 1
            state.batches_done = 0

        final_metrics = state.training_log[-1]["metrics"] if state.training_log else {}
        final_checkpoint_path = self.save_checkpoint(
            output_dir=output_dir,
            filename="final_model.pt",
            metrics=final_metrics,
            state=state
        )

        # Write training log and summary on main process only.
        log_path = ""
        summary_path = ""
        if self.accelerator.is_main_process:
            log_path = os.path.join(output_dir, "grpo_training_log.json")
            write_json(log_path, state.training_log)
            
            # Compute training time of this invocation.
            end_time = time.time()
            total_seconds = end_time - start_time
            hours, rem = divmod(total_seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            # Create summary of the training run.
            summary = {
                "total_duration_seconds": total_seconds,
                "total_duration_formatted": formatted_time,
                "epochs_completed": state.epoch,
                "batches_done_in_unfinished_epoch": state.batches_done,
                "num_epochs_configured": self.config.num_epochs,
                "early_stopped": state.early_stopped,
                "total_steps": state.global_step,
                "skipped_updates": state.skipped_updates,
                "best_val_reward_rate": state.best_reward_rate,
                "final_checkpoint_path": final_checkpoint_path,
                # Final training metrics.
                "final_metrics": final_metrics or None,
                "config": self.config.to_dict(),
                "run_info": self._run_info,
            }

            # Save training summary.
            summary_path = os.path.join(output_dir, "grpo_training_summary.json")
            write_json(summary_path, summary)
            
            print(f"Training log saved: {log_path}")
            print(f"Training summary saved: {summary_path}")
            print(f"Total training time: {formatted_time}")
        
        self.accelerator.wait_for_everyone()

        return {
            "global_step": state.global_step,
            "log_path": log_path,
            "final_checkpoint_path": final_checkpoint_path,
            "state": state,
        }

    def _validate(
        self,
        val_batches: List[Any],
        state: GRPOTrainingState,
        output_dir: str
    ) -> Dict[str, float]:
        """Evaluate, update early stopping, and save the best model."""
        val_metrics = self.evaluate(val_batches)
        curr_rate = val_metrics["reward_rate"]
        state.last_eval_step = state.global_step
        if state.training_log and state.training_log[-1]["step"] == state.global_step:
            state.training_log[-1]["metrics"]["val_reward_rate"] = curr_rate

        if self.accelerator.is_main_process:
            print(f"\n  Step {state.global_step}: validation reward rate "
                  f"{curr_rate:.4f} ({val_metrics['correct']}/{val_metrics['total']})")

        # Early stopping check (all processes see the same reduced reward_rate)
        if self.config.early_stopping and state.prev_reward_rate is not None:
            prev_rate = state.prev_reward_rate
            if abs(prev_rate) < 1e-8:
                improvement_ratio = float('inf') if curr_rate > prev_rate else 0.0
            else:
                improvement_ratio = (curr_rate - prev_rate) / abs(prev_rate)

            if improvement_ratio < self.config.early_stopping_epsilon:
                state.patience_counter += 1
                if self.accelerator.is_main_process:
                    print(f"  Early stopping: insufficient improvement ({improvement_ratio:.6f} < {self.config.early_stopping_epsilon}), patience {state.patience_counter}/{self.config.early_stopping_patience}")
                if state.patience_counter >= self.config.early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"\nEarly stopping triggered at step {state.global_step}!")
                    state.early_stopped = True
            else:
                state.patience_counter = 0

        state.prev_reward_rate = curr_rate

        # Save the best model after the early-stopping update, under its own
        # file name (never by renaming a periodic checkpoint).
        if state.best_reward_rate is None or curr_rate > state.best_reward_rate:
            state.best_reward_rate = curr_rate
            best_path = self.save_checkpoint(
                output_dir=output_dir,
                filename="best_model.pt",
                metrics=val_metrics,
                state=state
            )
            if self.accelerator.is_main_process:
                print(f"  New best model saved: {best_path}")

        return val_metrics

    def save_checkpoint(
        self,
        output_dir: str,
        filename: str,
        metrics: Dict[str, Any],
        state: GRPOTrainingState
    ) -> str:
        """Save a GRPO checkpoint with full resume state.

        Collective: call on every process (each contributes its RNG state);
        the main process writes the file atomically.
        """
        if self.policy_model is None or self.optimizer is None:
            raise ValueError("policy_model and optimizer must be initialized")

        rng_states = gather_rng_states(self.accelerator)
        checkpoint_path = os.path.join(output_dir, filename)

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

            # Unwrap model to get raw state dict for saving.
            unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
            scaler = self.accelerator.scaler

            checkpoint = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "epoch": state.epoch,
                "batches_done": state.batches_done,
                "step": state.global_step,
                "config": self.config.to_dict(),
                "metrics": metrics,
                "tokenizer_vocab_size": len(self.tokenizer.token2id) if self.tokenizer else None,
                "training_state": state.to_dict(),
                "rng_states": rng_states,
                "run_info": getattr(self, "_run_info", {}),
            }

            if isinstance(unwrapped_model, ArithmeticTransformer):
                checkpoint["model_config"] = describe_model(unwrapped_model)

            atomic_torch_save(checkpoint, checkpoint_path)

        self.accelerator.wait_for_everyone()
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Restore policy weights, optimizer, scheduler and fp16 scaler state.

        The objects created by setup_optimizer_and_scheduler() are reused, so
        the Adam moments and the LR position continue where they stopped.
        The reference model is deliberately left alone: it stays the SFT
        model the run started from, so the KL penalty keeps the same anchor.

        Returns:
            Checkpoint metadata (position, training_state, rng_states, ...).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        if self.optimizer is None:
            raise ValueError("call setup_optimizer_and_scheduler() before load_checkpoint()")

        # Unwrap before loading state dict so keys match regardless of
        # whether the model is wrapped with DDP.
        return load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.accelerator.unwrap_model(self.policy_model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.accelerator.scaler
        )

    def evaluate(self, val_dataloader) -> Dict[str, float]:
        """Run validation evaluation and return reward metrics."""
        if self.policy_model is None:
            raise ValueError("policy_model must be initialized")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be initialized")

        self.policy_model.eval()
        local_total = 0
        local_correct = 0

        # Sampling with a fixed per-process seed makes the metric repeatable
        # across evaluations and runs, and leaves the training RNG untouched.
        with isolated_rng(self.config.seed + self.accelerator.process_index):
            for batch in val_dataloader:
                prompts, ground_truth = self._unpack_batch(batch)

                # generate_candidates() is already Accelerator-aware.
                generated_texts, _ = self.generate_candidates(prompts, num_candidates=1)
                
                for idx, prompt in enumerate(prompts):
                    text = generated_texts[idx][0]
                    reward = self.verifier.compute_reward(text, ground_truth[idx])
                    local_correct += 1 if reward > 0.5 else 0
                    local_total += 1

        # Aggregate evaluation results across all GPUs.
        device = self.accelerator.device
        stats_tensor = torch.tensor([local_correct, local_total], device=device)
        
        # Sum local counts across all processes via reduce.
        gathered_stats = self.accelerator.reduce(stats_tensor, reduction="sum")
        
        global_correct = int(gathered_stats[0].item())
        global_total = int(gathered_stats[1].item())

        reward_rate = global_correct / max(global_total, 1)
        
        return {
            "reward_rate": reward_rate,
            "total": global_total,
            "correct": global_correct,
        }

    def generate_candidates(
        self,
        prompts: List[str],
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_gen_length: Optional[int] = None
    ) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
        """Generate multiple candidate responses per prompt."""
        self._require_generation_components()

        if num_candidates is None:
            num_candidates = self.config.num_candidates
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p
        if max_gen_length is None:
            max_gen_length = self.config.max_gen_length

        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")

        if not prompts:
            return [], []

        # Use accelerator.device for robust device placement.
        device = self.accelerator.device
        
        bos_id = self.tokenizer.token2id.get("<bos>", 0)
        eos_id = self.tokenizer.token2id.get("<eos>", None)
        pad_id = self.tokenizer.token2id.get("<pad>", 0)

        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_candidates)

        prompt_tokens_list = []
        for prompt in expanded_prompts:
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tokens_list.append([bos_id] + prompt_tokens)

        max_prompt_len = max(len(ids) for ids in prompt_tokens_list)
        padded_ids = []
        attention_masks = []
        for ids in prompt_tokens_list:
            pad_len = max_prompt_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.float32, device=device)

        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        log_probs_lists: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        # Clamp generation length to model's max_seq_length to avoid
        # positional embedding overflow when sequences are re-encoded
        # with special tokens in train_step.
        unwrapped = self.accelerator.unwrap_model(self.policy_model)
        if hasattr(unwrapped, 'max_seq_length'):
            effective_max_len = min(max_gen_length, unwrapped.max_seq_length - 2)
        else:
            effective_max_len = max_gen_length

        self.policy_model.eval()
        with torch.no_grad():
            while input_ids.shape[1] < effective_max_len:
                # Use accelerator.autocast() for automatic mixed precision.
                with self.accelerator.autocast():
                    logits = self._forward_model(
                        self.policy_model, input_ids, attention_mask=attention_mask
                    )
                
                next_token_logits = logits[:, -1, :]

                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    kth = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    indices_to_remove = next_token_logits < kth
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                token_log_prob = torch.log(
                    probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
                )

                active = ~finished
                if active.any():
                    for idx in torch.nonzero(active, as_tuple=False).squeeze(1).tolist():
                        log_probs_lists[idx].append(token_log_prob[idx])

                if eos_id is not None:
                    just_finished = active & (next_token == eos_id)
                else:
                    just_finished = torch.zeros_like(finished)

                next_token = torch.where(active, next_token, torch.tensor(pad_id, device=device))
                next_attention = torch.where(active, torch.ones_like(next_token), torch.zeros_like(next_token))

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                attention_mask = torch.cat([attention_mask, next_attention.unsqueeze(1).float()], dim=1)

                finished = finished | just_finished
                if finished.all():
                    break

        decoded_texts = [
            self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            for ids in input_ids
        ]

        all_texts: List[List[str]] = []
        all_log_probs: List[List[torch.Tensor]] = []
        idx = 0
        for _ in prompts:
            prompt_texts = []
            prompt_log_probs = []
            for _ in range(num_candidates):
                prompt_texts.append(decoded_texts[idx])
                lp = log_probs_lists[idx]
                prompt_log_probs.append(torch.stack(lp) if lp else torch.tensor([]))
                idx += 1
            all_texts.append(prompt_texts)
            all_log_probs.append(prompt_log_probs)

        return all_texts, all_log_probs

    def compute_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of generated sequence."""
        if self.policy_model is None:
            raise ValueError("policy_model must be provided to compute log probabilities")

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if generated_ids.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)

        prompt_len = input_ids.shape[1]
        
        # Return zero for degenerate sequences.
        if generated_ids.shape[1] <= 1:
            return torch.tensor(0.0, device=generated_ids.device)

        # Use accelerator.autocast() to match the mixed-precision dtype.
        with self.accelerator.autocast():
            logits = self.policy_model(generated_ids[:, :-1])
            
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = generated_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        start_index = max(prompt_len - 1, 0)
        token_log_probs = token_log_probs[:, start_index:]
        return torch.sum(token_log_probs)


