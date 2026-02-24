"""GRPO trainer module."""

from typing import Any, Dict, List, Optional, Tuple

import math
import json
import os
from datetime import datetime, timezone
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator

from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from evaluation.arithmetic_verifier import ArithmeticVerifier
from training.train_foundational_d import (
    get_linear_schedule_with_warmup,
    load_checkpoint,
)
from model.transformer_model import ArithmeticTransformer

from configs.grpo_config import GRPOConfig


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
        total_steps: Optional[int] = None,
        use_mixed_precision: bool = False,
        candidate_sub_batch_size: Optional[int] = None,
        accelerator: Accelerator = None
    ):
        self.config = config
        self.tokenizer = tokenizer

        warmup_ratio = 0.05

        # Require an Accelerator instance for distributed training.
        if accelerator is None:
            raise ValueError("Accelerator must be provided to GRPOTrainer")
        self.accelerator = accelerator

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = None
        self.scheduler = None
        self.verifier = ArithmeticVerifier()
        self.use_mixed_precision = use_mixed_precision
        self.candidate_sub_batch_size = candidate_sub_batch_size
        
        # GradScaler is handled automatically by Accelerate.
        self._scaler = None

        if self.tokenizer is None and tokenizer_path is not None:
            self.tokenizer = ArithmeticBPETokenizer()
            self.tokenizer.load(tokenizer_path)

        if self.policy_model is None or self.reference_model is None:
            if sft_checkpoint_path is not None:
                self._load_models_from_checkpoint(sft_checkpoint_path)

        if self.reference_model is not None:
            self._freeze_reference_model()

        # Initialize optimizer and scheduler (device placement is done by prepare()).
        if self.policy_model is not None:
            params = list(self.policy_model.parameters())
            if params:
                self.optimizer = torch.optim.AdamW(
                    params,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.01
                )
                if total_steps is not None:
                    real_warmup_steps = int(total_steps * warmup_ratio)
                    self.scheduler = get_linear_schedule_with_warmup(
                        optimizer=self.optimizer,
                        num_warmup_steps=real_warmup_steps,
                        num_training_steps=max(1, total_steps)
                    )

        # Prepare all models, optimizer, and scheduler for distributed training.
        # The reference model is also prepared to support FSDP/DeepSpeed sharding.
        self.policy_model, self.optimizer, self.scheduler, self.reference_model = self.accelerator.prepare(
            self.policy_model, self.optimizer, self.scheduler, self.reference_model
        )

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

    def reset_optimizer_and_scheduler(self, total_steps: Optional[int] = None) -> None:
        """Re-initialize optimizer/scheduler and prepare them."""
        if self.policy_model is None:
            raise ValueError("policy_model must be initialized")
        
        # Get parameters from the unwrapped model.
        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
        
        self.optimizer = torch.optim.AdamW(
            unwrapped_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        if total_steps is not None:
            # Use 5% of total steps for learning rate warmup.
            warmup_ratio = 0.05
            real_warmup_steps = int(total_steps * warmup_ratio)
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=real_warmup_steps,
                num_training_steps=max(1, total_steps)
            )
        
        # Re-prepare optimizer and scheduler after re-initialization.
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

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
        loss_scale: float = 1.0
    ) -> dict:
        """Execute single GRPO training step."""
        if len(prompts) != len(ground_truth):
            raise ValueError("prompts and ground_truth must have the same length")

        if self.policy_model is None or self.reference_model is None:
            raise ValueError("policy_model and reference_model must be initialized")
        if self.optimizer is None:
            raise ValueError("optimizer must be initialized for training")

        self._require_generation_components()
        self.policy_model.train()

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

        device_type = "cuda" if self.config.device == "cuda" else "cpu"
        positions_cache = {}

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
                with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self.use_mixed_precision
                ):
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

        # Use accelerator.backward() for distributed backward pass.
        self.accelerator.backward(scaled_loss)

        if do_step:
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            
            # Clip gradients using Accelerator.
            if self.config.gradient_clip > 0:
                self.accelerator.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.gradient_clip
                )
            
            # Accelerator handles loss scaling/unscaling internally.
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scheduler is not None:
                self.scheduler.step()

        avg_reward = rewards_tensor.mean().item()
        reward_rate = (rewards_tensor > 0.5).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_divergence.item(),
            "total_loss": total_loss.item(),
            "avg_reward": avg_reward,
            "reward_rate": reward_rate,
        }

    def train(
        self,
        train_dataloader,
        val_dataloader: Optional[Any] = None,
        output_dir: str = ".",
        start_epoch: int = 0,
        start_step: int = 0
    ) -> Dict[str, Any]:
        """Run GRPO training loop."""

        start_time = time.time()

        if self.policy_model is None or self.reference_model is None:
            raise ValueError("policy_model and reference_model must be initialized")
        if self.optimizer is None:
            raise ValueError("optimizer must be initialized for training")

        # [modified] only create output directory in the main process
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)

        training_log: List[Dict[str, Any]] = []
        global_step = start_step
        best_reward_rate = -1.0
        accum_steps = max(1, self.accelerator.gradient_accumulation_steps)
        total_steps = None
        try:
            total_steps = math.ceil(len(train_dataloader) / accum_steps) * self.config.num_epochs
        except TypeError:
            total_steps = None

        # Early stopping state
        prev_reward_rate = None
        es_patience_counter = 0
        should_stop_early = False

        for epoch in range(start_epoch, self.config.num_epochs):
            if self.accelerator.is_local_main_process:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"{'='*60}")

            step_start = None
            step_time = None
            accum_metrics = None
            accum_batches = 0

            if self.accelerator.is_local_main_process:
                progress_bar = tqdm(
                    enumerate(train_dataloader),
                    total=len(train_dataloader) if hasattr(train_dataloader, '__len__') else None,
                    desc=f"Epoch {epoch + 1}",
                )
            else:
                progress_bar = enumerate(train_dataloader)

            for batch_idx, batch in progress_bar:
                did_step = False
                if isinstance(batch, dict):
                    prompts = batch.get("prompts")
                    ground_truth = batch.get("ground_truth")
                else:
                    prompts, ground_truth = batch

                if prompts is None or ground_truth is None:
                    raise ValueError("batch must contain prompts and ground_truth")

                if batch_idx % accum_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    step_start = time.perf_counter()
                    step_time = None
                    accum_metrics = None
                    accum_batches = 0

                metrics = self.train_step(
                    prompts=prompts,
                    ground_truth=ground_truth,
                    do_step=False,
                    loss_scale=1.0 / accum_steps
                )
                if accum_metrics is None:
                    accum_metrics = {key: 0.0 for key in metrics}
                for key, value in metrics.items():
                    accum_metrics[key] += float(value)
                accum_batches += 1

                val_metrics = None
                if (batch_idx + 1) % accum_steps == 0:
                    # Clip gradients and perform optimizer step.
                    if self.config.gradient_clip > 0:
                        self.accelerator.clip_grad_norm_(
                            self.policy_model.parameters(),
                            self.config.gradient_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    global_step += 1
                    did_step = True
                    if step_start is not None:
                        step_time = time.perf_counter() - step_start
                    step_start = None

                    if global_step % self.config.save_every == 0:
                        self.save_checkpoint(
                            output_dir=output_dir,
                            step=global_step,
                            epoch=epoch + 1,
                            metrics=metrics,
                            is_final=False
                        )

                    if val_dataloader is not None and global_step % self.config.eval_every == 0:
                        val_metrics = self.evaluate(val_dataloader)


                        # Save best model on main process only.
                        if self.accelerator.is_local_main_process:
                             if val_metrics["reward_rate"] > best_reward_rate:
                                best_reward_rate = val_metrics["reward_rate"]
                                best_path = self.save_checkpoint(
                                    output_dir=output_dir,
                                    step=global_step,
                                    epoch=epoch + 1,
                                    metrics=val_metrics,
                                    is_final=False
                                )
                                best_model_path = os.path.join(output_dir, "best_model.pt")
                                if os.path.exists(best_model_path):
                                     os.remove(best_model_path)
                                os.rename(best_path, best_model_path)

                        # Early stopping check (all processes see the same reduced reward_rate)
                        if self.config.early_stopping and prev_reward_rate is not None:
                            curr_rate = val_metrics["reward_rate"]
                            if abs(prev_reward_rate) < 1e-8:
                                improvement_ratio = float('inf') if curr_rate > prev_reward_rate else 0.0
                            else:
                                improvement_ratio = (curr_rate - prev_reward_rate) / abs(prev_reward_rate)

                            if improvement_ratio < self.config.early_stopping_epsilon:
                                es_patience_counter += 1
                                if self.accelerator.is_local_main_process:
                                    print(f"  Early stopping: insufficient improvement ({improvement_ratio:.6f} < {self.config.early_stopping_epsilon}), patience {es_patience_counter}/{self.config.early_stopping_patience}")
                                if es_patience_counter >= self.config.early_stopping_patience:
                                    if self.accelerator.is_local_main_process:
                                        print(f"\nEarly stopping triggered at step {global_step}!")
                                    should_stop_early = True
                            else:
                                es_patience_counter = 0

                        prev_reward_rate = val_metrics["reward_rate"]

                    if should_stop_early:
                        break

                if self.scheduler is not None:
                    learning_rate = self.scheduler.get_last_lr()[0]
                else:
                    learning_rate = self.optimizer.param_groups[0]["lr"]

                # Update tqdm progress bar with current metrics
                if self.accelerator.is_local_main_process:
                    avg_metrics = metrics
                    if accum_metrics is not None and accum_batches:
                        avg_metrics = {k: v / accum_batches for k, v in accum_metrics.items()}

                    postfix = {
                        'loss': f"{avg_metrics['total_loss']:.4f}",
                        'kl': f"{avg_metrics['kl_divergence']:.4f}",
                        'reward': f"{avg_metrics['avg_reward']:.3f}",
                        'rw_rate': f"{avg_metrics['reward_rate']:.3f}",
                        'lr': f"{learning_rate:.2e}",
                        'step': global_step,
                    }
                    progress_bar.set_postfix(postfix)

                log_metrics = metrics
                if did_step and accum_metrics is not None and accum_batches:
                    log_metrics = {
                        key: value / accum_batches
                        for key, value in accum_metrics.items()
                    }

                training_log.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch_idx": batch_idx,
                    "metrics": {
                        "policy_loss": log_metrics["policy_loss"],
                        "kl_divergence": log_metrics["kl_divergence"],
                        "total_loss": log_metrics["total_loss"],
                        "avg_reward": log_metrics["avg_reward"],
                        "reward_rate": log_metrics["reward_rate"],
                        "val_reward_rate": val_metrics["reward_rate"] if val_metrics else None,
                    },
                    "learning_rate": learning_rate,
                    "step_time_s": step_time if did_step else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            if should_stop_early:
                break

        final_checkpoint_path = self.save_checkpoint(
            output_dir=output_dir,
            step=global_step,
            epoch=self.config.num_epochs,
            metrics=training_log[-1]["metrics"] if training_log else {},
            is_final=True
        )

        # Write training log and summary on main process only.
        log_path = ""
        summary_path = ""
        if self.accelerator.is_local_main_process:
            log_path = os.path.join(output_dir, "grpo_training_log.json")
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)
            
            # Compute total training time.
            end_time = time.time()
            total_seconds = end_time - start_time
            hours, rem = divmod(total_seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            # Create summary of the training run.
            summary = {
                "total_duration_seconds": total_seconds,
                "total_duration_formatted": formatted_time,
                "total_epochs": self.config.num_epochs,
                "total_steps": global_step,
                "final_checkpoint_path": final_checkpoint_path,
                # Final training metrics.
                "final_metrics": training_log[-1]["metrics"] if training_log else None
            }

            # Save training summary.
            summary_path = os.path.join(output_dir, "grpo_training_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"Training log saved: {log_path}")
            print(f"Training summary saved: {summary_path}")
            print(f"Total training time: {formatted_time}")
        
        self.accelerator.wait_for_everyone()

        return {
            "global_step": global_step,
            "log_path": log_path,
            "final_checkpoint_path": final_checkpoint_path,
        }

    def save_checkpoint(
        self,
        output_dir: str,
        step: int,
        epoch: int,
        metrics: Dict[str, Any],
        is_final: bool = False
    ) -> str:
        """Save GRPO checkpoint with metadata."""
        
        # Only save checkpoints on the main process.
        if not self.accelerator.is_local_main_process:
            return ""
        
        if self.policy_model is None or self.optimizer is None:
            raise ValueError("policy_model and optimizer must be initialized")

        os.makedirs(output_dir, exist_ok=True)

        # Unwrap model to get raw state dict for saving.
        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)

        checkpoint = {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "step": step,
            "config": self.config.to_dict(),
            "metrics": metrics,
        }

        if isinstance(unwrapped_model, ArithmeticTransformer):
            checkpoint["model_config"] = {
                "vocab_size": unwrapped_model.vocab_size,
                "d_model": unwrapped_model.d_model,
                "nhead": unwrapped_model.nhead,
                "num_layers": unwrapped_model.num_layers,
                "dim_feedforward": unwrapped_model.dim_feedforward,
                "dropout": unwrapped_model.dropout,
                "max_seq_length": unwrapped_model.max_seq_length,
            }

        if is_final:
            checkpoint_path = os.path.join(output_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load GRPO checkpoint and restore trainer state."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        # Load to CPU to avoid GPU memory contention across processes.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 1. Restore policy model.
        if self.policy_model is None:
            model_config = checkpoint.get("model_config")
            if model_config is None:
                raise ValueError("Checkpoint missing model_config")
            self.policy_model = ArithmeticTransformer(**model_config)
            # Newly created models must be prepared immediately.
            self.policy_model = self.accelerator.prepare(self.policy_model)

        # Unwrap before loading state dict so keys match regardless of
        # whether the model is wrapped with DDP.
        unwrapped_policy = self.accelerator.unwrap_model(self.policy_model)
        unwrapped_policy.load_state_dict(checkpoint["model_state_dict"])
        # 2. Restore reference model.
        if self.reference_model is None:
            self.reference_model = ArithmeticTransformer(**checkpoint["model_config"])
            # Newly created reference model must be prepared too.
            self.reference_model = self.accelerator.prepare(self.reference_model)
        
        # Unwrap reference model to load weights.
        unwrapped_ref = self.accelerator.unwrap_model(self.reference_model)
        unwrapped_ref.load_state_dict(checkpoint["model_state_dict"])
        self._freeze_reference_model()

        # 3. Restore optimizer.
        if self.optimizer is None:
            # Use unwrapped model parameters for the new optimizer.
            self.optimizer = torch.optim.AdamW(
                unwrapped_policy.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
            # Newly initialized optimizer must be prepared.
            self.optimizer = self.accelerator.prepare(self.optimizer)
            
        if checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 4. Restore scheduler.
        if checkpoint.get("scheduler_state_dict") and self.optimizer is not None:
            if self.scheduler is None:
                self.scheduler = get_linear_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=self.config.warmup_steps,
                    num_training_steps=1  # Placeholder; reset_optimizer_and_scheduler() sets the real value.
                )
                # Newly initialized scheduler must be prepared.
                self.scheduler = self.accelerator.prepare(self.scheduler)
                
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def evaluate(self, val_dataloader) -> Dict[str, float]:
        """Run validation evaluation and return reward metrics."""
        if self.policy_model is None:
            raise ValueError("policy_model must be initialized")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be initialized")

        self.policy_model.eval()
        local_total = 0
        local_correct = 0

        for batch in val_dataloader:
            if isinstance(batch, dict):
                prompts = batch.get("prompts")
                ground_truth = batch.get("ground_truth")
            else:
                prompts, ground_truth = batch

            if prompts is None or ground_truth is None:
                raise ValueError("batch must contain prompts and ground_truth")

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
        
        global_correct = gathered_stats[0].item()
        global_total = gathered_stats[1].item()

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


