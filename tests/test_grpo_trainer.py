"""Tests for the GRPO trainer's optimizer lifecycle, resume and accumulation."""

import copy
import tempfile
import unittest

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from configs.grpo_config import GRPOConfig
from data.arithmetic_tokenizer import ArithmeticBPETokenizer
from model.transformer_model import ArithmeticTransformer
from tests.helpers import build_workspace, remove_workspace
from training.grpo_trainer import GRPOTrainer, GRPOTrainingState


class GRPOTrainerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = build_workspace(num_samples=120)
        cls.tokenizer = ArithmeticBPETokenizer()
        cls.tokenizer.load(cls.workspace["tokenizer"])

    @classmethod
    def tearDownClass(cls):
        remove_workspace(cls.workspace)

    def setUp(self):
        AcceleratorState._reset_state(True)
        self.accelerator = Accelerator(cpu=True)
        torch.manual_seed(0)
        self.sft_model = ArithmeticTransformer(
            vocab_size=len(self.tokenizer.token2id), d_model=16, nhead=2,
            num_layers=1, dim_feedforward=32, max_seq_length=64,
        )
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        AcceleratorState._reset_state(True)

    def make_trainer(self, **config_overrides) -> GRPOTrainer:
        config = GRPOConfig(batch_size=2, num_candidates=2, max_gen_length=24,
                            save_every=1000, eval_every=1000, **config_overrides)
        return GRPOTrainer(
            config=config,
            tokenizer=self.tokenizer,
            policy_model=copy.deepcopy(self.sft_model),
            reference_model=copy.deepcopy(self.sft_model),
            accelerator=self.accelerator,
        )

    def test_one_optimizer_and_fixed_reference_across_resume(self):
        trainer = self.make_trainer()
        trainer.setup_optimizer_and_scheduler(total_steps=10, warmup_steps=1)
        # More than one prepared optimizer makes clip_grad_norm_() unscale
        # fp16 gradients repeatedly (the bug this guards against).
        self.assertEqual(len(self.accelerator._optimizers), 1)
        with self.assertRaises(RuntimeError):
            trainer.setup_optimizer_and_scheduler(total_steps=10, warmup_steps=1)

        prompts = ["Evaluate: 1 + 2 <think>", "Evaluate: 3 - 1 <think>"]
        trainer.train_step(prompts, [3, 2], do_step=True)
        state = GRPOTrainingState(epoch=0, batches_done=1, global_step=1)
        path = trainer.save_checkpoint(self.tmp.name, "checkpoint_step_1.pt", {}, state)

        # A fresh trainer resumed from the checkpoint.
        AcceleratorState._reset_state(True)
        self.accelerator = Accelerator(cpu=True)
        resumed = self.make_trainer()
        resumed.setup_optimizer_and_scheduler(total_steps=10, warmup_steps=1)
        meta = resumed.load_checkpoint(path)

        self.assertEqual(len(self.accelerator._optimizers), 1)
        self.assertGreater(len(resumed.optimizer.state), 0)          # Adam moments restored
        self.assertEqual(resumed.scheduler.last_epoch, 1)            # schedule position restored
        self.assertEqual(meta["batches_done"], 1)
        policy = self.accelerator.unwrap_model(resumed.policy_model).state_dict()
        saved = torch.load(path, map_location="cpu")["model_state_dict"]
        for key, value in saved.items():
            self.assertTrue(torch.equal(policy[key].cpu(), value))
        # The reference model is still the SFT model, in eval mode.
        reference = resumed.reference_model.state_dict()
        for key, value in self.sft_model.state_dict().items():
            self.assertTrue(torch.equal(reference[key].cpu(), value))
        self.assertFalse(resumed.reference_model.training)

    def test_last_short_accumulation_group_is_applied(self):
        trainer = self.make_trainer(num_epochs=2, gradient_accumulation_steps=2)
        trainer.setup_optimizer_and_scheduler(total_steps=4, warmup_steps=0)
        calls = []

        def fake_train_step(prompts, ground_truth, do_step, loss_scale, sync_gradients):
            calls.append((loss_scale, sync_gradients))
            params = trainer.accelerator.unwrap_model(trainer.policy_model).parameters()
            loss = sum(p.float().pow(2).sum() for p in params) * loss_scale
            trainer.accelerator.backward(loss)
            return {"policy_loss": 0.0, "kl_divergence": 0.0, "total_loss": 0.0,
                    "avg_reward": 0.0, "reward_rate": 0.0}

        trainer.train_step = fake_train_step
        batches = [(["p"], [1])] * 3
        result = trainer.train(batches, output_dir=self.tmp.name)

        # 3 batches per epoch with accumulation 2: groups of 2 and 1 -> 2 updates/epoch.
        self.assertEqual(result["global_step"], 4)
        self.assertEqual(trainer.scheduler.last_epoch, 4)
        self.assertEqual([c[0] for c in calls], [0.5, 0.5, 1.0] * 2)
        self.assertEqual([c[1] for c in calls], [False, True, True] * 2)

    def test_accelerator_accumulation_is_rejected(self):
        AcceleratorState._reset_state(True)
        self.accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        with self.assertRaises(ValueError):
            self.make_trainer()

    def test_validation_is_reproducible(self):
        trainer = self.make_trainer()
        trainer.setup_optimizer_and_scheduler(total_steps=10, warmup_steps=1)
        batches = [(["Evaluate: 1 + 2 <think>", "Evaluate: 4 - 1 <think>"], [3, 3])]
        torch.manual_seed(5)
        before = torch.rand(1)
        torch.manual_seed(5)
        first = trainer.evaluate(batches)
        self.assertTrue(torch.equal(torch.rand(1), before))   # training RNG untouched
        self.assertEqual(trainer.evaluate(batches), first)


if __name__ == "__main__":
    unittest.main()
