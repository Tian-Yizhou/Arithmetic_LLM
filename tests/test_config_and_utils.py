"""Fast tests for configuration resolution and the shared training utilities."""

import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from configs.grpo_config import GRPOConfig
from configs.lora_config import LoRAConfig
from configs.training_config import TrainingConfig
from training.common import (
    atomic_torch_save,
    capture_rng_state,
    check_resume_compatibility,
    get_linear_schedule_with_warmup,
    isolated_rng,
    legacy_resume_position,
    resolve_warmup_steps,
    restore_rng_state,
    step_scheduler_if_updated,
)
from training.train_grpo import split_and_shard_pairs


class TrainingConfigTest(unittest.TestCase):

    def test_precedence_cli_over_json_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w") as f:
                json.dump({"learning_rate": 3e-4, "gradient_accumulation_steps": 4,
                           "device": "cuda"}, f)  # 'device' is a removed legacy field
            config = TrainingConfig.resolve(
                defaults={"learning_rate": 1e-4, "num_epochs": 7},
                json_path=path,
                overrides={"learning_rate": None, "num_epochs": 2, "seed": None},
            )
        self.assertEqual(config.learning_rate, 3e-4)       # JSON beats default
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.num_epochs, 2)             # explicit flag beats default
        self.assertEqual(config.seed, 42)                  # untouched default

    def test_from_json_rebuilds_lora_config(self):
        config = TrainingConfig(lora_config=LoRAConfig(rank=4, alpha=8.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            config.to_json(path)
            loaded = TrainingConfig.from_json(path)
        self.assertIsInstance(loaded.lora_config, LoRAConfig)
        self.assertEqual(loaded.lora_config.rank, 4)
        self.assertEqual(loaded.to_dict()["lora_config"]["alpha"], 8.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            TrainingConfig(mixed_precision="fp8").validate()
        with self.assertRaises(ValueError):
            TrainingConfig(warmup_ratio=1.0).validate()
        TrainingConfig(warmup_steps=0).validate()  # explicit zero is allowed
        with self.assertRaises(ValueError):
            GRPOConfig(early_stopping=True, val_size=0).validate()


class ScheduleTest(unittest.TestCase):

    def test_resolve_warmup_steps(self):
        self.assertEqual(resolve_warmup_steps(200, None, 0.05), 10)
        self.assertEqual(resolve_warmup_steps(200, 0, 0.05), 0)     # explicit zero honoured
        self.assertEqual(resolve_warmup_steps(200, 30, 0.05), 30)   # explicit value wins
        with self.assertRaises(ValueError):
            resolve_warmup_steps(20, 1000, 0.05)

    def test_scheduler_not_advanced_on_skipped_update(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1.0)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10)

        optimizer.step_was_skipped = True   # what AcceleratedOptimizer reports on fp16 overflow
        self.assertTrue(step_scheduler_if_updated(optimizer, scheduler))
        self.assertEqual(scheduler.last_epoch, 0)

        optimizer.step_was_skipped = False
        optimizer.step()
        self.assertFalse(step_scheduler_if_updated(optimizer, scheduler))
        self.assertEqual(scheduler.last_epoch, 1)


class CheckpointUtilsTest(unittest.TestCase):

    def test_rng_state_round_trip_through_weights_only_checkpoint(self):
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rng.pt")
            atomic_torch_save({"rng_states": [state]}, path)
            self.assertFalse(os.path.exists(path + ".tmp"))
            loaded = torch.load(path, map_location="cpu")  # weights_only=True by default
        restore_rng_state(loaded["rng_states"][0])
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expected)

    def test_isolated_rng_leaves_outer_stream_untouched(self):
        torch.manual_seed(1)
        reference = torch.rand(3)
        torch.manual_seed(1)
        with isolated_rng(123):
            first = torch.rand(2)
        self.assertTrue(torch.equal(torch.rand(3), reference))
        with isolated_rng(123):
            self.assertTrue(torch.equal(torch.rand(2), first))

    def test_check_resume_compatibility(self):
        check_resume_compatibility({"a": 1}, {"a": 1}, "schedule")
        with self.assertRaisesRegex(ValueError, "total_steps"):
            check_resume_compatibility({"total_steps": 10}, {"total_steps": 20}, "schedule")

    def test_legacy_resume_position(self):
        # 27 updates per epoch: update 30 is 3 updates into the second epoch.
        self.assertEqual(legacy_resume_position(30, 27, 1), (1, 3))
        self.assertEqual(legacy_resume_position(30, 14, 2), (2, 4))


class GRPODataSplitTest(unittest.TestCase):

    def test_equal_shares_on_every_process(self):
        pairs = [{"prompt": str(i), "ground_truth": i} for i in range(41)]
        shares = [split_and_shard_pairs(pairs, 5, 0, 2, rank) for rank in range(2)]
        (train0, val0, sizes), (train1, val1, _) = shares
        self.assertEqual(len(train0), len(train1))
        self.assertEqual(len(val0), len(val1))
        self.assertEqual(sizes["train_pairs"] % 2, 0)
        self.assertEqual(sizes["val_pairs"] % 2, 0)
        seen = [p["ground_truth"] for p in train0 + train1 + val0 + val1]
        self.assertEqual(len(seen), len(set(seen)))      # no prompt in two places

    def test_split_is_deterministic(self):
        pairs = [{"prompt": str(i), "ground_truth": i} for i in range(20)]
        self.assertEqual(split_and_shard_pairs(pairs, 4, 7, 1, 0),
                         split_and_shard_pairs(pairs, 4, 7, 1, 0))

    def test_rejects_too_little_data(self):
        pairs = [{"prompt": str(i), "ground_truth": i} for i in range(4)]
        with self.assertRaises(ValueError):
            split_and_shard_pairs(pairs, 3, 0, 2, 0)    # one training prompt for two processes
        with self.assertRaises(ValueError):
            split_and_shard_pairs(pairs, 1, 0, 2, 0)    # one validation prompt for two processes


if __name__ == "__main__":
    unittest.main()
