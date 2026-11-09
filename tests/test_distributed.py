"""Two-process tests: real worker processes (torchrun, gloo backend, CPU).

These check what a single process cannot: that the learning-rate schedule
is sized for the per-process number of updates, that every process gets the
same data split and number of batches (otherwise DDP hangs), and that a
multi-process run resumes exactly.
"""

import os
import re
import unittest

import torch.distributed as dist

from tests.helpers import (
    assert_cli_succeeded,
    build_workspace,
    load,
    max_weight_difference,
    only_run_dir,
    remove_workspace,
    run_cli,
)
from tests.test_training_runs import foundational_args

# 27 global batches of 8 sharded over 2 processes -> 14 per process
# (Accelerate pads the last one so both processes run the same number).
UPDATES_PER_EPOCH_2P = 14


@unittest.skipUnless(dist.is_available(), "torch.distributed not available")
class TwoProcessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ws = build_workspace()

    @classmethod
    def tearDownClass(cls):
        remove_workspace(cls.ws)

    def test_foundational_schedule_and_resume(self):
        out = os.path.join(self.ws["root"], "found2")
        result = run_cli(foundational_args(self.ws, out, 2, "--save-every", "5"), nproc=2)
        assert_cli_succeeded(self, result, "TRAINING COMPLETED SUCCESSFULLY")
        output = result.stdout + result.stderr
        self.assertIn("Processes: 2", output)
        self.assertRegex(output, rf"{UPDATES_PER_EPOCH_2P} updates/epoch, "
                                 rf"{2 * UPDATES_PER_EPOCH_2P} updates in total")
        run_dir = only_run_dir(out, "foundational_")
        final = load(os.path.join(run_dir, "final_model.pt"))
        self.assertEqual(final["step"], 2 * UPDATES_PER_EPOCH_2P)
        # Before the fix the LR reached 0 after half of the updates on 2 GPUs.
        self.assertEqual(final["scheduler_state_dict"]["last_epoch"], 2 * UPDATES_PER_EPOCH_2P)
        self.assertEqual(len(final["rng_states"]), 2)

        result = run_cli(foundational_args(self.ws, out, 2, "--save-every", "5",
                                           "--resume-checkpoint",
                                           os.path.join(run_dir, "checkpoint_step_20.pt")), nproc=2)
        assert_cli_succeeded(self, result, "TRAINING COMPLETED SUCCESSFULLY")
        resumed = load(os.path.join(only_run_dir(out, os.path.basename(run_dir) + "_resumed"),
                                    "final_model.pt"))
        self.assertEqual(max_weight_difference(final, resumed), 0.0)

    def test_grpo_with_uneven_prompt_count(self):
        sft_out = os.path.join(self.ws["root"], "sft")
        result = run_cli(foundational_args(self.ws, sft_out, 1, "--save-every", "1000"))
        assert_cli_succeeded(self, result, "TRAINING COMPLETED SUCCESSFULLY")
        sft = os.path.join(only_run_dir(sft_out, "foundational_"), "final_model.pt")

        out = os.path.join(self.ws["root"], "grpo2")
        # 60 prompts, 5 held out: neither part divides evenly by 2 processes.
        result = run_cli(["run_grpo_training.py", "--tokenizer", self.ws["tokenizer"],
                          "--instruction-corpus", self.ws["instruction"], "--sft-checkpoint", sft,
                          "--output-dir", out, "--num-epochs", "1", "--batch-size", "2",
                          "--num-candidates", "2", "--max-gen-length", "32", "--val-size", "5",
                          "--eval-every", "2", "--save-every", "100",
                          "--gradient-accumulation-steps", "2", "--mixed-precision", "no"],
                         nproc=2)
        assert_cli_succeeded(self, result, "GRPO TRAINING COMPLETED SUCCESSFULLY")
        output = result.stdout + result.stderr
        self.assertIn("Training prompts: 54 (27 per process)", output)
        self.assertIn("Validation prompts: 4 (2 per process)", output)
        # 27 prompts -> 14 batches of 2 -> 7 updates with accumulation 2.
        self.assertTrue(re.search(r"7 updates in total", output), output[-3000:])
        final = load(os.path.join(only_run_dir(out, "grpo_"), "final_model.pt"))
        self.assertEqual(final["step"], 7)


if __name__ == "__main__":
    unittest.main()
