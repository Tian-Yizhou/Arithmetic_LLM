"""End-to-end tests of the training entry points (single process, CPU, fp32).

Each test runs the real command-line scripts on a tiny model, so they check
behaviour that would otherwise fail silently: schedule length, checkpoint survival,
exact mid-epoch resume, early-stopping metadata and GRPO resume.
"""

import json
import os
import unittest

import torch

from tests.helpers import (
    TINY_MODEL_ARGS,
    assert_cli_succeeded,
    build_workspace,
    load,
    max_weight_difference,
    only_run_dir,
    remove_workspace,
    run_cli,
)

# 240 samples -> 216 training lines -> 27 batches of 8 -> 27 updates per epoch.
UPDATES_PER_EPOCH = 27


def foundational_args(ws, out, epochs, *extra):
    return ["run_foundational_training.py",
            "--corpus-path", ws["foundational"], "--tokenizer-path", ws["tokenizer"],
            "--output-dir", out, "--batch-size", "8", "--learning-rate", "1e-3",
            "--num-epochs", str(epochs), "--mixed-precision", "no",
            *TINY_MODEL_ARGS, *extra]


class SupervisedRunTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ws = build_workspace()
        cls.out = os.path.join(cls.ws["root"], "runs")
        # save_every 9 puts every third periodic save on an epoch boundary,
        # where the old best-model rename destroyed it.
        result = run_cli(foundational_args(cls.ws, cls.out, 3, "--save-every", "9"))
        output = result.stdout + result.stderr
        if "TRAINING COMPLETED SUCCESSFULLY" not in output:
            raise RuntimeError(output[-4000:])
        cls.run_dir = only_run_dir(cls.out, "foundational_")

    @classmethod
    def tearDownClass(cls):
        remove_workspace(cls.ws)

    def test_schedule_ends_at_zero_after_all_updates(self):
        final = load(os.path.join(self.run_dir, "final_model.pt"))
        self.assertEqual(final["step"], 3 * UPDATES_PER_EPOCH)
        self.assertEqual(final["scheduler_state_dict"]["last_epoch"], 3 * UPDATES_PER_EPOCH)
        self.assertEqual(final["scheduler_state_dict"]["_last_lr"][0], 0.0)
        self.assertEqual(final["epoch"], 3)

    def test_periodic_checkpoints_survive_best_model_saves(self):
        files = set(os.listdir(self.run_dir))
        for step in range(9, 3 * UPDATES_PER_EPOCH + 1, 9):
            self.assertIn(f"checkpoint_step_{step}.pt", files)
        self.assertIn("best_model.pt", files)
        best = load(os.path.join(self.run_dir, "best_model.pt"))
        self.assertEqual(best["batches_done"], 0)      # end-of-epoch position

    def test_mid_epoch_resume_matches_uninterrupted_run(self):
        checkpoint = os.path.join(self.run_dir, "checkpoint_step_36.pt")   # epoch 2, batch 9
        result = run_cli(foundational_args(self.ws, self.out, 3, "--save-every", "9",
                                           "--resume-checkpoint", checkpoint))
        assert_cli_succeeded(self, result, "TRAINING COMPLETED SUCCESSFULLY")
        # Later checkpoints exist in the original directory, so the resumed
        # run writes to a new one instead of overwriting them.
        resumed_dir = only_run_dir(self.out, os.path.basename(self.run_dir) + "_resumed_step36")

        original = load(os.path.join(self.run_dir, "final_model.pt"))
        resumed = load(os.path.join(resumed_dir, "final_model.pt"))
        self.assertEqual(resumed["step"], original["step"])
        self.assertEqual(max_weight_difference(original, resumed), 0.0)
        with open(os.path.join(self.run_dir, "training_log.json")) as f:
            original_log = json.load(f)
        with open(os.path.join(resumed_dir, "training_log.json")) as f:
            self.assertEqual(json.load(f), original_log)

    def test_resume_rejects_a_different_schedule(self):
        checkpoint = os.path.join(self.run_dir, "checkpoint_step_36.pt")
        result = run_cli(foundational_args(self.ws, self.out, 5, "--save-every", "9",
                                           "--resume-checkpoint", checkpoint))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schedule differs", result.stdout + result.stderr)

    def test_early_stopping_metadata_and_finished_resume(self):
        out = os.path.join(self.ws["root"], "early_stop")
        # Any improvement below 1000% counts as insufficient: stops after epoch 2.
        result = run_cli(foundational_args(
            self.ws, out, 6, "--save-every", "1000", "--early-stopping",
            "--early-stopping-patience", "1", "--early-stopping-epsilon", "10"))
        assert_cli_succeeded(self, result, "Early stopping triggered at epoch 2")
        run_dir = only_run_dir(out, "foundational_")
        with open(os.path.join(run_dir, "training_summary.json")) as f:
            summary = json.load(f)
        self.assertEqual(summary["epochs_completed"], 2)
        self.assertTrue(summary["early_stopped"])
        self.assertEqual(load(os.path.join(run_dir, "final_model.pt"))["epoch"], 2)

        # Resuming a finished run does nothing and does not crash.
        result = run_cli(foundational_args(
            self.ws, out, 6, "--save-every", "1000", "--early-stopping",
            "--early-stopping-patience", "1", "--early-stopping-epsilon", "10",
            "--resume-checkpoint", os.path.join(run_dir, "final_model.pt")))
        assert_cli_succeeded(self, result, "already finished training")


class GRPORunTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ws = build_workspace()
        result = run_cli(foundational_args(cls.ws, os.path.join(cls.ws["root"], "sft"), 1,
                                           "--save-every", "1000"))
        if "TRAINING COMPLETED SUCCESSFULLY" not in result.stdout + result.stderr:
            raise RuntimeError((result.stdout + result.stderr)[-4000:])
        cls.sft = os.path.join(only_run_dir(os.path.join(cls.ws["root"], "sft"), "foundational_"),
                               "final_model.pt")
        cls.out = os.path.join(cls.ws["root"], "grpo")
        result = run_cli(cls.grpo_args(cls.out))
        if "GRPO TRAINING COMPLETED SUCCESSFULLY" not in result.stdout + result.stderr:
            raise RuntimeError((result.stdout + result.stderr)[-4000:])
        cls.run_dir = only_run_dir(cls.out, "grpo_")

    @classmethod
    def tearDownClass(cls):
        remove_workspace(cls.ws)

    @classmethod
    def grpo_args(cls, out, *extra):
        # 60 prompts - 6 validation = 54 -> 27 batches of 2 -> 9 updates/epoch
        # with accumulation 3.
        return ["run_grpo_training.py", "--tokenizer", cls.ws["tokenizer"],
                "--instruction-corpus", cls.ws["instruction"], "--sft-checkpoint", cls.sft,
                "--output-dir", out, "--num-epochs", "2", "--batch-size", "2",
                "--num-candidates", "2", "--max-gen-length", "32", "--learning-rate", "1e-4",
                "--val-size", "6", "--eval-every", "4", "--save-every", "3",
                "--gradient-accumulation-steps", "3", "--mixed-precision", "no", *extra]

    def test_schedule_validation_and_best_model(self):
        final = load(os.path.join(self.run_dir, "final_model.pt"))
        self.assertEqual(final["step"], 18)
        self.assertEqual(final["scheduler_state_dict"]["_last_lr"][0], 0.0)
        self.assertIn("best_model.pt", os.listdir(self.run_dir))
        log = final["training_state"]["training_log"]
        evaluated_steps = sorted({e["step"] for e in log if e["metrics"]["val_reward_rate"] is not None})
        self.assertEqual(evaluated_steps, [4, 8, 9, 12, 16, 18])   # every 4 updates + each epoch end

    def test_mid_epoch_resume_matches_uninterrupted_run(self):
        checkpoint = os.path.join(self.run_dir, "checkpoint_step_6.pt")
        result = run_cli(self.grpo_args(self.out, "--resume-checkpoint", checkpoint))
        assert_cli_succeeded(self, result, "GRPO TRAINING COMPLETED SUCCESSFULLY")
        resumed_dir = only_run_dir(self.out, os.path.basename(self.run_dir) + "_resumed_step6")
        original = load(os.path.join(self.run_dir, "final_model.pt"))
        resumed = load(os.path.join(resumed_dir, "final_model.pt"))
        self.assertEqual(max_weight_difference(original, resumed), 0.0)
        self.assertEqual(resumed["training_state"]["best_reward_rate"],
                         original["training_state"]["best_reward_rate"])

    def test_resume_rejects_a_different_sft_checkpoint(self):
        other_sft = os.path.join(self.ws["root"], "other_sft.pt")
        checkpoint = load(self.sft)
        checkpoint["model_state_dict"]["token_embedding.weight"] += 1.0
        torch.save(checkpoint, other_sft)
        args = self.grpo_args(self.out, "--resume-checkpoint",
                              os.path.join(self.run_dir, "checkpoint_step_6.pt"))
        args[args.index("--sft-checkpoint") + 1] = other_sft
        result = run_cli(args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SFT checkpoint differs", result.stdout + result.stderr)

    def test_early_stopping(self):
        out = os.path.join(self.ws["root"], "grpo_early_stop")
        result = run_cli(self.grpo_args(out, "--early-stopping", "--early-stopping-patience", "1",
                                        "--early-stopping-epsilon", "1000"))
        assert_cli_succeeded(self, result, "Early stopping triggered at step 8")
        run_dir = only_run_dir(out, "grpo_")
        with open(os.path.join(run_dir, "grpo_training_summary.json")) as f:
            summary = json.load(f)
        self.assertTrue(summary["early_stopped"])
        self.assertEqual(summary["total_steps"], 8)
        self.assertEqual(summary["epochs_completed"], 0)


if __name__ == "__main__":
    unittest.main()
