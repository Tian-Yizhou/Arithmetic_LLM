"""Shared fixtures for the training tests.

Corpora and tokenizers are git-ignored, so each test module builds a tiny
workspace in a temporary directory: a few hundred generated expressions, a
small BPE tokenizer and (where needed) a tiny trained checkpoint.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tiny model used by every CLI run in the tests.
TINY_MODEL_ARGS = [
    "--d-model", "32", "--nhead", "2", "--num-layers", "1",
    "--dim-feedforward", "64", "--max-seq-length", "256",
]


def build_workspace(num_samples: int = 240, seed: int = 0) -> dict:
    """Create corpora and a tokenizer in a new temporary directory."""
    sys.path.insert(0, REPO_ROOT)
    from data.arithmetic_tokenizer import ArithmeticBPETokenizer
    from data.generator import ExpressionGenerator
    from evaluation.evaluator import eval_expression

    root = tempfile.mkdtemp(prefix="arith_llm_test_")
    state = random.getstate()
    random.seed(seed)
    generator = ExpressionGenerator(max_depth=2, num_range=(1, 20), invalid_rate=0.0)
    entries = []
    while len(entries) < num_samples:
        result = eval_expression(generator.generate())
        if result["answer"] != "ERROR":
            entries.append(result)
    random.setstate(state)

    foundational = os.path.join(root, "foundational.txt")
    with open(foundational, "w") as f:
        for e in entries:
            f.write(f"{e['problem']} {e['solution']}".replace("\n", " ") + "\n")

    instruction = os.path.join(root, "instruction.jsonl")
    with open(instruction, "w") as f:
        for e in entries[:60]:
            f.write(json.dumps({
                "expression": e["expression"], "problem": e["problem"],
                "solution": e["solution"], "answer": e["answer"],
            }) + "\n")

    tokenizer_dir = os.path.join(root, "tokenizer")
    tokenizer = ArithmeticBPETokenizer(vocab_size=300)
    tokenizer.train(foundational)
    tokenizer.save(tokenizer_dir)

    return {
        "root": root,
        "foundational": foundational,
        "instruction": instruction,
        "tokenizer": tokenizer_dir,
    }


def remove_workspace(workspace: dict) -> None:
    shutil.rmtree(workspace["root"], ignore_errors=True)


def run_cli(args, nproc: int = 1, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a training CLI on CPU (fp32 unless the args say otherwise).

    nproc > 1 launches real worker processes with torchrun (gloo backend).
    """
    env = dict(os.environ, ACCELERATE_USE_CPU="true", PYTHONPATH=REPO_ROOT)
    if nproc == 1:
        cmd = [sys.executable] + args
    else:
        # torch's gloo barrier selects the MPS device on Apple silicon unless
        # the CPU fallback is enabled.
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        port = str(29500 + random.randint(100, 900))
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", str(nproc), "--master_port", port] + args
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)


def assert_cli_succeeded(testcase, result: subprocess.CompletedProcess, marker: str) -> None:
    """Check a CLI run finished.

    On macOS, multi-process gloo runs can abort in a C++ destructor *after*
    the script has finished (a PyTorch teardown issue reproducible without
    this project), so the success marker is what is checked there.
    """
    output = result.stdout + result.stderr
    testcase.assertIn(marker, output, msg=output[-4000:])
    if sys.platform != "darwin":
        testcase.assertEqual(result.returncode, 0, msg=output[-4000:])


def only_run_dir(parent: str, prefix: str) -> str:
    dirs = sorted(d for d in os.listdir(parent) if d.startswith(prefix))
    return os.path.join(parent, dirs[0])


def load(path: str) -> dict:
    # Default torch.load (weights_only=True) on purpose: checkpoints must stay
    # loadable without trusting pickled code.
    return torch.load(path, map_location="cpu")


def max_weight_difference(a: dict, b: dict) -> float:
    return max(
        (a["model_state_dict"][k].float() - b["model_state_dict"][k].float()).abs().max().item()
        for k in a["model_state_dict"]
    )
