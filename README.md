# Arithmetic LLM

A decoder-only transformer language model built from scratch that learns to solve arithmetic expressions through step-by-step reasoning.

```
Input:  Evaluate: 8 + ((19 + 17) - (15 - 11))

Output:
<think>
Step 1: 19 + 17 = 36
Step 2: 15 - 11 = 4
Step 3: 36 - 4 = 32
Step 4: 8 + 32 = 40
Expression now: 40
</think>
Final Result: 40
```

Everything is implemented from scratch — tokenizer, data pipeline, model architecture, training loops, and evaluation — with no reliance on pre-trained weights or external model libraries.

## Highlights

- **End-to-end training pipeline**: Four progressive stages from pre-training to reinforcement learning
- **Chain-of-thought reasoning**: Model generates interpretable intermediate steps, not just final answers
- **Multi-GPU distributed training**: Built on [HuggingFace Accelerate](https://huggingface.co/docs/accelerate) with mixed precision, gradient accumulation, and synchronized early stopping
- **Parameter-efficient fine-tuning**: LoRA adaptation with checkpoint merging support
- **GRPO reinforcement learning**: Group Relative Policy Optimization with verifiable arithmetic rewards

## Training Pipeline

The model is trained in four stages. Instruction fine-tuning can be done with all parameters or with LoRA; both start from the foundational model, and GRPO starts from the full instruction-tuned model (this is what `run_exp.sh` does):

```
                         ┌→ Instruction FT (full parameters) ─→ GRPO RL
Foundational             │   supervised on prompt/response       optimizes with verifiable
pre-training ────────────┤   pairs with chain of thought         arithmetic rewards
next-token prediction    │
on the arithmetic corpus └→ LoRA FT (parameter-efficient)
                             low-rank adapters on attention/FFN
```

### Results

| Training Stage | Accuracy | Parse Rate | Change vs. previous row |
|---|---|---|---|
| Foundational pre-training | 0.0% | 0.0% | — |
| Instruction fine-tuning | 41.4% | 88.8% | +41.4 pts |
| LoRA fine-tuning | 43.4% | 83.8% | +2.0 pts |
| **GRPO** | **72.7%** | 76.6% | **+29.3 pts** (vs. instruction) |

Provenance: these runs (February 1, 2026) were trained and evaluated with the **original single-GPU code**, before the Accelerate refactor. Each model was evaluated on 1,000 generated test expressions (100 for the foundational model) with the evaluator's defaults at the time: operands 1–20, nesting depth up to 5, sampling at temperature 0.8. The checkpoints and per-model details are listed in [docs/Project_Summary.md](docs/Project_Summary.md). The distributed pipeline has not been re-benchmarked yet; see [docs/Audit_and_Fix_Plan.md](docs/Audit_and_Fix_Plan.md).

## Architecture

The following parameters are adjustable.

| Component | Detail |
|---|---|
| Model type | Decoder-only transformer (causal LM) |
| Embedding dim | 256 |
| Attention heads | 8 |
| Layers | 6 |
| FFN hidden dim | 1,024 |
| Max sequence length | 512 tokens |
| Tokenizer | Custom BPE (vocab size ~1,000) |
| Weight tying | Embedding ↔ output projection |
| LoRA rank | 16 (alpha = 32) |

The tokenizer preserves arithmetic operators (`+`, `-`, `(`, `)`) as atomic symbols that are never merged during BPE training. Special tokens include `<think>` / `</think>` markers for chain-of-thought boundaries.

## Project Structure

```
Arithmetic_LLM/
├── model/                   # Transformer architecture, LoRA layers, adapter merging
├── training/                # Training loops: foundational, instruction, LoRA, GRPO
├── data/                    # Data generation, BPE tokenizer, DataLoader
├── configs/                 # Training, LoRA, and GRPO configurations
├── evaluation/              # Distributed evaluation harness, arithmetic verifier
├── tools/                   # Interactive solver, diagnostics
├── tests/                   # Unit and end-to-end training tests (CPU)
│
├── run_exp.sh               # Full pipeline script (single command)
├── run_foundational_training.py
├── run_instruction_training.py
├── run_instruction_training_lora.py
├── run_grpo_training.py
├── run_evaluation.py
├── train_tokenizer.py
├── demo.py                  # Batch demo with accuracy report
│
├── docs/                    # Usage guide, development log, audit records
└── single_gpu/              # Archived original single-GPU version
```

`run_evaluation.py` writes its metrics and sample outputs to `evaluation_results/` (created on first use).

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/Tian-Yizhou/Arithmetic_LLM.git
cd Arithmetic_LLM
pip install -r requirements.txt
```

Note: no special packages are needed, you may also run it with your own environment.

### Run the Full Pipeline

```bash
bash run_exp.sh
```

This executes the entire pipeline end-to-end: data generation → tokenizer training → foundational pre-training → instruction fine-tuning → LoRA fine-tuning → GRPO training → evaluation.

### Interactive Demo

After training, solve expressions interactively:

```bash
python -m tools.run_interactive \
  --model-path models/your_model/best_model.pt \
  --tokenizer-path data/tokenizer
```

```
Enter expression: 5 + (10 - 3)

Reasoning Steps:
  Step 1: 10 - 3 = 7
  Expression now: 5 + 7
  Step 2: 5 + 7 = 12
  Expression now: 12

Final Result: 12
```

### Batch Evaluation

```bash
python demo.py \
  --model-path models/your_model/best_model.pt \
  --tokenizer-path data/tokenizer \
  --num-examples 10
```

## Distributed Training

All training scripts support multi-GPU execution via `accelerate launch`:

```bash
accelerate launch run_foundational_training.py \
  --corpus-path data/foundational_corpus.txt \
  --tokenizer-path data/tokenizer \
  --output-dir models \
  --num-epochs 10
```

Key distributed features:
- **Automatic device management** — no manual `.to(device)` calls
- **Mixed precision** — `--mixed-precision fp16` (default), `bf16` or `no`
- **Gradient accumulation** — configurable effective batch size without code changes
- **Correct schedules on any number of GPUs** — the learning-rate schedule is sized in real optimizer updates per process
- **Synchronized early stopping** — all processes agree on when to stop via gathered metrics (all four stages)
- **Exact checkpoint resumption** — a run resumed mid-epoch continues with the same batches, random state and early-stopping state as an uninterrupted run
- **Portable checkpoints** — DDP-unwrapped before saving, written atomically, loadable in any context
- **Distributed evaluation** — test set sharded across GPUs, results aggregated

Checkpointing supports single-device and DDP training (FSDP/DeepSpeed are rejected at start-up). Resume an interrupted run with the same settings:

```bash
accelerate launch run_foundational_training.py \
  --resume-checkpoint models/foundational_YYYYMMDD_HHMMSS_ffffff/checkpoint_step_5000.pt \
  ...
```

## Tests

The tests run on CPU in under a minute and need no GPU or prepared data (they generate a tiny corpus and tokenizer):

```bash
python -m unittest discover -s tests -p "test_*.py" -t .
```

They cover configuration precedence, exact mid-epoch resume (bit-identical weights), checkpoint survival, early stopping, the GRPO optimizer lifecycle and accumulation, and real two-process runs (`torchrun`, gloo backend).

## Documentation

| Document | Description |
|---|---|
| [Detailed_Usage.md](docs/Detailed_Usage.md) | Full command reference for every pipeline stage |
| [README_DEV.md](docs/README_DEV.md) | Development log — distributed refactoring, technical challenges, and solutions |
| [Audit_and_Fix_Plan.md](docs/Audit_and_Fix_Plan.md) | Audit of the refactor: defects found, fixes, and how each was verified |
| [Project_Summary.md](docs/Project_Summary.md) | Training results, evaluation analysis, and performance breakdown |
| [Replicate Arithmetic_LLM.md](docs/Replicate%20Arithmetic_LLM.md) | Step-by-step replication guide |

## Tech Stack

- **[PyTorch](https://pytorch.org/)** — model, training loops, checkpointing
- **[HuggingFace Accelerate](https://huggingface.co/docs/accelerate)** — distributed training, mixed precision, gradient accumulation
- GRPO rewards come from a small rule-based verifier (`evaluation/arithmetic_verifier.py`) that extracts the final answer and compares it with the ground truth

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
