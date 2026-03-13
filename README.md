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

The model progresses through four training stages, each building on the previous:

```
Foundational          Instruction FT         LoRA FT             GRPO RL
Pre-training    →     (Full Parameter)  →    (Parameter-       → (Reinforcement
                                              Efficient)         Learning)

Next-token            Supervised on          Adapts attention     Optimizes with
prediction on         prompt/response        & FFN layers via     verifiable
arithmetic corpus     pairs with CoT         low-rank matrices    arithmetic rewards
```

### Results

| Training Stage | Accuracy | Parse Rate | Improvement |
|---|---|---|---|
| Foundational Pre-training | 0.0% | 0.0% | — |
| Instruction Fine-tuning | 44.3% | 88.8% | +41.4% |
| LoRA Fine-tuning | 46.7% | 87.2% | +2.0% |
| **GRPO (Best)** | **72.7%** | 76.6% | **+29.3%** |

Evaluated on 1,000 test expressions.

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
├── tests/                   # Unit tests
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
├── single_gpu/              # Archived original single-GPU version
└── evaluation_results/      # Saved metrics and sample outputs
```

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
- **FP16 mixed precision** — ~40% memory reduction with minimal accuracy loss
- **Gradient accumulation** — configurable effective batch size without code changes
- **Synchronized early stopping** — all processes agree on when to stop via gathered metrics
- **Portable checkpoints** — DDP-unwrapped before saving, loadable in any context
- **Distributed evaluation** — test set sharded across GPUs, results aggregated

Checkpoint resumption is supported for interrupted runs:

```bash
accelerate launch run_foundational_training.py \
  --resume-checkpoint models/foundational_YYMMDD_HHMMSS/checkpoint_step_5000.pt \
  ...
```

## Documentation

| Document | Description |
|---|---|
| [Detailed_Usage.md](Detailed_Usage.md) | Full command reference for every pipeline stage |
| [README_DEV.md](README_DEV.md) | Development log — distributed refactoring, technical challenges, and solutions |
| [Project_Summary.md](Project_Summary.md) | Training results, evaluation analysis, and performance breakdown |
| [Replicate Arithmetic_LLM.md](Replicate%20Arithmetic_LLM.md) | Step-by-step replication guide |

## Tech Stack

- **[PyTorch](https://pytorch.org/)** — model, training loops, checkpointing
- **[HuggingFace Accelerate](https://huggingface.co/docs/accelerate)** — distributed training, mixed precision, gradient accumulation
- **[SymPy](https://www.sympy.org/)** — symbolic arithmetic verification for GRPO rewards

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
