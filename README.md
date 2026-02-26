# Arithmetic LLM

A from-scratch transformer language model that learns to solve arithmetic expressions with step-by-step reasoning.

## About

This project builds a small language model from scratch -- including the tokenizer, data pipeline, model architecture, and training loop -- to evaluate arithmetic expressions such as `(5 + 3) * (12 - 4)` and produce structured, step-by-step solutions:

```
Input:  Evaluate: 5 + (10 - 3)

Output:
<think>
Step 1: 10 - 3 = 7
Expression now: 5 + 7
Step 2: 5 + 7 = 12
Expression now: 12
</think>
Final Result: 12
```

The training pipeline follows a four-stage progression:

1. **Foundational Pre-training** -- Next-token prediction on plain-text arithmetic corpus
2. **Instruction Fine-tuning** -- Supervised fine-tuning on prompt/response pairs (full-parameter or LoRA)
3. **GRPO Reinforcement Learning** -- Group Relative Policy Optimization with verifiable rewards

All training supports multi-GPU distributed execution (including device auto-identification) via [HuggingFace Accelerate](https://huggingface.co/docs/accelerate).

## Project Structure

```
Arithmetic_LLM/
├── configs/                 # Training and LoRA configurations
├── data/                    # Data generation, tokenizer, and data loading
├── model/                   # Transformer architecture and LoRA utilities
├── training/                # Training loops (foundational, instruction, LoRA, GRPO)
├── evaluation/              # Model evaluation and arithmetic verification
├── tools/                   # Interactive solver and utilities
├── tests/                   # Unit tests
├── run_foundational_training.py
├── run_instruction_training.py
├── run_instruction_training_lora.py
├── run_grpo_training.py
├── run_evaluation.py
├── train_tokenizer.py
├── demo.py
├── run_exp.sh               # Full pipeline script
└── single_gpu/              # Archived single-GPU version
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

### Run the Full Pipeline

```bash
bash run_exp.sh
```

This runs data generation, tokenizer training, all training stages, and evaluation end-to-end. See [Detailed_Usage.md](Detailed_Usage.md) for individual command reference.

### Interactive Demo

After training, solve expressions interactively:

```bash
python demo.py --model-path models/<your_model>/best_model.pt --tokenizer-path data/tokenizer
```

For example,

```bash
python demo.py --model-path models/grpo_YYMMDD_HHMMSS/best_model.pt --tokenizer-path data/tokenizer
```





## Documentation

| Document | Description |
|----------|-------------|
| [Detailed_Usage.md](Detailed_Usage.md) | Full command reference for every pipeline stage |
| [README_DEV.md](README_DEV.md) | Development log: architecture decisions, refactoring challenges, and lessons learned |
| [Replicate Arithmetic_LLM.md](Replicate%20Arithmetic_LLM.md) | Step-by-step replication guide |


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
