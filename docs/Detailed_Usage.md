# Detailed Usage

This document covers the full pipeline for training and evaluating the Arithmetic LLM, from data generation to interactive inference. All commands use HuggingFace Accelerate for distributed training. The `single_gpu/` directory contains equivalent single-GPU scripts that can be run directly with `python`.

> **Working directory**: All commands assume you are in the project root directory.


## 1. Data Generation

Generate arithmetic expression corpora used for training. The pipeline produces two types of corpora: a **foundational corpus** (plain text for next-token pre-training) and an **instruction corpus** (prompt-response pairs for supervised fine-tuning).

### 1.1 Foundational Corpus

```bash
python -m data.generate_foundational_plaintext \
  --num-samples 100000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.05 \
  --output-txt data/foundational_corpus.txt
```

**Parameters:**
- `--num-samples`: Number of expression-evaluation pairs to generate (required)
- `--max-depth`: Maximum depth of the expression tree (default: 5). Higher depth produces more complex, deeply nested expressions (e.g. depth 3 might yield `(2 + 3) * (4 - 1)`, depth 5 could yield `((2 + 3) * 4 - (1 + 2)) * (5 - 3)`)
- `--num-range`: Min and max of operand values (default: 1 20)
- `--invalid-rate`: Fraction of intentionally malformed expressions included to teach the model robustness against invalid inputs (default: 0.1)
- `--output-txt`: Path to save the shuffled plain-text corpus
- `--seed`: Random seed for reproducibility

**Output Format (each line is one training sample):**
```
Evaluate: 5 + (10 - 3)
<think> Step 1: 10 - 3 = 7 Expression now: 5 + 7 Step 2: 5 + 7 = 12 Expression now: 12 </think> Final Result: 12
```


### 1.2 Instruction Corpus

```bash
python -m data.generate_instruction_corpus_mixed \
  --num-samples 50000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.1 \
  --output-mixed data/instruction_corpus.txt
```

**Parameters:**
- `--output-mixed`: Path to save the mixed instruction corpus
- Other parameters are the same as the foundational corpus

This script generates two instruction sets (one with invalid expressions, one without), then merges and shuffles them together.

**Output Format (multi-line prompt/response pairs):**
```
Evaluate: 5 + (10 - 3) <think>
<think>
Step 1: 10 - 3 = 7
Expression now: 5 + 7
Step 2: 5 + 7 = 12
Expression now: 12
</think>
Final Result: 12
```


### 1.3 Test Data Generation

Generate a separate test set for evaluation. Use a wider `--num-range` than the training set to test generalization.

```bash
python -m data.generate_corpus \
  --instruction-only \
  --num-samples 1000 \
  --max-depth 4 \
  --num-range 1 30 \
  --invalid-rate 0.01 \
  --output-instruction data/instruction_corpus_test.txt
```

Additional parameters for `generate_corpus.py`:
- `--foundational-only` / `--instruction-only`: Generate only one type of corpus
- `--output-foundational`: Path for foundational corpus output
- `--output-instruction`: Path for instruction corpus output


### 1.4 Tokenizer Training

Train a BPE (Byte-Pair Encoding) tokenizer on the arithmetic corpus. The tokenizer learns subword merges from the corpus while protecting atomic symbols (operators, digits, parentheses) from being merged together.

```bash
python -m train_tokenizer \
  --corpus-path data/foundational_corpus.txt \
  --vocab-size 1000 \
  --output-dir data/tokenizer
```

**Parameters:**
- `--corpus-path`: Path to training corpus (required)
- `--vocab-size`: Target vocabulary size (default: 1000). For arithmetic, the actual distinct tokens are ~130-200, so 1000 provides ample headroom
- `--output-dir`: Directory to save tokenizer files (default: `data/tokenizer`)

**Special Tokens:**
`<pad>`, `<unk>`, `<bos>`, `<eos>`, `<think>`, `</think>`

**Corpus Size Recommendations:**
| Size | Use Case |
|------|----------|
| 10,000 | Quick testing, may underfit |
| 50,000 | Good balance of speed and accuracy |
| 100,000+ | Best generalization, longer training |


## 2. Model Training

The training pipeline has four stages, each building on the previous one:

1. **Foundational** -- Pre-train the transformer on plain-text arithmetic (next-token prediction)
2. **Instruction Fine-tuning** -- Fine-tune on prompt/response pairs (full-parameter or LoRA)
3. **GRPO** -- Reinforce correct reasoning with reward-based optimization

All distributed training scripts are launched with `accelerate launch` (e.g. `accelerate launch --num_processes 4 run_foundational_training.py ...`). Training uses fp16 mixed precision by default; choose it with the script's `--mixed-precision` flag (`fp16`, `bf16` or `no`), which takes precedence over `accelerate launch --mixed_precision`.

Checkpointing supports single-device and DDP (multi-GPU) training. The scripts refuse to start under FSDP or DeepSpeed, whose sharded state they do not save.

**Configuration precedence (supervised stages):** an explicit command-line flag overrides the value in the `--config` JSON file, which overrides the stage default. The run directory's `training_config.json` records the values actually used and can be passed back with `--config`.


### 2.1 Train Foundational Model

Pre-train the base transformer model on arithmetic expressions via next-token prediction.

```bash
accelerate launch run_foundational_training.py \
  --corpus-path data/foundational_corpus.txt \
  --tokenizer-path data/tokenizer \
  --output-dir models \
  --num-epochs 8 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.05 \
  --max-seq-length 512 \
  --gradient-accumulation-steps 1 \
  --early-stopping \
  --early-stopping-patience 2
```

**Training Parameters:**
- `--corpus-path`: Path to training corpus (required)
- `--tokenizer-path`: Path to tokenizer directory (required)
- `--output-dir`: Directory to save checkpoints (default: `models`)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size per device (default: 32)
- `--num-epochs`: Number of training epochs (default: 10)
- `--warmup-ratio`: Fraction of all optimizer updates used for linear learning-rate warm-up from 0 to the target rate (default: 0.05)
- `--warmup-steps`: Exact number of warm-up updates; overrides `--warmup-ratio` when given (0 disables warm-up). A value larger than the total number of updates is rejected
- `--gradient-clip`: Maximum gradient norm; gradients exceeding this are scaled down to prevent training instability (default: 1.0)
- `--gradient-accumulation-steps`: Accumulate gradients over N forward passes before updating weights, effectively simulating a larger batch size with limited GPU memory (default: 1)
- `--mixed-precision`: `fp16` (default), `bf16` or `no`
- `--seed`: Seed for the train/validation split, the batch order and initialization (default: 42)
- `--save-every`: Save a checkpoint every N optimizer updates (default: 1000)
- `--early-stopping` / `--no-early-stopping`: Enable early stopping based on the validation loss improvement ratio (default: off)
- `--early-stopping-patience`: Number of consecutive epochs with insufficient improvement before stopping (default: 3)
- `--early-stopping-epsilon`: Minimum loss improvement ratio required to continue training (default: 1e-4)
- `--resume-checkpoint`: Path to a checkpoint of this stage to resume an interrupted run (see [section 5](#5-resuming-from-checkpoints))
- `--config`: Path to a training configuration JSON file; explicit flags override its values
- `--model-config`: Path to model architecture JSON file (overrides CLI model parameters)

The learning-rate schedule is sized in optimizer updates per process: `ceil(batches per process / accumulation steps) × epochs`. It therefore ends at the last update on any number of GPUs. The validation loss is the mean loss per predicted token over the whole validation split, the same on every process.

**Model Architecture Parameters:**
- `--d-model`: Embedding dimension (default: 256)
- `--nhead`: Number of attention heads (default: 8)
- `--num-layers`: Number of transformer decoder layers (default: 6)
- `--dim-feedforward`: Hidden dimension of the feed-forward network in each layer (default: 1024)
- `--dropout`: Dropout rate (default: 0.1)
- `--max-seq-length`: Maximum sequence length the model can process (default: 512)

**Output:**
Training creates a timestamped directory (e.g. `models/foundational_20260225_143000_123456/`) containing:
- `best_model.pt` -- Best model checkpoint (lowest validation loss)
- `final_model.pt` -- Final model checkpoint
- `checkpoint_step_N.pt` -- Intermediate checkpoints (every `--save-every` updates)
- `training_config.json`, `model_config.json` -- Saved configurations
- `run_info.json` -- Resolved schedule (processes, batches and updates per epoch, warm-up) and data identity
- `training_log.json` -- Per-epoch metrics
- `training_summary.json` -- Final training summary, including the epochs actually completed and whether early stopping ended the run


### 2.2 Instruction Fine-tuning

Fine-tune the foundational model on instruction-formatted prompt/response pairs (full-parameter update).

```bash
accelerate launch run_instruction_training.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models \
  --num-epochs 10 \
  --batch-size 32 \
  --learning-rate 5e-5
```

**Parameters:**
- `--instruction-corpus-path`: Path to instruction corpus (required)
- `--foundational-checkpoint`: Path to foundational model checkpoint from step 2.1 (required)
- `--learning-rate`: Default 5e-5 (lower than foundational training to avoid catastrophic forgetting)
- `--num-epochs`: Default 5
- `--save-every`: Default 500
- `--resume-checkpoint`: Path to an instruction-training checkpoint to resume training

Other training parameters (`--warmup-ratio`, `--gradient-accumulation-steps`, `--mixed-precision`, `--seed`, `--early-stopping`, etc.) are the same as in 2.1.


### 2.3 LoRA Instruction Fine-tuning

Fine-tune with Low-Rank Adaptation (LoRA) adapters for parameter-efficient training. Only a small number of additional parameters are trained while the base model is frozen.

```bash
accelerate launch run_instruction_training_lora.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models \
  --num-epochs 10 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-target-modules attention,feedforward \
  --lora-dropout 0.1 \
  --save-merged-model
```

**LoRA-specific Parameters:**
- `--lora-rank`: Rank of the low-rank decomposition (default: 8). Lower rank = fewer trainable parameters but less expressive capacity
- `--lora-alpha`: Scaling factor for LoRA updates (default: 16.0). The effective update scale is `alpha / rank`; a common choice is `alpha = 2 * rank`
- `--lora-target-modules`: Comma-separated list of modules to apply LoRA to (default: `attention`). Options: `attention`, `feedforward`
- `--lora-dropout`: Dropout applied to LoRA layers (default: 0.0)
- `--save-merged-model`: After training, merge LoRA weights into the base model and save a standalone checkpoint for direct inference
- `--resume-checkpoint`: Path to a LoRA-training checkpoint to resume training

LoRA settings follow the same precedence as the training settings: an explicit `--lora-*` flag overrides the `lora_config` in a `--config` JSON file, which overrides the defaults above.

Other training parameters are the same as in 2.2.

**Output:**
- `lora_adapter.pt` -- LoRA adapter weights only
- `merged_model.pt` -- Standalone merged model (when `--save-merged-model` is used)

If you saved only the adapter, merge it later with:
```bash
python -m model.merge_lora_adapter \
  --base-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --adapter-path models/instruction_lora_YYYYMMDD_HHMMSS/lora_adapter.pt \
  --output-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt
```


### 2.4 GRPO Training

Train with Group Relative Policy Optimization (GRPO). For each prompt, the model generates multiple candidate responses and receives a verifiable reward (correct or incorrect). The policy is updated to favor higher-reward candidates.

```bash
accelerate launch run_grpo_training.py \
  --instruction-corpus data/instruction_corpus.txt \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models/ \
  --num-epochs 3 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --num-candidates 4 \
  --temperature 0.8 \
  --kl-penalty-coef 0.05
```

**Parameters:**
- `--sft-checkpoint`: Path to the supervised fine-tuned model from step 2.2 or 2.3 (required). "SFT" = Supervised Fine-Tuning
- `--data-mode`: `instruction` loads prompts from a corpus file; `generated` creates random expressions on-the-fly (default: `instruction`)
- `--num-candidates`: Number of candidate responses sampled per prompt for group ranking (default: 4)
- `--temperature`: Sampling temperature for candidate generation; higher = more diverse candidates (default: 0.8)
- `--top-k`: Top-k sampling; only sample from the top k most probable tokens (default: 50)
- `--top-p`: Nucleus sampling; sample from the smallest set of tokens whose cumulative probability exceeds p (default: 0.9)
- `--kl-penalty-coef`: KL divergence penalty coefficient; regularizes the policy to stay close to the reference model, preventing mode collapse (default: 0.05)
- `--max-gen-length`: Maximum length of a candidate sequence, prompt included, in tokens (default: 512; capped at the model's `max_seq_length - 2`)
- `--warmup-ratio` / `--warmup-steps`: As in 2.1 (default: 5% of all updates)
- `--gradient-accumulation-steps`: Number of prompt batches per optimizer update (default: 1). The last, shorter group of an epoch also produces an update
- `--val-size`: Number of prompts held out for validation (default: 256). Validation measures the share of held-out prompts answered correctly (one sampled answer each, with a fixed seed), and drives best-model saving and early stopping. `0` disables validation
- `--eval-every`: Run validation every N optimizer updates, and always at the end of each epoch (default: 250)
- `--save-every`: Save a checkpoint every N optimizer updates (default: 500)
- `--seed`: Seed for shuffling the prompts, the validation split and sampling (default: 42)
- `--mixed-precision`: `fp16` (default), `bf16` or `no`
- `--log-every`: Print training metrics every N steps (default: 50)
- `--candidate-sub-batch-size`: Process candidates in smaller sub-batches to reduce peak GPU memory; set this if you encounter OOM errors (default: None, process all at once)
- `--filter-invalid-instruction` / `--no-filter-invalid-instruction`: Filter out instruction entries with invalid or mismatched expressions (default: enabled)
- `--resume-checkpoint`: Path to a GRPO checkpoint to resume training. Pass the same `--sft-checkpoint` as the original run: it provides the fixed reference model for the KL penalty, and a different file is rejected
- `--early-stopping`, `--early-stopping-patience`, `--early-stopping-epsilon`: As in 2.1, applied to the validation reward rate and counted in evaluations rather than epochs (requires `--val-size` > 0)

The prompts are split so that every process gets the same number of training and validation prompts (at most `num_processes - 1` prompts per part are dropped); otherwise the processes would fall out of step and hang.

**Parameters for `generated` data mode:**
- `--num-samples`: Number of expressions to generate (default: 1000)
- `--max-depth`: Maximum expression tree depth (default: 5)
- `--num-range-min` / `--num-range-max`: Operand value range (default: 1 / 20)

**Output:**
- `checkpoint_step_N.pt` -- Periodic checkpoints
- `best_model.pt` -- Checkpoint with the best validation reward rate
- `final_model.pt` -- Final checkpoint
- `grpo_training_log.json` -- Per-batch training metrics, with the validation reward rate where one was measured
- `grpo_training_summary.json` -- Steps, epochs completed, early-stopping outcome, best validation reward rate, schedule and data identity


## 3. Model Evaluation

Evaluate a trained model by generating solutions for random arithmetic expressions and checking correctness. The evaluation script supports distributed inference via Accelerate.

```bash
accelerate launch run_evaluation.py \
  --model-path "$MODEL_PATH" \
  --tokenizer-path data/tokenizer \
  --num-samples 1000 \
  --max-gen-length 512 \
  --batch-size 1
```

**Parameters:**
- `--model-path`: Path to model checkpoint (required)
- `--tokenizer-path`: Path to tokenizer directory (required)
- `--base-checkpoint`: Path to base model checkpoint, required when evaluating a LoRA adapter directly (without merging first)
- `--num-samples`: Number of test expressions to generate (default: 1000)
- `--max-depth`: Maximum depth of test expressions (default: 5). Note that `run_exp.sh` generates training data with depth 4, so the default also tests deeper expressions than the model was trained on; pass `--max-depth 4` to evaluate in-distribution
- `--num-range`: Number range for test expressions (default: 1 20)
- `--max-gen-length`: Maximum generation length in tokens (default: 512)
- `--batch-size`: Batch size for inference (default: 1)
- `--output-dir`: Directory to save results (default: `evaluation_results`)

**Metrics:**
- **Exact Match Accuracy** -- Percentage of correct final results
- **Parse Success Rate** -- Percentage of parseable model outputs
- **Average Generation Length** -- Mean number of tokens generated

**Output:**
Evaluation creates timestamped files in the output directory:
- `evaluation_metrics_YYYYMMDD_HHMMSS.json` -- Detailed metrics
- `sample_outputs_YYYYMMDD_HHMMSS.json` -- Sample model outputs
- `evaluation_summary_YYYYMMDD_HHMMSS.txt` -- Human-readable summary


### 3.1 Evaluate Foundational Model

```bash
accelerate launch run_evaluation.py \
  --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000 \
  --max-depth 4
```

### 3.2 Evaluate Instruction Model

```bash
accelerate launch run_evaluation.py \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000
```

### 3.3 Evaluate LoRA Model

If you used `--save-merged-model` during training, evaluate the merged checkpoint directly:
```bash
accelerate launch run_evaluation.py \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000
```

To evaluate an unmerged LoRA adapter, provide the base checkpoint:
```bash
accelerate launch run_evaluation.py \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/lora_adapter.pt \
  --base-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000
```

### 3.4 Evaluate GRPO Model

```bash
accelerate launch run_evaluation.py \
  --model-path models/grpo_YYYYMMDD_HHMMSS_ffffff/best_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000
```

**Performance Reference:**

| Accuracy | Interpretation |
|----------|----------------|
| 80%+ | Excellent |
| 60-80% | Good |
| 40-60% | Moderate -- consider more training data or epochs |
| <40% | Poor -- check data quality and hyperparameters |

Parse success rate should be >90%. A low rate indicates the model needs better instruction tuning.


## 4. Interactive Solver

Use a trained model interactively to solve arithmetic problems in the terminal.

```bash
python -m tools.run_interactive \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer
```

**Parameters:**
- `--model-path`: Path to an instruction-tuned or GRPO model checkpoint (required)
- `--tokenizer-path`: Path to tokenizer directory (required)
- `--device`: Device for inference -- `cuda`, `mps`, `cpu`, or `auto` (default: `auto`)

**Usage:**
```
Enter expression: 5 + (10 - 3)

------------------------------------------------------------
SOLUTION:
------------------------------------------------------------

Reasoning Steps:
  Step 1: 10 - 3 = 7
  Expression now: 5 + 7
  Step 2: 5 + 7 = 12
  Expression now: 12

Final Result: 12
------------------------------------------------------------

Enter expression: exit
```

Type `exit`, `quit`, or `q` to exit. Press `Ctrl+C` to interrupt.


## 5. Resuming from Checkpoints

All four training scripts resume an interrupted run with `--resume-checkpoint`. Pass the same data and training settings as the original run, and point to any checkpoint of that run (`checkpoint_step_N.pt`, `best_model.pt` or `final_model.pt`):

```bash
accelerate launch run_foundational_training.py \
  --corpus-path data/foundational_corpus.txt \
  --tokenizer-path data/tokenizer \
  --output-dir models \
  --num-epochs 10 \
  --batch-size 32 \
  --resume-checkpoint models/foundational_YYYYMMDD_HHMMSS_ffffff/checkpoint_step_5000.pt
```

What is restored: model weights, optimizer state (Adam moments), learning-rate schedule position, fp16 loss scale, each process's random-number state, the exact position in the data (epoch and batch, so a checkpoint from the middle of an epoch continues with the remaining batches of that epoch), the best validation result, the early-stopping counters and the training log. For GRPO, the reference model is the `--sft-checkpoint` model, as in the original run. A resumed run ends with the same weights as an uninterrupted run with the same settings (the tests check this to the bit on CPU; GPU kernels may introduce tiny numerical differences).

Where it writes: into the checkpoint's own run directory. If that directory already contains later progress (a higher-step checkpoint, or a final model when you resume from an earlier checkpoint), a new sibling directory `<run>_resumed_step<N>_<timestamp>` is used instead, so newer files are never overwritten.

What is checked: the resume is refused, with a list of the differences, if the number of processes, batch size, accumulation, number of epochs, warm-up, learning rate, corpus, tokenizer or seed differ from the original run (and, for GRPO, the SFT checkpoint file). These determine the learning-rate schedule and the data order, so continuing with different values would not continue the same run. To train further with different settings, start a new run from the checkpoint's weights instead (e.g. as `--foundational-checkpoint`).

Older checkpoints (written before exact resume was added) still work: weights, optimizer and scheduler are restored and the position is derived from the update count, but the original data order, random state and early-stopping state cannot be recovered. Such a resume always writes to a new directory.

> **LoRA checkpoints:** a LoRA checkpoint can resume LoRA training only. Its weights can be loaded by the evaluator, merged into a plain model, or used as a starting checkpoint, but full-parameter training cannot be *resumed* from it, because its optimizer state covers only the adapter parameters.


## Acknowledgments

- Transformer architecture adapted from the TinyStories project
- BPE tokenization based on HuggingFace tokenizers library
- Distributed training powered by HuggingFace Accelerate
