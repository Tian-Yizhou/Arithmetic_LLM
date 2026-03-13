#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# activate environment
conda activate ds

## 1. Data Preparation

### 1.1 Corpus Generation


python -m data.generate_foundational_plaintext \
  --num-samples 100000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.05 \
  --output-txt data/foundational_corpus.txt


### 1.2 Tokenizer Training


python -m data.generate_instruction_corpus_mixed \
  --num-samples 20000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.0 \
  --output-mixed data/instruction_corpus.txt


### 1.3 Sequence Analysis


python -m data.generate_corpus \
  --instruction-only \
  --num-samples 1000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.01 \
  --output-instruction data/instruction_corpus_test.txt


### 1.4 Tokenizer Training


python -m train_tokenizer \
  --corpus-path data/foundational_corpus.txt \
  --output-dir data/tokenizer \
  --vocab-size 1000


## 2. Model Training

### 2.1 Train Fundational Model


# Train fundational model
echo ">>> Starting Foundational Training..."

accelerate launch run_foundational_training.py \
  --corpus-path data/foundational_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --max-seq-length 512 \
  --batch-size 32 \
  --learning-rate 4e-4 \
  --num-epochs 10 \
  --early-stopping \
  --early-stopping-patience 2

# set foundational best model path
FOUNDATIONAL_DIR=$(ls -td models/foundational_* | head -n 1)
FOUNDATIONAL_CKPT="${FOUNDATIONAL_DIR}/best_model.pt"


### 2.2  Train Instruction-tuned Model


# Train instruction-tuned model
echo ">>> Starting Instruction Training..."

accelerate launch run_instruction_training.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint "$FOUNDATIONAL_CKPT" \
  --batch-size 32 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-5 \
  --num-epochs 20 \
  --early-stopping \
  --early-stopping-patience 2

# set intruction model path
INSTRUCTION_DIR=$(ls -td models/instruction_* | grep -v lora | head -n 1)
INSTRUCTION_CKPT="${INSTRUCTION_DIR}/best_model.pt"


### 2.3  Fine-tune with LoRA adapters


# Fine-tune with LoRA adapters
echo ">>> Starting LoRA Training..."

accelerate launch run_instruction_training_lora.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint "$FOUNDATIONAL_CKPT" \
  --num-epochs 20 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-target-modules attention \
  --lora-dropout 0.1 \
  --save-merged-model \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 1 \
  --early-stopping \
  --early-stopping-patience 2

# set LoRA adapter path
LORA_DIR=$(ls -td models/instruction_lora_* | head -n 1)
LORA_ADAPTER="${LORA_DIR}/lora_adapter.pt"
LORA_MERGED_OUTPUT="${LORA_DIR}/merged_model.pt"


### 2.4 GRPO training


# 2.4 GRPO training
echo ">>> Starting GRPO Training..."

accelerate launch run_grpo_training.py \
  --tokenizer data/tokenizer \
  --instruction-corpus data/instruction_corpus.txt \
  --sft-checkpoint "$INSTRUCTION_CKPT" \
  --output-dir models/ \
  --num-epochs 5 \
  --num-candidates 4 \
  --temperature 0.8 \
  --batch-size 8 \
  --kl-penalty-coef 0.05 \
  --early-stopping \
  --early-stopping-patience 2

# set GRPO path
GRPO_DIR=$(ls -td models/grpo_* | head -n 1)
GRPO_CKPT="${GRPO_DIR}/final_model.pt"


## 3. Model Evaluation


### 3.1 Evaluate Foundational Model


# 3.1 Evaluate Foundational Model
echo ">>> Evaluating Foundational Model..."

accelerate launch run_evaluation.py \
  --model-path "$FOUNDATIONAL_CKPT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1


### 3.2 Evaluate Instruction


# 3.2 Evaluate Instruction
echo ">>> Evaluating Instruction Model..."

accelerate launch run_evaluation.py \
  --model-path "$INSTRUCTION_CKPT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000


### 3.3 Evaluate LoRA (Includes Merge Step)


# 3.3 Evaluate LoRA (Includes Merge Step)

# merge LoRA adapter with the base model
python -m model.merge_lora_adapter \
  --base-checkpoint "$FOUNDATIONAL_CKPT" \
  --adapter-path "$LORA_ADAPTER" \
  --output-path "$LORA_MERGED_OUTPUT"

# Evaluate the LoRA merged model
echo ">>> Evaluating LoRA-merged Model..."

accelerate launch run_evaluation.py \
  --model-path "$LORA_MERGED_OUTPUT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000


### 3.4 Evaluate GRPO


# 3.4 Evaluate GRPO
echo ">>> Evaluating GRPO Model..."

accelerate launch run_evaluation.py \
  --model-path "$GRPO_CKPT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

