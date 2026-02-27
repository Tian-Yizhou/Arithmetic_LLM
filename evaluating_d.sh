#!/bin/bash

# activate conda
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate ds

FOUNDATIONAL_DIR=$(ls -td models/foundational_* | head -n 1)
FOUNDATIONAL_CKPT="${FOUNDATIONAL_DIR}/best_model.pt"

INSTRUCTION_DIR=$(ls -td models/instruction_* | head -n 1)
INSTRUCTION_CKPT="${INSTRUCTION_DIR}/best_model.pt"

LORA_DIR=$(ls -td models/instruction_lora_* | head -n 1)
LORA_ADAPTER="${LORA_DIR}/lora_adapter.pt"
LORA_MERGED_OUTPUT="${LORA_DIR}/merged_model.pt"

GRPO_DIR=$(ls -td models/grpo_* | head -n 1)
GRPO_CKPT="${GRPO_DIR}/checkpoint_step_8500.pt"

#3.1 Evaluate the foundational model, performance would be bad
accelerate launch run_evaluation.py \
  --model-path ${FOUNDATIONAL_CKPT} \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1

# 4.1 Evaluate the model
accelerate launch run_evaluation.py \
  --model-path ${INSTRUCTION_CKPT} \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000


# 5.1 Evaluate the LoRA merged model (optional)
accelerate launch run_evaluation.py \
  --model-path ${LORA_MERGED_OUTPUT} \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

#6.1 eval GRPO model
accelerate launch run_evaluation.py \
  --model-path ${GRPO_CKPT} \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000
