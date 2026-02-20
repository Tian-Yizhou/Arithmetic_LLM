#!/bin/bash

set -e

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tyz

# 3. Train foundational model
accelerate launch run_foundational_training.py  \
  --corpus-path data/foundational_corpus.txt \
  --tokenizer-path data/tokenizer \
  --output-dir models/ \
  --num-epochs 10 \
  --learning-rate 2e-4 \
  --batch-size 48 \
  --gradient-accumulation-steps 1

# foundational best model path
FOUNDATIONAL_DIR=$(ls -td models/foundational_* | head -n 1)
FOUNDATIONAL_CKPT="${FOUNDATIONAL_DIR}/best_model.pt"

# 4. Train instruction-tuned model
accelerate launch run_instruction_training.py \
    --instruction-corpus-path data/instruction_corpus.txt \
    --tokenizer-path data/tokenizer \
    --foundational-checkpoint "$FOUNDATIONAL_CKPT" \
    --batch-size 48 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-5 \
    --num-epochs 10

INSTRUCTION_DIR=$(ls -td models/instruction_* | head -n 1)
INSTRUCTION_CKPT="${INSTRUCTION_DIR}/best_model.pt"

# 5 Fine-tune with LoRA adapters (optional)
accelerate launch run_instruction_training_lora.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/"$FOUNDATIONAL_CKPT"/best_model.pt \
  --num-epochs 10 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules attention \
  --save-merged-model \
  --batch-size 48 \
  --gradient-accumulation-steps 1


# 6 GRPO training (optional)
accelerate launch run_grpo_training.py \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/"$INSTRUCTION_CKPT"/best_model.pt \
  --output-dir models/grpo \
  --data-mode generated \
  --log-every 1 \
  --num-samples 1024 \
  --num-epochs 3 \
  --num-candidates 8 \
  --max-gen-length 511 \
  --temperature 0.8 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --kl-penalty-coef 0.05

