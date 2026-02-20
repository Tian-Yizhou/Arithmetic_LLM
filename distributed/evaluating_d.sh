#!/bin/bash

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tyz

#3.1 Evaluate the foundational model, performance would be bad
accelerate launch run_evaluation_d.py \
  --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1

# 4.1 Evaluate the model
accelerate launch run_evaluation_d.py \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

# merge LoRA adapter with the base model to get a standalone model (optional)
python merge_lora_adapter.py \
  --base-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --adapter-path models/instruction_lora_YYYYMMDD_HHMMSS/lora_adapter.pt \
  --output-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt

# 5.1 Evaluate the LoRA merged model (optional)
accelerate launch run_evaluation_d.py \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

#6.1 eval GRPO model
accelerate launch run_evaluation_d.py \
  --model-path models/grpo/grpo_20260208_064824_549081/final_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000
