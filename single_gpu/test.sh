###
 # @Author: Hannah
 # @Date: 2026-02-24 11:58:51
 # @LastEditTime: 2026-02-24 12:21:21
### 

# activate conda environment
# source conda. the path is D:\miniconda3
source /d/miniconda3/etc/profile.d/conda.sh
conda activate ds

# foundational best model path
FOUNDATIONAL_DIR=$(ls -td models/foundational_* | head -n 1)
FOUNDATIONAL_CKPT="${FOUNDATIONAL_DIR}/best_model.pt"

INSTRUCTION_DIR=$(ls -td models/instruction_* | head -n 1)
INSTRUCTION_CKPT="${INSTRUCTION_DIR}/best_model.pt"

LORA_DIR=$(ls -td models/instruction_lora_* | head -n 1)
LORA_ADAPTER="${LORA_DIR}/lora_adapter.pt"
LORA_MERGED_OUTPUT="${LORA_DIR}/merged_model.pt"

# 6. GRPO training (Optional)
echo ">>> [Train Step 4/4] Starting GRPO Training..."
accelerate launch run_grpo_training_d.py \
  --tokenizer data/tokenizer \
  --instruction-corpus data/instruction_corpus.txt \
  --sft-checkpoint "$INSTRUCTION_CKPT" \
  --output-dir models/grpo \
  --num-epochs 5 \
  --num-candidates 4 \
  --temperature 0.8 \
  --batch-size 8 \
  --kl-penalty-coef 0.05

GRPO_DIR=$(ls -td models/grpo/grpo_* | head -n 1)
GRPO_CKPT="${GRPO_DIR}/final_model.pt"