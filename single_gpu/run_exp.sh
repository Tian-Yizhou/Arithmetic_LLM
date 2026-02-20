
echo "PHASE 1: TRAINING PIPELINE"



# 3. Train foundational model
echo ">>> [Train Step 1/4] Starting Foundational Training..."
python run_foundational_training.py \
  --corpus-path data/foundational_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --max-seq-length 512 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --num-epochs 5

# foundational best model path
FOUNDATIONAL_DIR=$(ls -td models/foundational_* | head -n 1)
FOUNDATIONAL_CKPT="${FOUNDATIONAL_DIR}/best_model.pt"


# 4. Train instruction-tuned model
python run_instruction_training.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint "$FOUNDATIONAL_CKPT" \
  --num-epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-4


INSTRUCTION_DIR=$(ls -td models/instruction_* | head -n 1)
INSTRUCTION_CKPT="${INSTRUCTION_DIR}/best_model.pt"


# 5. Fine-tune with LoRA adapters (Optional)
echo ">>> [Train Step 3/4] Starting LoRA Training..."
python run_instruction_training_lora.py \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint "$FOUNDATIONAL_CKPT" \
  --num-epochs 5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules attention,feedforward \
  --save-merged-model


LORA_DIR=$(ls -td models/instruction_lora_* | head -n 1)
LORA_ADAPTER="${LORA_DIR}/lora_adapter.pt"
LORA_MERGED_OUTPUT="${LORA_DIR}/merged_model.pt"


# 6. GRPO training (Optional)
echo ">>> [Train Step 4/4] Starting GRPO Training..."
python run_grpo_training.py \
  --instruction-corpus data/instruction_corpus.txt \
  --tokenizer data/tokenizer \
  --sft-checkpoint "$INSTRUCTION_CKPT" \
  --output-dir models/grpo \
  --num-epochs 5 \
  --batch-size 8 \
  --num-candidates 4 \
  --temperature 0.8 \
  --kl-penalty-coef 0.05


GRPO_DIR=$(ls -td models/grpo/grpo_* | head -n 1)
GRPO_CKPT="${GRPO_DIR}/final_model.pt"


echo "PHASE 2: EVALUATION PIPELINE"


# 3.1 Evaluate Foundational
echo ">>> [Eval Step 1/4] Evaluating Foundational Model..."
python run_evaluation.py \
  --model-path "$FOUNDATIONAL_CKPT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1


# 4.1 Evaluate Instruction
echo ">>> [Eval Step 2/4] Evaluating Instruction Model..."
python run_evaluation.py \
  --model-path "$INSTRUCTION_CKPT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000


# 5.1 Evaluate LoRA (Includes Merge Step)

# merge LoRA adapter with the base model to get a standalone model (optional)
python merge_lora_adapter.py \
  --base-checkpoint "$FOUNDATIONAL_CKPT" \
  --adapter-path "$LORA_ADAPTER" \
  --output-path "$LORA_MERGED_OUTPUT"

# 5.1 Evaluate the LoRA merged model (optional)
python run_evaluation.py \
  --model-path "$LORA_MERGED_OUTPUT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000


# 6.1 Evaluate GRPO
echo ">>> [Eval Step 4/4] Evaluating GRPO Model..."
python run_evaluation.py \
  --model-path "$GRPO_CKPT" \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000