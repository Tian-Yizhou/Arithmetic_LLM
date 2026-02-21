#!/bin/bash
# ============================================================
# Arithmetic LLM — Demo Script
# ============================================================
# This script loads a trained model checkpoint and evaluates it
# on 10 randomly generated arithmetic expressions.
# Results are saved to demo_results.txt.
#
# Usage:
#   bash demo.sh
#
# Before running, update the paths below to match your setup.
# ============================================================

# ---------- Configuration (edit these paths) ----------

# Path to the instruction-tuned model checkpoint
MODEL_PATH="models/<your_checkpoint_dir>/best_model.pt"

# Path to the trained tokenizer directory
TOKENIZER_PATH="data/tokenizer"

# Output file for evaluation results
OUTPUT_FILE="demo_results.txt"

# Number of demo examples
NUM_EXAMPLES=10

# Maximum expression depth (controls complexity)
MAX_DEPTH=3

# Random seed for reproducible examples
SEED=42

# Device: auto, cuda, mps, or cpu
DEVICE="auto"

# ---------- Run demo ----------

echo "============================================================"
echo " Arithmetic LLM — Demo"
echo "============================================================"
echo ""
echo "Model      : $MODEL_PATH"
echo "Tokenizer  : $TOKENIZER_PATH"
echo "Output     : $OUTPUT_FILE"
echo "Examples   : $NUM_EXAMPLES"
echo ""

python demo.py \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --output "$OUTPUT_FILE" \
  --num-examples "$NUM_EXAMPLES" \
  --max-depth "$MAX_DEPTH" \
  --seed "$SEED" \
  --device "$DEVICE"

echo ""
echo "Done. Results written to: $OUTPUT_FILE"
