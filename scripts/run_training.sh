#!/usr/bin/env bash
# Run NeuralServe LoRA fine-tuning.
# Usage: bash scripts/run_training.sh [--lora_config PATH] [--training_config PATH]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LORA_CONFIG="${LORA_CONFIG:-$PROJECT_ROOT/configs/lora_config.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$PROJECT_ROOT/configs/training_config.yaml}"
RUN_NAME="${RUN_NAME:-lora-llama3-8b-$(date +%Y%m%d-%H%M%S)}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NeuralServe — LoRA Fine-tuning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LoRA config     : $LORA_CONFIG"
echo "  Training config : $TRAINING_CONFIG"
echo "  Run name        : $RUN_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Prepare dataset if not already done
if [ ! -f "$PROJECT_ROOT/data/alpaca_12k.jsonl" ]; then
    echo "Preparing dataset …"
    python "$PROJECT_ROOT/data/prepare_dataset.py" \
        --output "$PROJECT_ROOT/data/alpaca_12k.jsonl" \
        --size 12000
fi

cd "$PROJECT_ROOT"
python training/trainer.py \
    --lora_config "$LORA_CONFIG" \
    --training_config "$TRAINING_CONFIG" \
    --run_name "$RUN_NAME" \
    "$@"

echo "Training complete. Adapter saved to outputs/lora-llama3-8b/final_adapter/"
