#!/usr/bin/env bash
# Start the NeuralServe inference server locally (without Docker).
# Usage: bash scripts/start_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from .env …"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

: "${MODEL_PATH:?MODEL_PATH must be set (e.g. meta-llama/Meta-Llama-3.1-8B)}"
: "${REDIS_URL:=redis://localhost:6379}"
: "${PORT:=8000}"
: "${HOST:=0.0.0.0}"
: "${LOG_LEVEL:=info}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NeuralServe — Inference Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model     : $MODEL_PATH"
echo "  Adapter   : ${ADAPTER_PATH:-none}"
echo "  Redis     : $REDIS_URL"
echo "  Listen    : $HOST:$PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"
exec uvicorn serving.app:create_app \
    --factory \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level "$LOG_LEVEL"
