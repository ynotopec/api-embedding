#!/usr/bin/env bash
# Usage:
#   source run.sh [IP] [PORT]
#
# Compatible shell + systemd:
#   bash -lc 'source /path/bge-m3-api/run.sh 0.0.0.0 8000'

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$PROJECT_NAME}"

HOST="${1:-${HOST:-0.0.0.0}}"
PORT="${2:-${PORT:-8000}}"

cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

BACKEND="${BACKEND:-vllm}"
MODEL_ID="${MODEL_ID:-BAAI/bge-m3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-BAAI/bge-m3}"
API_KEY="${API_KEY:-}"

VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_TASK="${VLLM_TASK:-embed}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Backend: $BACKEND"
echo "==> Model:   $MODEL_ID"
echo "==> Listen:  $HOST:$PORT"

if [[ "$BACKEND" == "vllm" ]]; then
  args=(
    vllm serve "$MODEL_ID"
    --task "$VLLM_TASK"
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --dtype "$VLLM_DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  )

  if [[ -n "$API_KEY" ]]; then
    args+=(--api-key "$API_KEY")
  fi

  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    args+=(--trust-remote-code)
  fi

  exec "${args[@]}"
fi

if [[ "$BACKEND" == "flagembedding" ]]; then
  export HOST PORT MODEL_ID SERVED_MODEL_NAME API_KEY MAX_MODEL_LEN
  exec python "$PROJECT_DIR/app.py"
fi

echo "ERROR: BACKEND must be 'vllm' or 'flagembedding'"
exit 1
