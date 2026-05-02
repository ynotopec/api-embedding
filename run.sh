#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$PROJECT_NAME}"

HOST="${1:-${HOST:-0.0.0.0}}"
PORT="${2:-${PORT:-8001}}"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

MODEL_ID="${MODEL_ID:-BAAI/bge-m3}"
MODEL_ALIAS="${MODEL_ALIAS:-BAAI/bge-m3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ALIAS}"

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$MODEL_ID"
fi

API_KEY="${API_KEY:-}"

DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

QUANTIZATION="${QUANTIZATION:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Seulement utile si ton install FlashInfer a un mismatch.
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-0}"

# Plus robuste avec CUDA + multiprocessing.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found: $VENV_DIR"
  echo "Run ./install.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

ARGS=(
  --model "$MODEL_ID"
  --host "$HOST"
  --port "$PORT"
  --runner pooling
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --trust-remote-code
)

if [[ -n "$API_KEY" ]]; then
  ARGS+=(--api-key "$API_KEY")
fi

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ "${DISABLE_LOG_REQUESTS:-1}" == "1" ]]; then
  ARGS+=(--no-enable-log-requests)
fi

if [[ -n "$QUANTIZATION" ]]; then
  ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ -n "$KV_CACHE_DTYPE" ]]; then
  ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

echo "==> Starting vLLM embeddings"
echo "    project: $PROJECT_NAME"
echo "    venv:    $VENV_DIR"
echo "    model:   $MODEL_ID"
echo "    name:    $SERVED_MODEL_NAME"
echo "    url:     http://$HOST:$PORT"
echo "    api:     POST /v1/embeddings"
echo "    api:     POST /pooling"
echo "    mem:     GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "    limits:  MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
