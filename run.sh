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

MODEL_ID="${MODEL_ID:-}"
MODEL_ALIAS="${MODEL_ALIAS:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ALIAS}"

if [[ -z "$MODEL_ID" ]]; then
  echo "ERROR: MODEL_ID is required."
  echo "Set it in $PROJECT_DIR/.env or export it before starting the server."
  exit 1
fi

API_KEY="${API_KEY:-}"

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
)

if [[ -n "$SERVED_MODEL_NAME" ]]; then
  ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "$API_KEY" ]]; then
  ARGS+=(--api-key "$API_KEY")
fi

OPTIONAL_VALUE_ARGS=(
  "RUNNER:--runner"
  "DTYPE:--dtype"
  "MAX_MODEL_LEN:--max-model-len"
  "GPU_MEMORY_UTILIZATION:--gpu-memory-utilization"
  "MAX_NUM_SEQS:--max-num-seqs"
  "MAX_NUM_BATCHED_TOKENS:--max-num-batched-tokens"
  "QUANTIZATION:--quantization"
  "KV_CACHE_DTYPE:--kv-cache-dtype"
)

for mapping in "${OPTIONAL_VALUE_ARGS[@]}"; do
  variable="${mapping%%:*}"
  flag="${mapping#*:}"
  value="${!variable:-}"
  if [[ -n "$value" ]]; then
    ARGS+=("$flag" "$value")
  fi
done

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi
if [[ "${DISABLE_LOG_REQUESTS:-0}" == "1" ]]; then
  ARGS+=(--no-enable-log-requests)
fi

echo "==> Starting vLLM embeddings"
echo "    project: $PROJECT_NAME"
echo "    venv:    $VENV_DIR"
echo "    model:   $MODEL_ID"
echo "    name:    ${SERVED_MODEL_NAME:-$MODEL_ID}"
echo "    url:     http://$HOST:$PORT"
echo "    api:     POST /v1/embeddings"
echo

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
