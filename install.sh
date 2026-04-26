#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$PROJECT_NAME}"

cd "$PROJECT_DIR"

echo "==> Project: $PROJECT_NAME"
echo "==> Venv:    $VENV_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv user-wide"
  python3 -m pip install --user -U uv
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Creating/upgrading venv"
uv venv "$VENV_DIR" --python python3

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading packaging tools"
uv pip install -U pip setuptools wheel

# Load .env only for install choices
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

BACKEND="${BACKEND:-vllm}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
VLLM_VERSION="${VLLM_VERSION:-}"

echo "==> Installing common fallback server deps"
uv pip install -U -r requirements-flagembedding.txt

if [[ "$INSTALL_VLLM" == "1" || "$BACKEND" == "vllm" ]]; then
  echo "==> Installing/upgrading vLLM"
  if [[ -n "$VLLM_VERSION" ]]; then
    uv pip install -U "vllm==$VLLM_VERSION"
  else
    uv pip install -U vllm
  fi
fi

echo
echo "✅ Install OK"
echo
echo "Next:"
echo "  cp .env.example .env"
echo "  source run.sh 0.0.0.0 8000"
