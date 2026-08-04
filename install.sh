#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$PROJECT_NAME}"

VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
CUDA_VERSION="${CUDA_VERSION:-130}"

echo "==> Project:      $PROJECT_DIR"
echo "==> Project name: $PROJECT_NAME"
echo "==> Venv:         $VENV_DIR"
echo "==> vLLM:         $VLLM_VERSION + cu$CUDA_VERSION"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_system_deps() {
  if ! need_cmd apt-get; then
    echo "==> apt-get not found; skipping system dependency installation."
    return
  fi

  local missing=()
  for cmd in curl jq git python3; do
    if ! need_cmd "$cmd"; then
      missing+=("$cmd")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    echo "==> System dependencies appear to be installed; skipping apt-get."
    return
  fi

  echo "==> Missing system commands: ${missing[*]}"
  echo "==> Set INSTALL_SYSTEM_DEPS=1 to install apt packages automatically."

  if [[ "${INSTALL_SYSTEM_DEPS:-0}" != "1" ]]; then
    echo "ERROR: system dependencies are missing, but automatic apt install is disabled."
    echo "Install them manually or rerun with INSTALL_SYSTEM_DEPS=1."
    exit 1
  fi

  local apt_cmd=(apt-get)
  if [[ "${EUID}" -ne 0 ]]; then
    if ! need_cmd sudo; then
      echo "ERROR: sudo is required for INSTALL_SYSTEM_DEPS=1 when not running as root."
      exit 1
    fi
    apt_cmd=(sudo apt-get)
  fi

  echo "==> Installing system dependencies with ${apt_cmd[*]}..."
  "${apt_cmd[@]}" update
  "${apt_cmd[@]}" install -y \
    curl \
    ca-certificates \
    jq \
    git \
    build-essential \
    python3 \
    python3-venv \
    python3-dev
}

install_system_deps

echo "==> Installing/upgrading uv..."
if ! need_cmd uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  uv self update || true
fi

if ! need_cmd uv; then
  echo "ERROR: uv not found after installation."
  echo "Add ~/.local/bin to PATH or reopen your shell."
  exit 1
fi

echo "==> Detecting architecture..."
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)
    WHEEL_ARCH="x86_64"
    ;;
  aarch64|arm64)
    WHEEL_ARCH="aarch64"
    ;;
  *)
    echo "ERROR: unsupported architecture: $ARCH"
    exit 1
    ;;
esac

VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${WHEEL_ARCH}.whl"

echo "==> Architecture: $ARCH -> $WHEEL_ARCH"
echo "==> Wheel:        $VLLM_WHEEL"

mkdir -p "$(dirname "$VENV_DIR")"

echo "==> Creating/upgrading venv..."
uv venv "$VENV_DIR" --python 3.11

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading base tooling..."
uv pip install -U pip setuptools wheel packaging

echo "==> Installing vLLM CUDA ${CUDA_VERSION}..."
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_PRERELEASE=allow
export UV_TORCH_BACKEND="cu${CUDA_VERSION}"

uv pip install -U \
  "$VLLM_WHEEL" \
  --torch-backend "cu${CUDA_VERSION}" \
  --index-strategy unsafe-best-match \
  --prerelease allow \
  --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}" \
  --extra-index-url "https://pypi.org/simple"

echo "==> Installing helper packages..."
uv pip install -U \
  "huggingface_hub[cli]>=0.34.0,<1.0" \
  hf_transfer \
  requests \
  openai \
  orjson

# `uv pip install` resolves each invocation independently. Keep the helper
# upgrade from replacing the huggingface-hub version required by transformers,
# then fail here with a useful dependency report rather than later at startup.
echo "==> Checking dependency compatibility..."
PIP_CHECK_OUTPUT="$(mktemp)"
if ! uv pip check >"$PIP_CHECK_OUTPUT" 2>&1; then
  cat "$PIP_CHECK_OUTPUT"
  if grep -q '^Found 1 incompatibility$' "$PIP_CHECK_OUTPUT" \
    && grep -q '^The package `nvidia-cusparselt-cu13` was built for a different platform$' "$PIP_CHECK_OUTPUT"; then
    echo "WARNING: ignoring uv pip check platform warning for nvidia-cusparselt-cu13."
    echo "This package is pulled by CUDA 13 wheels and is not required on every platform."
  else
    rm -f "$PIP_CHECK_OUTPUT"
    exit 1
  fi
fi
rm -f "$PIP_CHECK_OUTPUT"

echo "==> Checking dependency versions..."
python - <<'PY'
import importlib.metadata as md

packages = [
    "torch",
    "vllm",
    "transformers",
    "huggingface-hub",
    "openai",
    "orjson",
    "cuda-bindings",
    "cuda-python",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "flashinfer-python",
    "flashinfer-jit-cache",
]

for pkg in packages:
    try:
        print(f"{pkg}: {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}: not installed")
PY

echo "==> Checking Python imports..."
python - <<'PY'
import torch
import vllm
import transformers
import huggingface_hub
import openai
import orjson

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("huggingface-hub:", huggingface_hub.__version__)
print("openai:", openai.__version__)
print("orjson: ok")

if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
PY

echo "==> Checking vLLM OpenAI API server CLI..."
python -m vllm.entrypoints.openai.api_server --help >/dev/null

echo
echo "✅ Install OK"
echo
echo "Next:"
echo "  cp .env.example .env"
echo "  nano .env"
echo "  ./run.sh 0.0.0.0 8001"
