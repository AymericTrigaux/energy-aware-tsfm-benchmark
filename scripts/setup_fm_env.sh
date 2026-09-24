#!/usr/bin/env bash
# One-shot setup of .venv_fm on the ESAT vierre64 server (CPU-only).
# Bootstraps Python 3.11 via `uv`, then installs foundation-model deps.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Keep large caches off the backed-up home (~/no_backup is excluded from backup).
export UV_INSTALL_DIR="$HOME/no_backup/.local/bin"
export UV_PYTHON_INSTALL_DIR="$HOME/no_backup/uv/python"
export UV_CACHE_DIR="$HOME/no_backup/uv/cache"
mkdir -p "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

# 1. Install `uv` if missing.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv to $UV_INSTALL_DIR …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# 2. Get Python 3.11 (downloaded into ~/no_backup, not home).
uv python install 3.11

# 3. Create .venv_fm with that interpreter.
if [ ! -d "$PROJECT_DIR/.venv_fm" ]; then
    echo "[setup] creating .venv_fm …"
    uv venv --python 3.11 "$PROJECT_DIR/.venv_fm"
fi
# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv_fm/bin/activate"

# 4. Install dependencies (CPU-only torch wheels).
echo "[setup] installing PyTorch (CPU) …"
uv pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.4.*"

echo "[setup] installing FM stack …"
# Pin setuptools<80 — newer setuptools dropped pkg_resources, which lightning
# / uni2ts / lag-llama all still depend on at import time.
uv pip install \
    "setuptools<80" \
    numpy "pandas<2.2" matplotlib scikit-learn pyarrow tqdm \
    huggingface_hub gluonts \
    chronos-forecasting \
    "uni2ts>=1.2.0" \
    codecarbon carbontracker

# Lag-Llama: install from the bundled clone (not on PyPI).
if [ -d "$PROJECT_DIR/lag-llama" ]; then
    echo "[setup] installing lag-llama from local clone …"
    uv pip install -e "$PROJECT_DIR/lag-llama"
fi

# Optional: TimesFM-200M (skip if you only want the truly small models).
# uv pip install timesfm

# 5. HF cache on no_backup.
echo "export HF_HOME=\"\$HOME/no_backup/hf_cache\"" >> "$PROJECT_DIR/.venv_fm/bin/activate"

echo
echo "[ok] .venv_fm ready."
python -c "import torch, sys; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
