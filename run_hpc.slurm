#!/bin/bash
#SBATCH --job-name=tsfm_bench
#SBATCH --output=logs/tsfm_%j.out
#SBATCH --error=logs/tsfm_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # adjust to your cluster's GPU partition name

# ── Project root (adjust if you submit from a different directory) ────────────
PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start      : $(date)"
echo "Project dir: $PROJECT_DIR"
echo "=========================================="

# ── Python environment ────────────────────────────────────────────────────────
# Option A: use the .venv_fm you transfer alongside the project
VENV="$PROJECT_DIR/.venv_fm"

if [ ! -d "$VENV" ]; then
    echo "[setup] creating .venv_fm with system Python 3.11 …"
    # Load a Python 3.11 module if your HPC uses environment modules:
    # module load Python/3.11.5-GCCcore-13.2.0
    python3.11 -m venv "$VENV"
    source "$VENV/bin/activate"

    echo "[setup] installing Python dependencies …"
    pip install --upgrade pip --quiet
    pip install numpy pandas matplotlib scikit-learn \
                codecarbon carbontracker pyarrow tqdm \
                torch torchvision \
                chronos-forecasting \
                timesfm \
                huggingface_hub \
                gluonts \
                --quiet

    # lag-llama is not on PyPI — install from the bundled clone
    if [ -d "$PROJECT_DIR/lag-llama" ]; then
        echo "[setup] installing lag-llama from local clone …"
        pip install -e "$PROJECT_DIR/lag-llama" --quiet
    else
        echo "[warn] lag-llama directory not found — lag_llama model will fail"
    fi

    # uni2ts for Moirai
    pip install uni2ts --quiet
else
    source "$VENV/bin/activate"
fi

echo "[env] Python: $(python --version)"
echo "[env] Torch : $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"

# ── HuggingFace cache: put it on scratch/local storage, not home ──────────────
# Adjust $VSC_SCRATCH or $TMPDIR to whatever scratch path your HPC provides.
export HF_HOME="${VSC_SCRATCH:-$TMPDIR}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
echo "[env] HF cache: $HF_HOME"

# ── Logging directory ─────────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"

# ── Run benchmark ─────────────────────────────────────────────────────────────
# --energy_tool all  → CodeCarbon + CarbonTracker + nvidia-smi (recommended on GPU nodes)
# --device is auto-detected (cuda preferred), but you can pin it explicitly:
#   --device cuda

python run_tsfm_benchmark.py \
    --models chronos_mini timesfm_200m lag_llama moirai_small \
    --horizons 1 4 96 \
    --test_years 1.0 \
    --stride 96 \
    --context_length 512 \
    --num_samples 100 \
    --energy_tool all \
    --gpu_index 0 \
    --results_dir results \
    --no_plots

EXIT_CODE=$?
echo "=========================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
