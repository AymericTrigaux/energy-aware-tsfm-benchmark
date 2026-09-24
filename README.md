# Energy-aware benchmark of time series foundation models

Walk-forward forecasting benchmark on Belgian electricity load (Elia, 15 minute
resolution) that scores four classical models and eleven foundation model
variants on accuracy and on measured energy. Every run records CPU package
energy from RAPL, GPU power from nvidia-smi and whole-chassis power from the
server's BMC, alongside the two software trackers CodeCarbon and CarbonTracker.

The measurements behind the paper were taken on a Dell Inc. PowerEdge R7725
server with two AMD EPYC 9455 processors and one NVIDIA RTX PRO 6000 Blackwell
Server Edition GPU, running Linux with world-readable RAPL counters.

## Paper

```bibtex
@misc{trigaux2026fourorders,
  title  = {Four Orders of Magnitude: Energy-Aware Benchmarking of Time Series Foundation Models},
  author = {Trigaux, Aymeric and Qaiser and Kazmi},
  year   = {2026},
  note   = {Workshop paper, preprint forthcoming}
}
```

Code developed with the assistance of Claude Code; benchmark design,
measurements and analysis are the authors' own.

## Installation

The environment that produced every published number is pinned in
`requirements.lock` (Python 3.12.14). Recreate it with `uv`:

```bash
uv venv --python 3.12 .venv_fm
source .venv_fm/bin/activate
uv pip install -r requirements.lock
```

Three notes on the lock file:

- PyTorch is pinned as `torch==2.11.0+cu128`. That wheel comes from the PyTorch
  index rather than PyPI, so add
  `--extra-index-url https://download.pytorch.org/whl/cu128` to the install
  command on a machine with a CUDA 12.8 driver, or drop the `+cu128` suffix for
  a CPU-only install.
- `uni2ts` (Moirai) is pinned to a git commit and, like `pytorch-lightning`
  and Lag-Llama, still imports `pkg_resources`; keep `setuptools<80`.
- Lag-Llama is not on PyPI. Clone
  `https://github.com/time-series-foundation-models/lag-llama` into
  `lag-llama/` at the project root and export
  `PYTHONPATH="$PWD/lag-llama:$PYTHONPATH"` before running that model.

`requirements.txt` is a short unpinned list for the classical models only.

## Data

Download the Elia total load export from the Elia open data portal and place
it at `data/raw/Data Elia Load.csv` (semicolon delimited). Then:

```bash
python prepare_data.py elia      # writes data/processed/elia_load_15min.csv
python prepare_data.py etth1     # optional: fetches ETTh1 for the cross-dataset check
```

`data/` is not tracked by git.

## Running

Three environment variables locate the interpreter, the model cache and the
results root. Every script under `scripts/` sets each one only if it is unset,
with the value of the measurement host as default, so on that host nothing
needs exporting. Elsewhere, export all three:

```bash
export BENCH_PY=/path/to/.venv_fm/bin/python
export HF_HOME=/path/to/hf_cache
export BENCH_RESULTS=/path/to/results
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
```

Classical models run on the CPU, foundation models on the GPU:

```bash
# naive, linear, lgbm, arima (sarima also exists but is slow)
python run_benchmark.py --models naive linear lgbm arima \
    --horizons 1 4 96 --stride 2 --test_years 1.0 \
    --tuned_params results/tuned_params.json \
    --energy_tool both --bmc --rapl --no_plots

# chronos_*, timesfm_*, moirai_*, moirai2_small, lag_llama
python run_tsfm_benchmark.py --models chronos_bolt_mini \
    --horizons 1 4 96 --stride 2 --test_years 1.0 \
    --batch_size 16 --device cuda --gpu_index 0 \
    --energy_tool all --bmc --rapl --no_plots
```

Horizons are in steps of 15 minutes (1, 4 and 96 steps = 15 minutes, 1 hour and
1 day). `--stride 2` places a forecast origin every 30 minutes across the test
year, 17,473 origins per horizon. Runs must be sequential: every meter here
sees the whole machine, so two concurrent runs would each absorb the other's
draw. `scripts/host_idle_check.sh` refuses to start a measurement while the
host is busy.

Each run writes one directory, `<results>/wf_<timestamp>/` for classical models
and `<results>/tsfm_<timestamp>/` for foundation models, containing
`metrics.csv` (one row per model and horizon), `predictions.parquet` and the raw
output of every meter that was on.

## What is measured

| Meter | Flag | Reads | Raw output in the run directory | Columns in metrics.csv |
|---|---|---|---|---|
| CodeCarbon 3.2.6 | `--energy_tool codecarbon`, `both` or `all` | CPU from the RAPL package counters, GPU from nvidia-smi, RAM from a size model | `codecarbon/emissions.csv` | `fit_cc_*`, `eval_cc_*`, `cc_cpu_mode` |
| CarbonTracker 2.4.4 | `--energy_tool carbontracker`, `both` or `all` | CPU from the RAPL package counters, GPU through NVML | `carbontracker/<tag>/*.log` | `fit_ct_*`, `eval_ct_*`, `ct_cpu_avg_w` |
| nvidia-smi poller | `--energy_tool nvidia_smi` or `all` (foundation models only) | `power.draw.instant` every 100 ms, trapezoid integral | `nvidia_smi/<tag>_gpu_power.csv` | `fit_nv_*`, `eval_nv_*` |
| BMC chassis meter | `--bmc` | the ACPI `power_meter` hwmon sensor (whole chassis, 2 s hardware average) every 2 s | `bmc_power.csv` | `bmc_gross_kwh`, `bmc_incremental_kwh`, `bmc_baseline_w`, `bmc_baseline_std_w`, `bmc_mean_w`, `bmc_peak_w`, `bmc_run_window_s` |
| RAPL package meter | `--rapl` | `energy_uj` of `package-0` and `package-1` every 2 s, wrap-safe | `rapl_power.csv` | `rapl_gross_kwh`, `rapl_incremental_kwh`, `rapl_baseline_w`, `rapl_baseline_std_w`, `rapl_mean_w`, `rapl_peak_w`, `rapl_run_window_s` |

`<tag>` is `<run_id>_<model>_<phase>` with phase `fit` or `eval`. The two
software trackers wrap the fit and the eval phase separately; the BMC and RAPL
meters wrap the eval phase only, with a 60 s idle baseline sampled before and
after it. `gross` is the integral over the run window, `incremental` the
integral of power above the baseline mean, clamped at zero. `cc_cpu_mode`
records whether CodeCarbon read RAPL or fell back to its load model, so a row's
CPU energy can be read as a measurement or as an estimate.

The energy columns are in kWh, the emission columns in kg CO2 equivalent using
the Belgian grid intensity, and every row also carries the 1 minute load
average and GPU utilisation at the start and end of the run.

## Repeat campaign

Run-to-run variance is measured by repeating one model several times in
sequence, each run with all five meters on:

```bash
scripts/host_idle_check.sh                       # gate: load average below 8, GPU idle
scripts/run_repeats.sh --models lgbm chronos_bolt_mini --repeats 10
python scripts/summarise_repeats.py --root results/repeats_<timestamp>
```

The summariser reports mean, standard deviation, minimum and maximum across
repeats for accuracy, for each meter's energy and power figures, and for the
split of CodeCarbon's total into the measured and the modelled part.

## Reproducing the paper

`docs/reproduce.md` gives the exact commands, tags and known gaps behind every
published figure. `docs/checkpoint_revisions.md` lists the Hugging Face commit
each foundation model is pinned to, with parameter counts.

## Repository layout

```
run_benchmark.py        classical models, walk-forward
run_tsfm_benchmark.py   foundation models, walk-forward
prepare_data.py         Elia and ETTh1 preparation
power_curves.py         per-second power profiles and composite figure
plot_thesis.py          paper figures
src/models.py           naive, ARIMA and SARIMA, Ridge, LightGBM
src/models_tsfm.py      Chronos, TimesFM, Moirai, Lag-Llama wrappers
src/metrics.py          error metrics and the five energy meters
scripts/                sequential campaign runners and the summariser
config/, docs/          example config, reproduction notes
results/tuned_params.json   tuned Ridge and LightGBM hyperparameters
```

## Licence

See the `LICENSE` file.
