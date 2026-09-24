# Reproducing the thesis results

Exact commands for the four pipelines behind the thesis. Read the
[Known gaps](#known-gaps) section first — two of them cannot currently be
reproduced from what is in this repository, and one figure input no longer exists.

## Environment

All benchmark scripts invoke this interpreter directly:

```
/volume1/no_backup/r1062653/thesis_fm_env/bin/python     # Python 3.12.14
```

The `.venv` (Python 3.14) described in `CLAUDE.md` **does not exist on this host**;
`thesis_fm_env` is the environment that produced every result in `results/`.
Its exact package set is pinned in [`requirements.lock`](../requirements.lock).

```bash
# Recreate it
uv venv --python 3.12 .venv_fm
uv pip install -r requirements.lock
```

`requirements.txt` is an unpinned convenience list and is not sufficient for
reproduction — it omits `lightgbm`, `optuna`, `holidays`, `torch`,
`chronos-forecasting`, `timesfm`, `uni2ts`, `gluonts` and `huggingface_hub`.

Every run needs these environment variables (set by all `scripts/*.sh`):

```bash
export HF_HOME=/volume1/no_backup/r1062653/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PWD/lag-llama:${PYTHONPATH:-}"     # lag_llama only
```

Foundation-model weights are pinned to explicit commits; see
[`checkpoint_revisions.md`](checkpoint_revisions.md).

---

## 1. Final stride-2 run (the headline results)

Stride 2 = a forecast origin every 30 minutes across a one-year test window,
17,473 origins per horizon. Runs are **strictly sequential**: CodeCarbon attributes
system-wide energy to the running process, so concurrent runs corrupt attribution.

Two scripts, 15 models total:

```bash
# 11 models: 4 classical + 6 small/mini FMs + Moirai-2.0   -> results/stride2_remaining_<TS>/
scripts/run_stride2_remaining.sh

# 4 larger FM variants                                      -> results/fm_large_<TS>/
scripts/run_fm_large.sh
```

The underlying invocations, if you need one model in isolation:

```bash
# classical (naive, linear, lgbm, arima) — CPU, CodeCarbon + CarbonTracker
python run_benchmark.py --models lgbm --horizons 1 4 96 --stride 2 \
    --tuned_params results/tuned_params.json \
    --energy_tool both --results_dir results/<dir> --no_plots

# foundation models — GPU, CodeCarbon + CarbonTracker + nvidia-smi
python run_tsfm_benchmark.py --models chronos_bolt_mini \
    --horizons 1 4 96 --test_years 1.0 --stride 2 \
    --batch_size 16 --device cuda --gpu_index 0 \
    --energy_tool all --results_dir results/<dir> --no_plots
```

Only `linear` and `lgbm` take `--tuned_params`. SARIMA is excluded: it was run at a
different stride and is not energy-comparable with the rest of the cohort.

**Memory:** SARIMA/SARIMAX walk-forward is single-worker only. Never run a pool of
SARIMAX fits — each results object holds gigabytes of Kalman state at s=96.

## 2. Profiling runs (stride 96, power curves)

Per-second power curves used for the methodology-validation figures. `stride` stays
at its default of 96 here; the variable is `--n_origins` per model.

```bash
# All 8 models at 10 Hz + rebuild of the composite figure + sanity checks
./run_hires.sh

# One model
python power_curves.py measure --model chronos_bolt_mini --n_origins 1200 \
    --nvsmi_interval_ms 100 --cc_poll_interval_s 1.0 --tag hires
```

`run_hires.sh` uses these `model:n_origins` pairs: `naive:1457`, `linear:1457`,
`lgbm:1457`, `arima:800`, `chronos_bolt_mini:1200`, `timesfm_200m:700`,
`moirai_small:500`, `lag_llama:60`.

## 3. ETTh1 cross-dataset check

**This pipeline cannot currently be run — see [Known gaps](#known-gaps).** Recorded
here for the record; the parameters are recovered from
`logs/etth1_focused_classical_20260514_193940.log` and
`logs/etth1_focused_fm_20260514_194014.log`.

```bash
python prepare_etth1.py          # MISSING from the repository

python run_benchmark.py --data data/processed/etth1.csv \
    --timestamp_col datetime --target_col ot --freq h \
    --horizons 1 24 --stride 2 --test_years 0.4 --naive_s 24 \
    --models naive lgbm --energy_tool both --no_plots

python run_tsfm_benchmark.py --data data/processed/etth1.csv \
    --timestamp_col datetime --target_col ot --freq h \
    --horizons 1 24 --stride 2 --test_years 0.4 \
    --models chronos_bolt_mini --batch_size 16 --device cuda \
    --energy_tool all --no_plots
```

The logged run loaded 17,420 hourly observations spanning 2016-07-01 to 2018-06-26,
split 13,915 train / 3,505 test, giving 1,741 origins.

## 4. Figures

```bash
# Composite power curves -> figures/power_curves_composite.{png,pdf}
python power_curves.py composite --tag hires \
    --out figures/power_curves_composite.png

# Supporting power-curve analysis (cross-validation table, per-origin curves)
python power_curves.py analyze --tag hires

# Chapter 6 figures, incl. pretraining_amortization -> figures/chapter6_unified/
python plot_thesis.py unified --metrics results/combined_all/metrics.csv
```

Both paper figures are authored at their printed size so that font sizes on the
canvas are the sizes on the page: the composite at 5.17 x 5.9 in (0.94 NeurIPS
textwidth), the amortization plot at 4.68 x 4.5 in (0.85 textwidth). Both write a
PNG at 300 dpi plus a sibling PDF. **Do not re-add `bbox_inches="tight"` to either**
— it re-crops to the artists' extent, changing the output width and silently
rescaling every point size.

Other `plot_thesis.py` subcommands (`bars`, `carbon`, `pareto`, `pareto_h96`,
`backends`, `params`, `variants`) default to metrics paths under
`results/combined/` and `results/combined_all/` that no longer exist; pass
`--metrics` explicitly.

---

## Known gaps

Four things in this repository cannot be reproduced as-is. They are pre-existing;
none is caused by the scripts documented above.

**1. `results/combined_all/metrics.csv` no longer exists.** It is the input for
`plot_thesis.py unified`. It can be rebuilt by concatenating the 15 per-model
`metrics.csv` files from the two stride-2 runs:

```bash
python - <<'PY'
import pandas as pd, glob
srcs = (sorted(glob.glob("results/stride2_remaining_20260512_204727/*/*/metrics.csv"))
        + sorted(glob.glob("results/fm_large_20260512_182035/*/*/metrics.csv")))
pd.concat([pd.read_csv(f) for f in srcs], ignore_index=True) \
  .to_csv("results/combined_all/metrics.csv", index=False)
PY
```

The rebuild reproduces **all 15 models' MAE values exactly** and all 11
foundation-model energies to within the 5% tolerance of the built-in check in
`cmd_unified`. It does **not** reproduce the four classical models' energies:
`naive` is off by 48.9%, `linear` 20.3%, `lgbm` 8.6%, `arima` 5.8%. No run
surviving on disk matches those reference energies — `energy_pass_20260512_102839`
is 85-92% off and has the wrong origin count (1,457, not 17,473). The classical
energy rows in the original `combined_all` therefore came from a run that is gone.
Figures depending on classical energy — including the LightGBM floor in
`pretraining_amortization` — will differ slightly from the published versions.

**2. The ETTh1 pipeline is broken.** `prepare_etth1.py` is absent (only
`prepare_data.py` exists), `data/processed/etth1.csv` is absent, and the ETTh1
metrics that `cmd_unified` reads for its robustness figure
(`results/classical/20260514_173941_lgbm_naive_etth1/`,
`results/foundation_models/20260514_174014_chronos_bolt_mini_etth1/`) are absent.
`cmd_unified` catches this and skips the figure. Note `CLAUDE.md` describes ETTh1 as
14,400 hours ending 2018-02-20, but the logged run used 17,420 hours ending
2018-06-26.

**3. `tune_hyperparams.py` cannot run in the pinned environment.** `optuna` is not
installed in `thesis_fm_env` and so is absent from `requirements.lock`. Separately,
the provenance of `results/tuned_params.json` is lost: the only surviving Optuna
logs (`logs/tune_lgbm.log`, `logs/tune_linear.log`, both 25 trials) report best
parameters that **differ from those in the JSON** — alpha 31.34 vs 34.06, LightGBM
`n_estimators` 582 vs 488. The JSON matches what the stride-2 run consumed, so the
values are right; the run that produced them is unrecorded. Both tuning logs also
show `use_holidays: True`, but the benchmark never passes `--use_holidays`, so the
parameters were selected under a feature set the benchmark did not use.

**4. `requirements.lock` was produced with `pip freeze`, not `uv`.** `uv` is not
installed on this host and installing it would have modified the environment. The
format is identical and `uv pip install -r requirements.lock` consumes it directly.
`uv lock` is not usable without a `pyproject.toml`.

## Repository naming

`CLAUDE.md` refers to three files that do not exist under those names:
`src/tsfm_models.py` is `src/models_tsfm.py`; `prepare_dataset.py` and
`prepare_etth1.py` are absent (only `prepare_data.py`); and
`plot_power_curves_composite.py`, `measure_power_curve.py` and
`analyze_power_curves.py` referenced in `THESIS_REPORT.md` were folded into
`power_curves.py` subcommands.
