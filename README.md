# Thesis – Energy-aware benchmarking (ARIMA first)

This folder contains a *reproducible* baseline pipeline to:
1) load & preprocess Belgian electricity load data,
2) train + evaluate ARIMA/SARIMA-style models,
3) measure training/inference energy and carbon with CodeCarbon,
4) log results (accuracy + energy) to CSV/JSON,
5) produce basic plots.

It is designed to match my thesis methodology: model-level energy (training + inference), then optional scaling to data-centre energy (PUE) and carbon (grid intensity). 

## Quick start

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Put your dataset in `data/`
Supported input formats:
- CSV with a timestamp column
- Parquet (preferred)

### 4) Run ARIMA baseline experiment
```bash
python scripts/run_arima_energy_benchmark.py \
  --input data/elia_load.csv \
  --timestamp_col datetime \
  --target_col totalload \
  --freq 15T \
  --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 \
  --horizons 4 96 \
  --order 2 1 2 \
  --seasonal_order 1 1 1 96 \
  --energy_tool codecarbon
```

Results go to:
- `results/metrics_runs.csv`
- `results/plots/`
- `results/codecarbon/` (raw CodeCarbon logs)

## Notes
- For hourly ARIMA, set `--freq 1H` and resample in preprocessing.
- Use `--auto_order` to run an AIC/BIC search over candidate (p,d,q) and (P,D,Q,s).

