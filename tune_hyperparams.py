"""Hyperparameter tuning with Optuna for Ridge and LightGBM forecasters.

Runs BEFORE the main benchmark and saves the best params to a JSON file.
The benchmark loads those params via --tuned_params.

Validation strategy: last 20% of the training set as a held-out validation
set (chronological split, no shuffling). Same approach as in the KUL
time-series forecasting course Notebook 6.

Usage:
    python tune_hyperparams.py --model linear --n_trials 50
    python tune_hyperparams.py --model lgbm --n_trials 100
    python tune_hyperparams.py --model linear --n_trials 50 --use_holidays
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data import load_timeseries, clean_timeseries
from src.metrics import mae
from src.models import build_feature_matrix, DEFAULT_LAGS

warnings.filterwarnings("ignore")


# --- CLI ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=["linear", "lgbm"])
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--data", default="data/processed/elia_load_15min.csv")
    p.add_argument("--timestamp_col", default="datetime")
    p.add_argument("--target_col", default="totalload")
    p.add_argument("--freq", default="15min")
    p.add_argument("--test_years", type=float, default=1.0,
                   help="Same test window as the benchmark (kept out during tuning).")
    p.add_argument("--val_frac", type=float, default=0.20,
                   help="Fraction of the training set used as validation.")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 96])
    p.add_argument("--use_holidays", action="store_true")
    p.add_argument("--output", default="results/tuned_params.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --- Data helpers ---

def _prepare_data(args: argparse.Namespace):
    """Load, clean, and split data. Returns (y_train_inner, y_val)."""
    df = load_timeseries(
        args.data,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        freq=args.freq,
    )
    df = clean_timeseries(df, impute_method="interpolate")
    y_all = df["y"]

    # Remove the test window (same boundary as the benchmark)
    last_ts = y_all.index[-1]
    whole_years = int(args.test_years)
    extra_days = int(round((args.test_years - whole_years) * 365.25))
    test_start = last_ts - pd.DateOffset(years=whole_years, days=extra_days)
    y_train_full = y_all[y_all.index < test_start]

    # Chronological 80/20 split within training data
    n = len(y_train_full)
    n_inner = int(n * (1 - args.val_frac))
    y_inner = y_train_full.iloc[:n_inner]
    y_val = y_train_full.iloc[n_inner:]

    print(f"  Inner train: {len(y_inner):,} obs  ({y_inner.index[0]} → {y_inner.index[-1]})")
    print(f"  Validation : {len(y_val):,} obs  ({y_val.index[0]} → {y_val.index[-1]})")
    return y_inner, y_val


def _eval_mae(estimator, y_inner: pd.Series, y_val: pd.Series,
              horizons: List[int], use_holidays: bool) -> float:
    """Fit on y_inner, return mean MAE across horizons on y_val."""
    from sklearn.base import clone

    y_combined = pd.concat([y_inner, y_val])
    train_end = len(y_inner)
    maes = []

    for h in horizons:
        X_all, y_tgt = build_feature_matrix(
            y_combined, DEFAULT_LAGS, horizon=h,
            calendar=True, use_holidays=use_holidays,
        )
        origin_positions = np.array([
            y_combined.index.get_loc(idx) for idx in X_all.index
        ])
        target_positions = origin_positions + h

        train_mask = origin_positions < train_end
        val_mask = target_positions >= train_end

        X_train = X_all.values[train_mask]
        y_train_h = y_tgt.values[train_mask]
        X_val = X_all.values[val_mask & ~train_mask]
        y_val_h = y_tgt.values[val_mask & ~train_mask]

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        m = clone(estimator)
        m.fit(X_train, y_train_h)
        y_pred = m.predict(X_val)
        maes.append(mae(y_val_h, y_pred))

    return float(np.mean(maes)) if maes else float("inf")


# --- Optuna objectives ---

def _objective_linear(trial, y_inner, y_val, horizons, use_holidays):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    est = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return _eval_mae(est, y_inner, y_val, horizons, use_holidays)


def _objective_lgbm(trial, y_inner, y_val, horizons, use_holidays):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("pip install lightgbm") from exc

    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "verbose":           -1,
    }
    est = LGBMRegressor(**params)
    return _eval_mae(est, y_inner, y_val, horizons, use_holidays)


# --- Main ---

def main():
    args = parse_args()

    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nTuning {args.model.upper()} with Optuna ({args.n_trials} trials)")
    print(f"  Horizons   : {args.horizons}")
    print(f"  use_holidays: {args.use_holidays}")

    y_inner, y_val = _prepare_data(args)

    if args.model == "linear":
        objective = lambda trial: _objective_linear(
            trial, y_inner, y_val, args.horizons, args.use_holidays
        )
    else:
        objective = lambda trial: _objective_lgbm(
            trial, y_inner, y_val, args.horizons, args.use_holidays
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _progress(study, trial):
        if trial.number % 10 == 0 or trial.number == args.n_trials - 1:
            print(f"  Trial {trial.number + 1:>4}/{args.n_trials}  "
                  f"value={trial.value:.2f}  "
                  f"best={study.best_value:.2f}  "
                  f"params={study.best_params}")

    study.optimize(objective, n_trials=args.n_trials, callbacks=[_progress])

    best = study.best_params
    print(f"\nBest params for {args.model}: {best}")
    print(f"Best mean MAE: {study.best_value:.2f} MW")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    existing: Dict = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing[args.model] = best
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
