"""Walk-forward benchmark for Belgian electricity load forecasting.

Loads the Elia 15-min dataset, splits the last N years as test, then evaluates
each requested model using a rolling-origin protocol. At each origin t (every
`stride` steps), the model produces predictions for all horizons and these are
compared against the true values. Energy consumption is tracked via CodeCarbon
and/or CarbonTracker.

Usage:
    python run_benchmark.py --models naive linear lgbm arima sarima
    python run_benchmark.py --models naive --energy_tool none
    python run_benchmark.py --models linear --no_plots
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_timeseries, clean_timeseries, split_last_n_years, append_run_csv, freq_to_minutes
from src.metrics import (
    mae, mape, rmse, EnergyMeter, CarbonTrackerMeter, BmcPowerMeter, RaplPowerMeter,
    sample_host_conditions,
)
from src.models import (
    ArimaConfig,
    ARIMAPredictor,
    NaivePredictor,
    MLPredictor,
    make_linear_forecaster,
    make_lgbm_forecaster,
)


# --- CLI ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward benchmark for electricity load forecasting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data", default="data/processed/elia_load_15min.csv")
    p.add_argument("--timestamp_col", default="datetime")
    p.add_argument("--target_col", default="totalload")
    p.add_argument("--freq", default="15min")
    p.add_argument("--delimiter", default=None)

    # Evaluation protocol
    p.add_argument("--test_years", type=float, default=1.0,
                   help="Length of the test window in calendar years.")
    p.add_argument("--stride", type=int, default=96,
                   help="Steps between consecutive forecast origins (96 = 1 day).")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 96],
                   help="Forecast horizons in steps (1=15min, 4=1h, 96=1day).")

    # Models
    p.add_argument("--models", nargs="+", default=["naive", "linear"],
                   choices=["naive", "arima", "sarima", "linear", "lgbm"])

    # ARIMA / SARIMA
    p.add_argument("--arima_order", type=int, nargs=3, default=[2, 1, 2],
                   metavar=("p", "d", "q"))
    p.add_argument("--sarima_order", type=int, nargs=3, default=[2, 1, 2],
                   metavar=("p", "d", "q"))
    p.add_argument("--sarima_seasonal_order", type=int, nargs=4, default=[1, 1, 1, 96],
                   metavar=("P", "D", "Q", "s"))
    p.add_argument("--naive_s", type=int, default=96,
                   help="Seasonal period for the naive baseline (default 96 = 1 day at 15min). "
                        "Use 24 for hourly data with daily seasonality.")
    p.add_argument("--maxiter", type=int, default=50,
                   help="Max MLE iterations for SARIMAX fitting.")
    p.add_argument("--max_train_obs", type=int, default=4000,
                   help="Cap on training obs for ARIMA/SARIMA (most recent N rows).")

    # ML feature options
    p.add_argument("--use_holidays", action="store_true",
                   help="Add binary is_holiday feature (requires pip install holidays).")
    p.add_argument("--tuned_params", default=None,
                   help="Path to JSON from tune_hyperparams.py. Overrides defaults.")
    p.add_argument("--seasonal_residual", type=int, default=0, metavar="S",
                   help="Train ML models on residuals y[t+h]-y[t+h-S] (S=96 → daily). "
                        "Default 0 disables; recommended for strongly seasonal series.")

    # Energy tracking
    p.add_argument("--energy_tool", default="both",
                   choices=["codecarbon", "carbontracker", "both", "none"],
                   help="Energy tracking backend. 'both' runs both in parallel.")
    p.add_argument("--bmc", action="store_true",
                   help="Also measure chassis power via the ACPI/BMC power meter "
                        "(hwmon 'power_meter') during the eval phase. Off by default.")
    p.add_argument("--bmc_baseline_s", type=float, default=60.0,
                   help="Length of each BMC baseline window (before and after the "
                        "eval phase), in seconds. Only used with --bmc.")
    p.add_argument("--rapl", action="store_true",
                   help="Also measure socket CPU power via Intel RAPL (package-0 + "
                        "package-1 energy_uj) during the eval phase. Off by default.")
    p.add_argument("--rapl_baseline_s", type=float, default=60.0,
                   help="Length of each RAPL baseline window (before and after the "
                        "eval phase), in seconds. Only used with --rapl and without "
                        "--bmc; with --bmc the RAPL meter shares the BMC windows.")
    p.add_argument("--chassis_cooldown_s", type=float, default=0.0,
                   help="Pause between the end of the eval phase and the after-baseline "
                        "of the BMC and RAPL meters, in seconds. Sampled and labelled "
                        "'cooldown' in both traces, excluded from the baseline.")

    # Output
    p.add_argument("--results_dir", default=os.environ.get("BENCH_RESULTS", "results"))
    p.add_argument("--no_plots", action="store_true",
                   help="Skip plot generation (faster for headless runs).")
    p.add_argument("--dpi", type=int, default=100)

    return p.parse_args()


# --- Walk-forward evaluation ---

def walk_forward(
    predictor,
    y_all: pd.Series,
    test_start_idx: int,
    horizons: List[int],
    stride: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Rolling-origin evaluation. Returns (y_true_by_h, y_pred_by_h)."""
    n = len(y_all)
    max_h = max(horizons)

    y_true_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    y_pred_by_h: Dict[int, List[float]] = {h: [] for h in horizons}

    origins = range(test_start_idx, n - max_h, stride)
    n_origins = len(origins)
    print(f"    walk-forward: {n_origins} origins, stride={stride}")

    for i, origin_idx in enumerate(origins):
        if i % 50 == 0:
            print(f"    origin {i+1}/{n_origins} (idx={origin_idx})", flush=True)

        if isinstance(predictor, NaivePredictor):
            preds = predictor.predict(origin_idx, horizons)
        else:
            preds = predictor.predict(origin_idx)

        for h in horizons:
            target_idx = origin_idx + h
            if target_idx < n and not np.isnan(preds.get(h, float("nan"))):
                y_true_by_h[h].append(float(y_all.iloc[target_idx]))
                y_pred_by_h[h].append(float(preds[h]))

    return (
        {h: np.array(v) for h, v in y_true_by_h.items()},
        {h: np.array(v) for h, v in y_pred_by_h.items()},
    )


def compute_metrics(
    y_true_by_h: Dict[int, np.ndarray],
    y_pred_by_h: Dict[int, np.ndarray],
) -> Dict[int, Dict[str, float]]:
    """Compute MAE, RMSE, MAPE per horizon."""
    metrics = {}
    for h in y_true_by_h:
        yt = y_true_by_h[h]
        yp = y_pred_by_h[h]
        if len(yt) == 0:
            metrics[h] = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"), "n": 0}
        else:
            metrics[h] = {
                "MAE":  mae(yt, yp),
                "RMSE": rmse(yt, yp),
                "MAPE": mape(yt, yp),
                "n":    len(yt),
            }
    return metrics


# --- Plotting ---

def plot_forecast_sample(
    y_all: pd.Series,
    y_true_by_h: Dict[int, np.ndarray],
    y_pred_by_h: Dict[int, np.ndarray],
    test_start_idx: int,
    stride: int,
    label: str,
    run_dir: str,
    dpi: int = 100,
) -> None:
    """Save a 2-week time-series plot and scatter plot per horizon."""
    os.makedirs(run_dir, exist_ok=True)

    for h in sorted(y_true_by_h.keys()):
        yt = y_true_by_h[h]
        yp = y_pred_by_h[h]
        n = len(yt)
        if n == 0:
            continue

        origin_indices = list(range(test_start_idx, len(y_all) - h, stride))[:n]
        target_indices = [o + h for o in origin_indices if o + h < len(y_all)]
        n2 = min(len(target_indices), n)
        yt = yt[:n2]
        yp = yp[:n2]
        t_axis = y_all.index[target_indices[:n2]]

        # 2-week time series
        fig, ax = plt.subplots(figsize=(14, 4))
        nshow = min(1344, n2)  # 14 days × 96 steps/day
        ax.plot(t_axis[:nshow], yt[:nshow], label="Actual", color="steelblue", lw=1)
        ax.plot(t_axis[:nshow], yp[:nshow], label=label, color="tomato", lw=1, alpha=0.85)
        ax.set_title(f"{label} · h={h} steps ({h*15} min ahead) — first 14 days of test")
        ax.set_ylabel("Load (MW)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"{label}_h{h}_week.png"), dpi=dpi)
        plt.close(fig)

        # Scatter (actual vs predicted)
        fig, ax = plt.subplots(figsize=(5, 5))
        idx_samp = np.random.choice(n2, min(n2, 2000), replace=False)
        ax.scatter(yt[idx_samp], yp[idx_samp], s=3, alpha=0.4, color="steelblue")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("Actual (MW)")
        ax.set_ylabel("Predicted (MW)")
        ax.set_title(f"{label} · h={h} — scatter")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"{label}_h{h}_scatter.png"), dpi=dpi)
        plt.close(fig)

    print(f"  [{label}] plots saved to {run_dir}/")


def plot_extra_comparisons(
    y_all: pd.Series,
    y_true_by_h: Dict[int, np.ndarray],
    y_pred_by_h: Dict[int, np.ndarray],
    test_start_idx: int,
    stride: int,
    label: str,
    run_dir: str,
    dpi: int = 100,
) -> None:
    """Save zoom, multi-horizon, and seasonal comparison plots."""
    horizons = sorted(y_true_by_h.keys())
    max_h = max(horizons)
    n_all = len(y_all)

    _PALETTE = ["tomato", "darkorange", "seagreen", "purple", "brown"]
    h_colors = {h: _PALETTE[i % len(_PALETTE)] for i, h in enumerate(horizons)}

    n_preds = min(len(v) for v in y_true_by_h.values())
    origins = list(range(test_start_idx, n_all - max_h, stride))[:n_preds]
    target_idx_by_h = {h: [o + h for o in origins] for h in horizons}

    def _collect_window(win_start: int, win_end: int) -> Dict[int, tuple]:
        result: Dict[int, tuple] = {}
        for h in horizons:
            ts_w, yt_w, yp_w = [], [], []
            for i, ti in enumerate(target_idx_by_h[h]):
                if win_start <= ti < win_end and i < n_preds:
                    ts_w.append(y_all.index[ti])
                    yt_w.append(float(y_true_by_h[h][i]))
                    yp_w.append(float(y_pred_by_h[h][i]))
            result[h] = (ts_w, yt_w, yp_w)
        return result

    # 14-day zoom, one subplot per horizon
    zoom_start = min(test_start_idx + 7 * 96, n_all - 15 * 96)
    zoom_end = min(zoom_start + 14 * 96, n_all)
    actual_zoom = y_all.iloc[zoom_start:zoom_end]
    win_zoom = _collect_window(zoom_start, zoom_end)

    fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 3.5 * len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        ax.plot(actual_zoom.index, actual_zoom.values, color="steelblue", lw=1,
                label="Actual", zorder=3)
        ts_w, _, yp_w = win_zoom[h]
        if ts_w:
            ax.plot(ts_w, yp_w, "o-", color=h_colors[h], ms=5, lw=1,
                    label=f"Pred h={h} ({h*15} min)", alpha=0.9, zorder=4)
        ax.set_ylabel("Load (MW)")
        ax.set_title(f"h={h} ({h*15} min ahead)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{label} — 14-day zoom: horizon comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f"{label}_zoom14d.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 7-day window, all horizons on one panel
    week_start = min(test_start_idx + 14 * 96, n_all - 8 * 96)
    week_end = min(week_start + 7 * 96, n_all)
    actual_week = y_all.iloc[week_start:week_end]
    win_week = _collect_window(week_start, week_end)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual_week.index, actual_week.values, color="steelblue", lw=1.5,
            label="Actual", zorder=3)
    for h in horizons:
        ts_w, _, yp_w = win_week[h]
        if ts_w:
            ax.plot(ts_w, yp_w, "o--", color=h_colors[h], ms=5, lw=1,
                    label=f"h={h} ({h*15} min)", alpha=0.85, zorder=4)
    ax.set_title(f"{label} — 7-day comparison (all horizons)")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f"{label}_week_all.png"), dpi=dpi)
    plt.close(fig)

    # Winter vs Summer
    def _find_month_window(month: int, days: int = 7) -> Optional[Tuple[int, int]]:
        for i in range(test_start_idx, n_all - days * 96):
            if y_all.index[i].month == month:
                return i, i + days * 96
        return None

    winter = _find_month_window(1)  # January
    summer = _find_month_window(7)  # July

    if winter and summer:
        fig, axes = plt.subplots(1, 2, figsize=(18, 4), sharey=True)
        for ax, (ws, we), season_name in zip(
            axes, [winter, summer], ["Winter (January)", "Summer (July)"]
        ):
            actual_s = y_all.iloc[ws:we]
            ax.plot(actual_s.index, actual_s.values, color="steelblue", lw=1.5,
                    label="Actual", zorder=3)
            win_s = _collect_window(ws, we)
            for h in horizons:
                ts_w, _, yp_w = win_s[h]
                if ts_w:
                    ax.plot(ts_w, yp_w, "o--", color=h_colors[h], ms=5, lw=1,
                            label=f"h={h} ({h*15} min)", alpha=0.85)
            ax.set_title(f"{label} — {season_name}")
            ax.set_ylabel("Load (MW)")
            ax.set_xlabel("Date")
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"{label} — Seasonal comparison", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"{label}_seasonal.png"), dpi=dpi)
        plt.close(fig)
    else:
        missing = (["January"] if not winter else []) + (["July"] if not summer else [])
        print(f"  [{label}] seasonal plot skipped: {', '.join(missing)} not in test period")

    print(f"  [{label}] extra plots saved to {run_dir}/")


# --- Result logging ---

def log_metrics(
    run_id: str,
    label: str,
    metrics_by_h: Dict[int, Dict[str, float]],
    fit_energy,
    eval_energy,
    run_dir: str,
    step_minutes: int = 15,
    bmc_summary: Optional[dict] = None,
    host_start: Optional[dict] = None,
    host_end: Optional[dict] = None,
    rapl_summary: Optional[dict] = None,
    cc_cpu_mode: Optional[str] = None,
    ct_cpu_avg_w: Optional[float] = None,
) -> None:
    """Append one row per horizon to the run's metrics.csv file."""
    bmc_summary = bmc_summary or {}
    rapl_summary = rapl_summary or {}
    # Always emitted so the schema stays rectangular whether or not --bmc is on.
    BMC_COLS = ("bmc_gross_kwh", "bmc_incremental_kwh", "bmc_baseline_w",
                "bmc_baseline_std_w", "bmc_mean_w", "bmc_peak_w", "bmc_run_window_s")
    # Same rule for --rapl.
    RAPL_COLS = ("rapl_gross_kwh", "rapl_incremental_kwh", "rapl_baseline_w",
                 "rapl_baseline_std_w", "rapl_mean_w", "rapl_peak_w", "rapl_run_window_s")
    host_start = host_start or {}
    host_end = host_end or {}
    csv_path = os.path.join(run_dir, "metrics.csv")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _r(val, decimals=8):
        if val is None:
            return None
        try:
            return None if np.isnan(val) else round(float(val), decimals)
        except (TypeError, ValueError):
            return None

    for h, m in sorted(metrics_by_h.items()):
        row = {
            "run_id":               run_id,
            "timestamp":            ts,
            "model":                label,
            "horizon_steps":        h,
            "horizon_minutes":      h * step_minutes,
            "n_predictions":        m.get("n", 0),
            "MAE":                  _r(m["MAE"],  4),
            "RMSE":                 _r(m["RMSE"], 4),
            "MAPE_pct":             _r(m["MAPE"], 4),
            "fit_cc_energy_kwh":    _r(fit_energy.energy_kwh)    if fit_energy  else None,
            "fit_cc_emissions_kg":  _r(fit_energy.emissions_kg)  if fit_energy  else None,
            "fit_duration_s":       _r(fit_energy.duration_s, 2) if fit_energy  else None,
            "eval_cc_energy_kwh":   _r(eval_energy.energy_kwh)   if eval_energy else None,
            "eval_cc_emissions_kg": _r(eval_energy.emissions_kg) if eval_energy else None,
            "eval_duration_s":      _r(eval_energy.duration_s, 2) if eval_energy else None,
            "fit_ct_energy_kwh":    _r(fit_energy.ct_energy_kwh)   if fit_energy  else None,
            "fit_ct_emissions_kg":  _r(fit_energy.ct_emissions_kg) if fit_energy  else None,
            "eval_ct_energy_kwh":   _r(eval_energy.ct_energy_kwh)  if eval_energy else None,
            "eval_ct_emissions_kg": _r(eval_energy.ct_emissions_kg) if eval_energy else None,
            "energy_tool":          fit_energy.tool if fit_energy else None,
        }
        for col in BMC_COLS:
            row[col] = _r(bmc_summary.get(col), 8)
        for col in RAPL_COLS:
            row[col] = _r(rapl_summary.get(col), 8)
        # Host conditions at run start / end, so idle state is on record per row.
        for key, name in (("load1", "host_load1"),
                          ("gpu_util_percent", "host_gpu_util_pct"),
                          ("cpu_util_percent", "host_cpu_util_pct")):
            row[f"{name}_start"] = _r(host_start.get(key), 3)
            row[f"{name}_end"] = _r(host_end.get(key), 3)
        # Which CPU source each tracker actually used, so a row's CPU energy
        # can be read as measured (RAPL) or modelled (cpu_load). ct_cpu_avg_w
        # is CarbonTracker's own average CPU watts for the eval phase (0.0 on
        # a permission failure, empty when it reported None).
        row["cc_cpu_mode"] = cc_cpu_mode
        row["ct_cpu_avg_w"] = _r(ct_cpu_avg_w, 3)
        append_run_csv(csv_path, row)

    print(f"  [{label}] metrics → {csv_path}")


# --- Helpers ---

def _load_tuned_params(args: argparse.Namespace, model_name: str) -> dict:
    """Load hyperparameters from the JSON file produced by tune_hyperparams.py."""
    if not getattr(args, "tuned_params", None):
        return {}
    if not os.path.exists(args.tuned_params):
        print(f"  [warn] --tuned_params file not found: {args.tuned_params}")
        return {}
    with open(args.tuned_params, "r", encoding="utf-8") as f:
        all_params = json.load(f)
    return all_params.get(model_name, {})


# --- Per-model runner ---

def run_model(
    model_name: str,
    y_all: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    test_start_idx: int,
    horizons: List[int],
    stride: int,
    args: argparse.Namespace,
    run_id: str,
    run_dir: str,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Fit, walk-forward evaluate, log, and plot one model. Returns (metrics, y_true, y_pred)."""
    host_start = sample_host_conditions()
    energy_tool = args.energy_tool
    cc_enabled = energy_tool in ("codecarbon", "both")
    ct_enabled = energy_tool in ("carbontracker", "both")
    cc_dir = os.path.join(run_dir, "codecarbon")
    ct_dir = os.path.join(run_dir, "carbontracker")

    def _make_meters(phase: str, cc_on: bool = True):
        tag = f"{run_id}_{model_name}_{phase}"
        cc = EnergyMeter(tag, cc_dir, enabled=(cc_enabled and cc_on))
        ct = CarbonTrackerMeter(tag, ct_dir, enabled=(ct_enabled and cc_on))
        # Build the tracker objects now so start() is cheap and their
        # construction never lands inside a chassis-meter window.
        cc.prepare(); ct.prepare()
        return cc, ct

    def _combined(cc_meter: EnergyMeter, ct_meter: CarbonTrackerMeter):
        cc_res = cc_meter.stop()
        ct_res = ct_meter.stop()
        return cc_res.with_ct(ct_res)

    # Build and fit predictor
    if model_name == "naive":
        predictor = NaivePredictor(s=getattr(args, "naive_s", 96))
        # Naive has negligible cost — skip energy tracking for fit
        cc_fit, ct_fit = _make_meters("fit", cc_on=False)
        cc_fit.start(); ct_fit.start()
        predictor.fit(y_train)
        fit_energy = _combined(cc_fit, ct_fit)
        predictor.set_series(y_all)

    elif model_name in ("arima", "sarima"):
        cfg = ArimaConfig(
            order=tuple(args.arima_order if model_name == "arima" else args.sarima_order),
            seasonal_order=(0, 0, 0, 0) if model_name == "arima"
                           else tuple(args.sarima_seasonal_order),
        )
        predictor = ARIMAPredictor(
            cfg=cfg, horizons=horizons, maxiter=args.maxiter,
            max_train_obs=args.max_train_obs, label=model_name,
        )
        cc_fit, ct_fit = _make_meters("fit")
        cc_fit.start(); ct_fit.start()
        predictor.fit(y_train)
        fit_energy = _combined(cc_fit, ct_fit)
        predictor.set_series(y_all, test_start_idx)

    elif model_name == "linear":
        tuned = _load_tuned_params(args, "linear")
        alpha = tuned.get("alpha", 1.0)
        if tuned:
            print(f"  [linear] using tuned params: alpha={alpha:.4g}")
        forecaster = make_linear_forecaster(alpha=alpha, use_holidays=args.use_holidays,
                                             seasonal_period=args.seasonal_residual)
        predictor = MLPredictor(forecaster, label="linear")
        cc_fit, ct_fit = _make_meters("fit")
        cc_fit.start(); ct_fit.start()
        predictor.fit(y_train, horizons)
        fit_energy = _combined(cc_fit, ct_fit)
        predictor.set_series(y_all)

    elif model_name == "lgbm":
        tuned = _load_tuned_params(args, "lgbm")
        if tuned:
            print(f"[lgbm] using tuned params: {tuned}")
        forecaster = make_lgbm_forecaster(use_holidays=args.use_holidays,
                                           seasonal_period=args.seasonal_residual, **tuned)
        predictor = MLPredictor(forecaster, label="lgbm")
        cc_fit, ct_fit = _make_meters("fit")
        cc_fit.start(); ct_fit.start()
        predictor.fit(y_train, horizons)
        fit_energy = _combined(cc_fit, ct_fit)
        predictor.set_series(y_all)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"  [{model_name}] fit done in {fit_energy.duration_s:.1f}s")

    # Walk-forward evaluation
    # RAPL samples continuously and is told when phases switch; BMC blocks on
    # its own baseline windows. Starting RAPL first and stopping it last makes
    # the two meters share identical "before" / run / "after" windows. Without
    # --bmc the runner sleeps the RAPL baseline itself so the windows still
    # exist.
    bmc_on = bool(getattr(args, "bmc", False))
    rapl_on = bool(getattr(args, "rapl", False))
    eval_tag = f"{run_id}_{model_name}_eval"
    bmc_eval = BmcPowerMeter(
        eval_tag, run_dir,
        baseline_before_s=args.bmc_baseline_s,
        baseline_after_s=args.bmc_baseline_s,
        enabled=bmc_on,
        cooldown_s=args.chassis_cooldown_s,
    )
    rapl_eval = RaplPowerMeter(
        eval_tag, run_dir,
        baseline_before_s=args.rapl_baseline_s,
        baseline_after_s=args.rapl_baseline_s,
        enabled=rapl_on,
        cooldown_s=args.chassis_cooldown_s,
    )
    # Tracker objects are built before any chassis window opens, so their
    # construction cost is not inside the measured run.
    cc_eval, ct_eval = _make_meters("eval")
    rapl_eval.start()                       # RAPL phase "before"
    bmc_eval.start()                        # blocks bmc_baseline_s if --bmc
    if rapl_on and not bmc_on:
        time.sleep(args.rapl_baseline_s)
    rapl_eval.begin_run()                   # RAPL phase "run"
    cc_eval.start(); ct_eval.start()
    y_true_by_h, y_pred_by_h = walk_forward(predictor, y_all, test_start_idx, horizons, stride)
    rapl_eval.end_run()                     # RAPL phase "cooldown" then "after"; run = walk-forward + tracker starts only
    eval_energy = _combined(cc_eval, ct_eval)
    cc_cpu_mode = cc_eval.cpu_mode()
    ct_cpu_avg_w = ct_eval.cpu_avg_w()
    bmc_eval.stop()                         # blocks cooldown + bmc_baseline_s if --bmc
    if rapl_on and not bmc_on:
        time.sleep(args.chassis_cooldown_s + args.rapl_baseline_s)
    rapl_eval.stop()
    bmc_summary = bmc_eval.summary()
    rapl_summary = rapl_eval.summary()
    print(f"  [{model_name}] evaluation done in {eval_energy.duration_s:.1f}s")
    if bmc_summary:
        print(f"  [{model_name}] BMC: gross={bmc_summary['bmc_gross_kwh']:.3e} kWh  "
              f"incremental={bmc_summary['bmc_incremental_kwh']:.3e} kWh  "
              f"baseline={bmc_summary['bmc_baseline_w']:.1f}±"
              f"{bmc_summary['bmc_baseline_std_w']:.1f} W  "
              f"mean={bmc_summary['bmc_mean_w']:.1f} W  peak={bmc_summary['bmc_peak_w']:.1f} W")
    if rapl_summary:
        print(f"  [{model_name}] RAPL: gross={rapl_summary['rapl_gross_kwh']:.3e} kWh  "
              f"incremental={rapl_summary['rapl_incremental_kwh']:.3e} kWh  "
              f"baseline={rapl_summary['rapl_baseline_w']:.1f}±"
              f"{rapl_summary['rapl_baseline_std_w']:.1f} W  "
              f"mean={rapl_summary['rapl_mean_w']:.1f} W  peak={rapl_summary['rapl_peak_w']:.1f} W  "
              f"windows={rapl_summary['rapl_baseline_before_s']:.0f}s/"
              f"{rapl_summary['rapl_run_window_s']:.0f}s/{rapl_summary['rapl_baseline_after_s']:.0f}s")

    # Metrics
    metrics_by_h = compute_metrics(y_true_by_h, y_pred_by_h)
    for h, m in sorted(metrics_by_h.items()):
        print(f"    h={h:3d} ({h*15:4d}min)  MAE={m['MAE']:.1f}  "
              f"RMSE={m['RMSE']:.1f}  MAPE={m['MAPE']:.2f}%  n={m['n']}")

    # Plots
    if not args.no_plots:
        plot_forecast_sample(y_all, y_true_by_h, y_pred_by_h,
                             test_start_idx, stride, label=model_name,
                             run_dir=run_dir, dpi=args.dpi)
        plot_extra_comparisons(y_all, y_true_by_h, y_pred_by_h,
                               test_start_idx, stride, label=model_name,
                               run_dir=run_dir, dpi=args.dpi)

    host_end = sample_host_conditions()
    log_metrics(run_id, model_name, metrics_by_h, fit_energy, eval_energy, run_dir,
                step_minutes=getattr(args, "_step_minutes", 15),
                bmc_summary=bmc_summary,
                host_start=host_start, host_end=host_end,
                rapl_summary=rapl_summary,
                cc_cpu_mode=cc_cpu_mode, ct_cpu_avg_w=ct_cpu_avg_w)

    return metrics_by_h, y_true_by_h, y_pred_by_h


# --- Predictions collector ---

def build_predictions_df(
    model_name: str,
    y_all: pd.Series,
    y_true_by_h: Dict[int, np.ndarray],
    y_pred_by_h: Dict[int, np.ndarray],
    test_start_idx: int,
    stride: int,
) -> pd.DataFrame:
    """Convert walk-forward arrays into a tidy long-format DataFrame."""
    rows: List[dict] = []
    n = len(y_all)

    for h, yt in y_true_by_h.items():
        yp = y_pred_by_h[h]
        n_pred = len(yt)
        if n_pred == 0:
            continue
        origin_indices = list(range(test_start_idx, n - h, stride))[:n_pred]
        target_indices = [o + h for o in origin_indices if o + h < n]
        n2 = min(len(target_indices), n_pred)
        timestamps = y_all.index[target_indices[:n2]]
        for i, ts in enumerate(timestamps):
            rows.append({
                "timestamp": ts,
                "model":     model_name,
                "horizon":   h,
                "actual":    float(yt[i]),
                "predicted": float(yp[i]),
            })

    return pd.DataFrame(rows)


# --- Summary table ---

def print_comparison_table(
    all_metrics: Dict[str, Dict[int, Dict[str, float]]],
    horizons: List[int],
) -> None:
    print("\n" + "=" * 70)
    print("WALK-FORWARD BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<10} {'h':>4} {'h(min)':>7}  {'MAE':>8}  {'RMSE':>8}  {'MAPE%':>7}  {'n':>6}")
    print("-" * 70)

    for model_name, metrics_by_h in all_metrics.items():
        for h in sorted(horizons):
            m = metrics_by_h.get(h, {})
            print(f"{model_name:<10} {h:>4} {h*15:>7}  "
                  f"{m.get('MAE', float('nan')):>8.1f}  "
                  f"{m.get('RMSE', float('nan')):>8.1f}  "
                  f"{m.get('MAPE', float('nan')):>7.2f}  "
                  f"{m.get('n', 0):>6}")
        print()

    print("=" * 70)


# --- Main ---

def main() -> None:
    args = parse_args()
    args._step_minutes = freq_to_minutes(args.freq)
    run_id = "wf_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(args.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun folder: {run_dir}")

    print(f"\n[{run_id}] Loading data from {args.data} …")
    df = load_timeseries(args.data, timestamp_col=args.timestamp_col,
                         target_col=args.target_col, freq=args.freq,
                         delimiter=args.delimiter)
    df = clean_timeseries(df, impute_method="interpolate")
    y_all = df["y"]
    print(f"  Loaded {len(y_all):,} obs from {y_all.index[0]} to {y_all.index[-1]}")

    df_train, df_test = split_last_n_years(df, args.test_years)
    y_train = df_train["y"]
    y_test = df_test["y"]
    test_start_idx = len(y_train)

    print(f"  Train: {len(y_train):,} obs  ({y_train.index[0]} → {y_train.index[-1]})")
    print(f"  Test : {len(y_test):,} obs  ({y_test.index[0]} → {y_test.index[-1]})")
    print(f"  Horizons: {args.horizons}  Stride: {args.stride}  Models: {args.models}")

    all_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    all_pred_frames: List[pd.DataFrame] = []

    for model_name in args.models:
        print(f"\n{'─'*60}\nRunning model: {model_name.upper()}\n{'─'*60}")
        try:
            metrics_by_h, y_true_by_h, y_pred_by_h = run_model(
                model_name=model_name, y_all=y_all, y_train=y_train, y_test=y_test,
                test_start_idx=test_start_idx, horizons=args.horizons,
                stride=args.stride, args=args, run_id=run_id, run_dir=run_dir,
            )
            all_metrics[model_name] = metrics_by_h
            all_pred_frames.append(
                build_predictions_df(model_name, y_all, y_true_by_h, y_pred_by_h,
                                     test_start_idx, args.stride)
            )
        except Exception as exc:
            print(f"  [{model_name}] ERROR: {exc}")
            import traceback
            traceback.print_exc()

    if all_pred_frames:
        predictions_path = os.path.join(run_dir, "predictions.parquet")
        pd.concat(all_pred_frames, ignore_index=True).to_parquet(predictions_path, index=False)
        print(f"\nPredictions saved → {predictions_path}")

    if all_metrics:
        print_comparison_table(all_metrics, args.horizons)

    print(f"\nDone. Results in {run_dir}/")


if __name__ == "__main__":
    main()
