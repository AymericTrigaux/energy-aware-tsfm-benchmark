"""power_curves.py — Power-vs-time measurement and analysis.

Subcommands:
  measure    Measure per-second power curves for one model's walk-forward inference
  analyze    Post-process power-curve runs (energy cross-validation, per-origin plots)
  composite  Compose per-model power-curve runs into a single comparison figure

Usage:
    python power_curves.py measure --model chronos_bolt_mini --n_origins 200
    python power_curves.py measure --model lgbm --n_origins 2000
    python power_curves.py analyze --tag cmp2 dense
    python power_curves.py composite --tag cmp2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.palette import MODEL_COLOR as _MODEL_COLOR

_CLASSICAL_MODELS = {"naive", "linear", "lgbm", "arima", "sarima"}

_MODEL_ORDER = [
    "naive", "linear", "lgbm", "arima", "sarima",
    "chronos_bolt_mini", "chronos_mini",
    "timesfm_200m", "moirai_small", "lag_llama",
]


# --- CC power extraction (shared by all subcommands) ---

def reextract_cc(emissions_csv: Path, cc_start_offset_s: float = 0.0) -> pd.DataFrame:
    """Re-extract per-sample power from raw CC emissions.csv with trailing-row clip."""
    if not emissions_csv.exists():
        return pd.DataFrame(columns=["elapsed_s", "power_w", "cumulative_kwh"])
    df = pd.read_csv(emissions_csv)
    if "duration" not in df.columns or "energy_consumed" not in df.columns:
        return pd.DataFrame(columns=["elapsed_s", "power_w", "cumulative_kwh"])

    duration = df["duration"].to_numpy(dtype=float)
    cum_kwh  = df["energy_consumed"].to_numpy(dtype=float)

    if len(duration) >= 3:
        dts = np.diff(duration)
        median_dt = float(np.median(dts[:-1]))
        if dts[-1] < 0.5 * median_dt:
            duration = duration[:-1]
            cum_kwh  = cum_kwh[:-1]

    elapsed = duration + cc_start_offset_s
    if len(duration) == 0:
        return pd.DataFrame(columns=["elapsed_s", "power_w", "cumulative_kwh"])
    elif len(duration) == 1:
        dt    = max(duration[0], 1e-6)
        power = np.array([cum_kwh[0] * 3.6e6 / dt])
    else:
        dt       = np.diff(duration, prepend=0.0)
        dt       = np.where(dt <= 0, np.nan, dt)
        delta_kwh = np.diff(cum_kwh, prepend=0.0)
        power    = delta_kwh * 3.6e6 / dt

    return pd.DataFrame({"elapsed_s": elapsed, "power_w": power, "cumulative_kwh": cum_kwh})


# --- measure ---

class _CCFlusher:
    """Background thread that calls EmissionsTracker.flush() at a fixed interval."""

    def __init__(self, energy_meter, interval_s: float):
        self.meter    = energy_meter
        self.interval = float(interval_s)
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.meter.enabled or self.meter._tracker is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.meter._tracker.flush()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _build_classical(model_name, args, y_train, y_all, test_start_idx, horizons):
    from src.models import (
        ArimaConfig, ARIMAPredictor, NaivePredictor, MLPredictor,
        make_linear_forecaster, make_lgbm_forecaster,
    )
    if model_name == "naive":
        predictor = NaivePredictor(s=args.naive_s)
        predictor.fit(y_train); predictor.set_series(y_all)
        return predictor
    if model_name in ("arima", "sarima"):
        cfg = ArimaConfig(
            order=tuple(args.arima_order if model_name == "arima" else args.sarima_order),
            seasonal_order=(0, 0, 0, 0) if model_name == "arima"
                           else tuple(args.sarima_seasonal_order),
        )
        predictor = ARIMAPredictor(cfg=cfg, horizons=horizons, maxiter=args.maxiter,
                                   max_train_obs=args.max_train_obs, label=model_name)
        predictor.fit(y_train); predictor.set_series(y_all, test_start_idx)
        return predictor
    if model_name == "linear":
        predictor = MLPredictor(make_linear_forecaster(alpha=1.0), label="linear")
        predictor.fit(y_train, horizons); predictor.set_series(y_all)
        return predictor
    if model_name == "lgbm":
        predictor = MLPredictor(make_lgbm_forecaster(), label="lgbm")
        predictor.fit(y_train, horizons); predictor.set_series(y_all)
        return predictor
    raise ValueError(f"Unknown classical model: {model_name}")


def _build_tsfm(model_name, args, y_train, y_all, test_start_idx, horizons):
    import inspect
    from src.models_tsfm import TSFM_MODELS
    if model_name not in TSFM_MODELS:
        raise ValueError(f"Unknown TSFM: {model_name}. Available: {sorted(TSFM_MODELS)}")
    cls      = TSFM_MODELS[model_name]
    accepted: set = set()
    for klass in cls.__mro__:
        if klass is object:
            continue
        try:
            accepted.update(inspect.signature(klass.__init__).parameters.keys())
        except (TypeError, ValueError):
            pass
    kwargs: dict = {"horizons": horizons}
    for attr in ("device", "context_length", "num_samples", "batch_size",
                 "patch_size", "point_forecast_mode"):
        if attr in accepted and getattr(args, attr, None) is not None:
            kwargs[attr] = getattr(args, attr)
    predictor = cls(**kwargs)
    predictor.fit(y_train); predictor.set_series(y_all, test_start_idx)
    return predictor


def _walk_forward(predictor, y_all, test_start_idx, horizons, stride, n_origins, batch_size=1):
    n          = len(y_all)
    max_h      = max(horizons)
    all_origins = list(range(test_start_idx, n - max_h, stride))[:n_origins]
    n_total    = len(all_origins)
    use_batch  = batch_size > 1 and hasattr(predictor, "predict_batch")
    is_naive   = type(predictor).__name__ == "NaivePredictor"
    print(f"    walk-forward: {n_total} origins, stride={stride}, batched={use_batch}", flush=True)
    if use_batch:
        for chunk_start in range(0, n_total, batch_size):
            chunk = all_origins[chunk_start: chunk_start + batch_size]
            predictor.predict_batch(chunk)
    else:
        for i, origin_idx in enumerate(all_origins):
            if i % 50 == 0:
                print(f"    origin {i + 1}/{n_total}", flush=True)
            if is_naive:
                predictor.predict(origin_idx, horizons)
            else:
                predictor.predict(origin_idx)
    return n_total


def _extract_nvsmi(nvsmi_csv: Path, t0: float) -> pd.DataFrame:
    if not nvsmi_csv.exists():
        return pd.DataFrame(columns=["elapsed_s", "power_w"])
    df = pd.read_csv(nvsmi_csv)
    if "timestamp_s" not in df.columns or "power_w" not in df.columns:
        return pd.DataFrame(columns=["elapsed_s", "power_w"])
    return pd.DataFrame({
        "elapsed_s": df["timestamp_s"].to_numpy() - t0,
        "power_w":   df["power_w"].to_numpy(),
    })


def _plot_overlay(cc_df, nv_df, model_name, out_path, dpi=120):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if not cc_df.empty:
        ax.plot(cc_df["elapsed_s"], cc_df["power_w"],
                label="CodeCarbon (total system, flushed @1Hz)", color="steelblue", lw=1.2)
    if not nv_df.empty:
        ax.plot(nv_df["elapsed_s"], nv_df["power_w"],
                label="nvidia-smi (GPU only, polled @1Hz)", color="darkorange", lw=1.2, alpha=0.9)
    ax.set_xlabel("Elapsed time during inference (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Power vs time — {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def cmd_measure(args: argparse.Namespace) -> None:
    from src.data import load_timeseries, clean_timeseries, split_last_n_years
    from src.metrics import EnergyMeter, NvidiaSmiMeter

    model_name   = args.model
    is_classical = model_name in _CLASSICAL_MODELS
    ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    run_dir = Path(args.results_dir) / f"powercurve_{model_name}_{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun folder: {run_dir}")

    print(f"\nLoading data from {args.data} …")
    df = load_timeseries(args.data, timestamp_col=args.timestamp_col,
                         target_col=args.target_col, freq=args.freq,
                         delimiter=args.delimiter)
    df = clean_timeseries(df, impute_method="interpolate")
    y_all = df["y"]
    df_train, _ = split_last_n_years(df, args.test_years)
    y_train       = df_train["y"]
    test_start_idx = len(y_train)
    print(f"  {len(y_all):,} obs total · train={len(y_train):,} · test={len(y_all)-test_start_idx:,}")

    print(f"\nBuilding & fitting {model_name} …")
    t_build = time.time()
    if is_classical:
        predictor = _build_classical(model_name, args, y_train, y_all,
                                     test_start_idx, args.horizons)
    else:
        predictor = _build_tsfm(model_name, args, y_train, y_all,
                                test_start_idx, args.horizons)
    print(f"  ready in {time.time() - t_build:.1f}s")

    cc_dir = run_dir / "codecarbon"
    nv_dir = run_dir / "nvidia_smi"
    cc_tag = f"powercurve_{model_name}_{ts}_cc"
    nv_tag = f"powercurve_{model_name}_{ts}_nv"

    cc_meter = EnergyMeter(cc_tag, str(cc_dir), enabled=not args.no_cc)
    nv_meter = NvidiaSmiMeter(
        nv_tag, str(nv_dir),
        gpu_index=args.gpu_index,
        poll_interval_ms=args.nvsmi_interval_ms,
        enabled=not args.no_nvsmi,
    )

    t0 = time.time()
    nv_meter.start()
    cc_meter.start()
    flusher = _CCFlusher(cc_meter, args.cc_poll_interval_s)
    flusher.start()
    time.sleep(2.0)

    print(f"\nRunning walk-forward (≤ {args.n_origins} origins) …")
    n_done = _walk_forward(predictor, y_all, test_start_idx, args.horizons,
                           args.stride, n_origins=args.n_origins, batch_size=args.batch_size)
    time.sleep(2.0)

    flusher.stop()
    cc_result = cc_meter.stop()
    nv_result = nv_meter.stop()
    duration_s = time.time() - t0

    print(f"\nInference window: {duration_s:.1f}s, {n_done} origins")

    cc_csv  = cc_dir / "emissions.csv"
    nv_csv  = Path(nv_result.details_path) if nv_result.details_path \
              else nv_dir / f"{nv_tag}_gpu_power.csv"
    cc_start_offset_s = (cc_meter._t0 - t0) if cc_meter._t0 is not None else 0.0
    cc_curve = reextract_cc(cc_csv, cc_start_offset_s)
    nv_curve = _extract_nvsmi(nv_csv, t0)

    cc_out = run_dir / "cc_power_log.csv"
    nv_out = run_dir / "nvsmi_power_log.csv"
    cc_curve.to_csv(cc_out, index=False)
    nv_curve.to_csv(nv_out, index=False)
    print(f"  cc_power_log    → {cc_out} ({len(cc_curve)} rows)")
    print(f"  nvsmi_power_log → {nv_out} ({len(nv_curve)} rows)")

    if not args.no_plot:
        plot_path = run_dir / "power_curve.png"
        _plot_overlay(cc_curve, nv_curve, model_name, plot_path, dpi=args.dpi)
        print(f"  plot            → {plot_path}")

    summary = {
        "model": model_name, "is_classical": is_classical,
        "run_dir": str(run_dir), "n_origins": n_done, "stride": args.stride,
        "horizons": args.horizons, "duration_s": round(duration_s, 2), "t0_unix": t0,
        "data": args.data,
        "cc": {"enabled": not args.no_cc, "tool": cc_result.tool,
               "energy_kwh": cc_result.energy_kwh, "emissions_kg": cc_result.emissions_kg,
               "duration_s": cc_result.duration_s, "n_samples": int(len(cc_curve)),
               "details_path": cc_result.details_path},
        "nvidia_smi": {"enabled": not args.no_nvsmi, "tool": nv_result.tool,
                       "energy_kwh": nv_result.energy_kwh, "emissions_kg": nv_result.emissions_kg,
                       "duration_s": nv_result.duration_s, "n_samples": int(len(nv_curve)),
                       "details_path": nv_result.details_path,
                       "gpu_index": args.gpu_index, "poll_ms": args.nvsmi_interval_ms},
        "cc_poll_interval_s": args.cc_poll_interval_s,
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  summary         → {summary_path}\n\nDone. {run_dir}/")


# --- analyze ---

@dataclass
class _Run:
    model:   str
    run_dir: Path
    meta:    dict
    cc:      pd.DataFrame
    nv:      pd.DataFrame


def _discover_runs(results_dir: Path, tags: List[str]) -> Dict[str, Path]:
    by_model: Dict[str, Path] = {}
    for tag in tags:
        for run_dir in sorted(results_dir.glob(f"powercurve_*_{tag}")):
            summary = run_dir / "summary.json"
            if not summary.exists():
                continue
            try:
                meta = json.loads(summary.read_text())
            except json.JSONDecodeError:
                continue
            model = meta.get("model")
            if model is None:
                continue
            existing = by_model.get(model)
            if existing is None or run_dir.stat().st_mtime > existing.stat().st_mtime:
                by_model[model] = run_dir
    return by_model


def _load_run(run_dir: Path) -> _Run:
    meta = json.loads((run_dir / "summary.json").read_text())
    cc   = reextract_cc(run_dir / "codecarbon" / "emissions.csv")
    nv_path = run_dir / "nvsmi_power_log.csv"
    nv = pd.read_csv(nv_path) if nv_path.exists() else pd.DataFrame()
    return _Run(model=meta["model"], run_dir=run_dir, meta=meta, cc=cc, nv=nv)


def _trapezoid_kwh(times_s: np.ndarray, power_w: np.ndarray) -> float:
    if len(times_s) < 2:
        return 0.0
    mask = ~(np.isnan(times_s) | np.isnan(power_w))
    t, w = times_s[mask], power_w[mask]
    if len(t) < 2:
        return 0.0
    return float(np.trapz(w, t)) / 3.6e6


def _cross_validate_energy(runs: List[_Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        cc_sum = float(run.meta["cc"]["energy_kwh"] or 0.0)
        nv_sum = float(run.meta["nvidia_smi"]["energy_kwh"] or 0.0)
        cc_int = _trapezoid_kwh(run.cc["elapsed_s"].to_numpy(), run.cc["power_w"].to_numpy()) \
                 if not run.cc.empty else 0.0
        nv_int = _trapezoid_kwh(run.nv["elapsed_s"].to_numpy(), run.nv["power_w"].to_numpy()) \
                 if not run.nv.empty else 0.0
        cc_re = (cc_int - cc_sum) / cc_sum * 100 if cc_sum else float("nan")
        nv_re = (nv_int - nv_sum) / nv_sum * 100 if nv_sum else float("nan")
        rows.append({"model": run.model, "n_origins": run.meta["n_origins"],
                     "duration_s": run.meta["duration_s"],
                     "cc_summary_Wh": cc_sum * 1000, "cc_integrated_Wh": cc_int * 1000,
                     "cc_rel_err_pct": cc_re,
                     "nv_summary_Wh": nv_sum * 1000, "nv_integrated_Wh": nv_int * 1000,
                     "nv_rel_err_pct": nv_re})
    return pd.DataFrame(rows)


def _plot_per_origin_curves(runs: List[_Run], out_path: Path, dpi: int) -> None:
    fig, (ax_nv, ax_cc) = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for run in runs:
        n = run.meta["n_origins"] or 1
        color = _MODEL_COLOR.get(run.model)
        label = f"{run.model} (n={n}, win={run.meta['duration_s']:.0f}s)"
        if not run.nv.empty:
            ax_nv.plot(run.nv["elapsed_s"], run.nv["power_w"] / n * 1000,
                       color=color, lw=1.4, label=label, alpha=0.9)
        if not run.cc.empty:
            ax_cc.plot(run.cc["elapsed_s"], run.cc["power_w"] / n * 1000,
                       color=color, lw=1.4, marker="o", ms=3.5, label=label, alpha=0.9)
    for ax, title in [(ax_nv, "nvidia-smi GPU power per origin"),
                      (ax_cc, "CodeCarbon total system power per origin")]:
        ax.set_title(title)
        ax.set_ylabel("mW / origin")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8, ncols=2)
    ax_cc.set_xlabel("Elapsed time (s)")
    fig.suptitle("Per-origin normalized power", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_energy_per_origin_bar(runs: List[_Run], out_path: Path, dpi: int) -> None:
    rows = []
    for run in runs:
        n = run.meta["n_origins"] or 1
        cc_kwh = _trapezoid_kwh(run.cc["elapsed_s"].to_numpy(), run.cc["power_w"].to_numpy()) \
                 if not run.cc.empty else 0.0
        nv_kwh = _trapezoid_kwh(run.nv["elapsed_s"].to_numpy(), run.nv["power_w"].to_numpy()) \
                 if not run.nv.empty else 0.0
        rows.append({"model": run.model,
                     "cc_J_per_origin": cc_kwh * 3.6e6 / n,
                     "nv_J_per_origin": nv_kwh * 3.6e6 / n})
    df = pd.DataFrame(rows)
    ordered = [m for m in _MODEL_ORDER if m in df["model"].values]
    df = df.set_index("model").loc[ordered].reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df["cc_J_per_origin"], w, label="CodeCarbon (total system)", color="steelblue")
    ax.bar(x + w/2, df["nv_J_per_origin"], w, label="nvidia-smi (GPU only)", color="darkorange")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("Energy per origin (J)")
    ax.set_title("Energy per origin")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_lag_llama_batches(run: _Run, batch_size: int, out_path: Path, dpi: int) -> None:
    if run.nv.empty:
        return
    nv = run.nv.copy()
    baseline  = float(np.percentile(nv["power_w"], 10))
    threshold = baseline + 0.4 * (float(nv["power_w"].max()) - baseline)
    active    = nv["power_w"] >= threshold
    if not active.any():
        active_start, active_end = float(nv["elapsed_s"].min()), float(nv["elapsed_s"].max())
    else:
        active_start = float(nv.loc[active, "elapsed_s"].min())
        active_end   = float(nv.loc[active, "elapsed_s"].max())
    n_origins = run.meta["n_origins"]
    n_batches = max(1, int(np.ceil(n_origins / batch_size)))
    boundaries = np.linspace(active_start, active_end, n_batches + 1)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(run.cc["elapsed_s"], run.cc["power_w"], color="steelblue", lw=1.4,
            label="CodeCarbon (total system)")
    ax.plot(run.nv["elapsed_s"], run.nv["power_w"], color="darkorange", lw=1.4,
            label="nvidia-smi (GPU)")
    for b in boundaries[1:-1]:
        ax.axvline(b, color="gray", lw=0.6, ls="--", alpha=0.6, zorder=0)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Lag-Llama power curve with batch boundaries "
                 f"(batch_size={batch_size}, {n_batches} batches)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_lag_llama_cpu_ram(run: _Run, out_path: Path, dpi: int) -> None:
    if run.cc.empty or run.nv.empty:
        return

    def _active_bounds(t, w):
        baseline  = float(np.percentile(w, 10))
        threshold = baseline + 0.4 * (float(w.max()) - baseline)
        mask = w >= threshold
        if not mask.any():
            return float(t.min()), float(t.max())
        return float(t[mask].min()), float(t[mask].max())

    nv_t = run.nv["elapsed_s"].to_numpy(); nv_w = run.nv["power_w"].to_numpy()
    cc_t = run.cc["elapsed_s"].to_numpy(); cc_w = run.cc["power_w"].to_numpy()
    nv_s, nv_e = _active_bounds(nv_t, nv_w)
    cc_s, cc_e = _active_bounds(cc_t, cc_w)
    active_start = max(nv_s, cc_s); active_end = min(nv_e, cc_e)
    mask   = (nv_t >= active_start) & (nv_t <= active_end)
    t_grid = nv_t[mask]; nv_w2 = nv_w[mask]
    cc_interp = np.interp(t_grid, cc_t, cc_w)
    diff = cc_interp - nv_w2

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t_grid, cc_interp, color="steelblue", lw=1.5, label="CodeCarbon (total system)")
    ax.plot(t_grid, nv_w2, color="darkorange", lw=1.5, label="nvidia-smi (GPU)")
    ax.plot(t_grid, diff, color="seagreen", lw=1.5, ls="--", label="CC − nvsmi (CPU + RAM)")
    ax.set_xlabel("Elapsed time (s)"); ax.set_ylabel("Power (W)")
    ax.set_title("Lag-Llama — total vs GPU vs CPU/RAM (active window)")
    ax.grid(True, alpha=0.3); ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_chronos_amortization(sweep_meta: List[dict], out_path: Path, dpi: int) -> None:
    if not sweep_meta:
        return
    ns      = np.array([m["n_origins"] for m in sweep_meta], dtype=float)
    cc_j    = np.array([m["cc"]["energy_kwh"] * 3.6e6 for m in sweep_meta])
    nv_j    = np.array([m["nvidia_smi"]["energy_kwh"] * 3.6e6 for m in sweep_meta])
    order   = np.argsort(ns)
    ns      = ns[order]; cc_j = cc_j[order]; nv_j = nv_j[order]
    cc_per_n = cc_j / ns; nv_per_n = nv_j / ns

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    axL.loglog(ns, cc_per_n, "o-", color="steelblue",  label="CodeCarbon", lw=1.5, ms=6)
    axL.loglog(ns, nv_per_n, "s-", color="darkorange", label="nvidia-smi", lw=1.5, ms=6)
    axL.set_xlabel("n_origins"); axL.set_ylabel("Energy per origin (J)")
    axL.set_title("Per-origin cost vs workload")
    axL.grid(True, alpha=0.3, which="both"); axL.legend(fontsize=9)
    axR.plot(ns, cc_j, "o-", color="steelblue",  label="CodeCarbon", lw=1.5, ms=6)
    axR.plot(ns, nv_j, "s-", color="darkorange", label="nvidia-smi", lw=1.5, ms=6)
    axR.set_xlabel("n_origins"); axR.set_ylabel("Total energy (J)")
    axR.set_title("Total energy vs workload")
    axR.grid(True, alpha=0.3); axR.legend(fontsize=9)
    fig.suptitle("Chronos-Bolt mini — fixed overhead vs marginal cost",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if len(ns) >= 2:
        slope_cc, int_cc = np.polyfit(ns, cc_j, 1)
        slope_nv, int_nv = np.polyfit(ns, nv_j, 1)
        print(f"\nChronos amortization fit (J = fixed + marginal·n):")
        print(f"  CC: fixed={int_cc:.2f} J, marginal={slope_cc:.4f} J/origin")
        print(f"  NV: fixed={int_nv:.2f} J, marginal={slope_nv:.4f} J/origin")


def cmd_analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_map = _discover_runs(Path(args.results_dir), args.tag)
    if not runs_map:
        raise SystemExit(f"No runs found for tags {args.tag}")

    ordered_models = [m for m in _MODEL_ORDER if m in runs_map] + \
                     sorted(m for m in runs_map if m not in _MODEL_ORDER)
    runs = [_load_run(runs_map[m]) for m in ordered_models]

    xv = _cross_validate_energy(runs)
    xv_csv = out_dir / "energy_cross_validation.csv"
    xv.to_csv(xv_csv, index=False, float_format="%.4f")
    print("=== Energy cross-validation (Wh) ===")
    print(xv.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\nSaved → {xv_csv}")

    _plot_per_origin_curves(runs, out_dir / "power_curves_per_origin.png", args.dpi)
    _plot_energy_per_origin_bar(runs, out_dir / "energy_per_origin_bar.png", args.dpi)
    print(f"Per-origin curves → {out_dir / 'power_curves_per_origin.png'}")

    ll_run = next((r for r in runs if r.model == "lag_llama"), None)
    if ll_run is not None:
        _plot_lag_llama_batches(ll_run, args.lag_llama_batch_size,
                                out_dir / "lag_llama_batch_boundaries.png", args.dpi)
        _plot_lag_llama_cpu_ram(ll_run, out_dir / "lag_llama_cpu_ram_contribution.png", args.dpi)

    sweep: List[dict] = []
    for run_dir in sorted(Path(args.results_dir).glob(
            f"powercurve_chronos_bolt_mini_*_{args.sweep_tag_prefix}*")):
        summary = run_dir / "summary.json"
        if summary.exists():
            try:
                sweep.append(json.loads(summary.read_text()))
            except json.JSONDecodeError:
                pass
    if sweep:
        _plot_chronos_amortization(sweep, out_dir / "chronos_amortization.png", args.dpi)
        print(f"Chronos sweep → {out_dir / 'chronos_amortization.png'}")


# --- composite ---

# Print geometry for the composite figure.
#
# NeurIPS \textwidth is ~5.5 in and this figure is placed at 0.94 textwidth, so it
# prints ~5.17 in wide. The figure is authored at that width (rather than authored
# wide and shrunk) so that 1 pt on the canvas is ~1 pt on the page: font sizes below
# are therefore the sizes the reader actually sees. Authoring at the old 13 in and
# scaling to fit would have required ~28 pt type to read as 11 pt in print.
_COMPOSITE_FIGSIZE = (5.17, 5.9)   # 0.94 x 5.5 in NeurIPS textwidth
_COMPOSITE_RC = {
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize":  9,
    "lines.linewidth":  2.0,
}


def cmd_composite(args: argparse.Namespace) -> None:
    runs = _discover_runs(Path(args.results_dir), args.tag)
    if not runs:
        raise SystemExit(f"No runs found for tags {args.tag} in {args.results_dir}")

    ordered = [m for m in _MODEL_ORDER if m in runs] + \
              sorted(m for m in runs if m not in _MODEL_ORDER)

    # Scoped so the other subcommands keep matplotlib's defaults.
    with plt.rc_context(_COMPOSITE_RC):
        _composite_figure(args, runs, ordered)


def _composite_figure(args, runs, ordered) -> None:
    fig, (ax_nv, ax_cc) = plt.subplots(2, 1, figsize=_COMPOSITE_FIGSIZE, sharex=True)
    max_t = 0.0

    for model in ordered:
        run_dir = runs[model]
        cc_csv  = run_dir / "codecarbon" / "emissions.csv"
        nv_csv  = run_dir / "nvsmi_power_log.csv"
        meta    = json.loads((run_dir / "summary.json").read_text())
        color   = _MODEL_COLOR.get(model)
        n       = meta.get("n_origins", "?")
        dur     = meta.get("duration_s", 0.0)
        label   = f"{model}  (n={n}, win={dur:.0f}s)"

        cc = reextract_cc(cc_csv)
        nv = pd.read_csv(nv_csv) if nv_csv.exists() else pd.DataFrame()

        if not nv.empty:
            ax_nv.plot(nv["elapsed_s"], nv["power_w"], color=color, lw=2.0,
                       label=label, alpha=0.9)
            max_t = max(max_t, float(nv["elapsed_s"].max()))
        if not cc.empty:
            ax_cc.plot(cc["elapsed_s"], cc["power_w"], color=color, lw=2.0,
                       marker="o", ms=4.5, label=label, alpha=0.9)
            max_t = max(max_t, float(cc["elapsed_s"].max()))

    ax_nv.set_title("nvidia-smi GPU power")
    ax_nv.set_ylabel("Power (W)")
    ax_nv.grid(True, alpha=0.3)

    ax_cc.set_title("CodeCarbon total system power")
    ax_cc.set_xlabel("Elapsed time (s)")
    ax_cc.set_ylabel("Power (W)")
    ax_cc.grid(True, alpha=0.3)
    ax_cc.set_xlim(left=-1, right=max_t + 1)

    # Both panels plot the same series, so one shared legend replaces the two
    # per-axes ones. At print width an in-axes legend of ~10 entries covers the
    # curves it is labelling, so it moves below the panels instead.
    handles, labels = ax_nv.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_cc.get_legend_handles_labels()
    # Two columns, not three: the labels carry "(n=..., win=...s)" and three
    # columns of that overflow the print width and get clipped.
    # Anchored inside the canvas, with tight_layout reserving the strip below the
    # axes for it. Kept in-canvas deliberately: the figure is saved without
    # bbox_inches="tight" so the file is exactly _COMPOSITE_FIGSIZE, which is what
    # makes the point sizes above survive into print unscaled.
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncols=2, frameon=False, handlelength=1.6, columnspacing=1.1)

    fig.suptitle("Power vs time during walk-forward inference", fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for path in (out_path, out_path.with_suffix(".pdf")):
        fig.savefig(path, dpi=args.dpi)
        written.append(str(path))
    plt.close(fig)
    print(f"Composite plot → {', '.join(written)}  (dpi={args.dpi})")
    print(f"Models included ({len(ordered)}): {', '.join(ordered)}")


# --- CLI dispatcher ---

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # measure
    pm = sub.add_parser("measure", help="Measure per-second power curves for one model")
    pm.add_argument("--model", required=True)
    pm.add_argument("--data", default="data/processed/elia_load_15min.csv")
    pm.add_argument("--timestamp_col", default="datetime")
    pm.add_argument("--target_col", default="totalload")
    pm.add_argument("--freq", default="15min")
    pm.add_argument("--delimiter", default=None)
    pm.add_argument("--test_years", type=float, default=1.0)
    pm.add_argument("--stride", type=int, default=96)
    pm.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 96])
    pm.add_argument("--n_origins", type=int, default=200)
    pm.add_argument("--naive_s", type=int, default=96)
    pm.add_argument("--arima_order", type=int, nargs=3, default=[2, 1, 2])
    pm.add_argument("--sarima_order", type=int, nargs=3, default=[2, 1, 2])
    pm.add_argument("--sarima_seasonal_order", type=int, nargs=4, default=[1, 1, 1, 96])
    pm.add_argument("--maxiter", type=int, default=50)
    pm.add_argument("--max_train_obs", type=int, default=4000)
    pm.add_argument("--context_length", type=int, default=None)
    pm.add_argument("--num_samples", type=int, default=None)
    pm.add_argument("--device", default=None)
    pm.add_argument("--batch_size", type=int, default=16)
    pm.add_argument("--patch_size", default="auto")
    pm.add_argument("--point_forecast_mode", default="median",
                    choices=["median", "mean"])
    pm.add_argument("--cc_poll_interval_s", type=float, default=1.0)
    pm.add_argument("--nvsmi_interval_ms", type=int, default=100)
    pm.add_argument("--gpu_index", default="0")
    pm.add_argument("--no_cc", action="store_true")
    pm.add_argument("--no_nvsmi", action="store_true")
    pm.add_argument("--results_dir", default="results")
    pm.add_argument("--tag", default=None)
    pm.add_argument("--no_plot", action="store_true")
    pm.add_argument("--dpi", type=int, default=120)

    # analyze
    pa = sub.add_parser("analyze", help="Post-process power-curve runs")
    pa.add_argument("--tag", nargs="+", default=["cmp2", "dense"])
    pa.add_argument("--sweep_tag_prefix", default="sweep_n")
    pa.add_argument("--results_dir", default="results")
    pa.add_argument("--out_dir", default="figures")
    pa.add_argument("--dpi", type=int, default=150)
    pa.add_argument("--lag_llama_batch_size", type=int, default=4)

    # composite
    pc = sub.add_parser("composite", help="Compose per-model power curves into one figure")
    pc.add_argument("--tag", nargs="+", default=["cmp2"])
    pc.add_argument("--results_dir", default="results")
    pc.add_argument("--out", default="figures/power_curves_composite.png")
    # 300 dpi for print; a sibling .pdf is written alongside the .png.
    pc.add_argument("--dpi", type=int, default=300)

    args = p.parse_args()
    {"measure": cmd_measure, "analyze": cmd_analyze, "composite": cmd_composite}[args.command](args)


if __name__ == "__main__":
    main()
