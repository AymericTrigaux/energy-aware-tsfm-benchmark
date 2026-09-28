"""Walk-forward benchmark for zero-shot foundation model forecasters.

Same rolling-origin protocol as run_benchmark.py, targeting the foundation
models in src/models_tsfm.py (Chronos, TimesFM, Lag-Llama, Moirai).

All models are used zero-shot. Energy brackets the inference loop only.
Results use the same CSV schema as run_benchmark.py for direct concatenation.

Run with .venv_fm: source .venv_fm/bin/activate

Usage:
    python run_tsfm_benchmark.py --models chronos_mini timesfm_200m lag_llama
    python run_tsfm_benchmark.py --models chronos_mini --energy_tool none
    python run_tsfm_benchmark.py --models chronos_mini --stride 192
    python run_tsfm_benchmark.py --models timesfm_200m --no_plots
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import (
    load_timeseries,
    clean_timeseries,
    split_last_n_years,
    append_run_csv,
    freq_to_minutes,
)
from src.metrics import (
    mae,
    mape,
    rmse,
    EnergyMeter,
    CarbonTrackerMeter,
    NvidiaSmiMeter,
    BmcPowerMeter,
    RaplPowerMeter,
    EnergyResult,
    sample_host_conditions,
)
from src.models_tsfm import TSFM_MODELS


# --- CLI ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward benchmark for zero-shot foundation model forecasters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data", default="data/processed/elia_load_15min.csv",
                   help="Path to the Elia load CSV/Parquet file.")
    p.add_argument("--timestamp_col", default="datetime")
    p.add_argument("--target_col", default="totalload")
    p.add_argument("--freq", default="15min")
    p.add_argument("--delimiter", default=None)

    p.add_argument("--test_years", type=float, default=1.0,
                   help="Length of the test window in calendar years.")
    p.add_argument(
        "--stride", type=int, default=96,
        help=(
            "Steps between consecutive forecast origins.  "
            "Default 96 = 1 day (same as run_benchmark.py).  "
            "Use a larger value (e.g. 192 or 384) to reduce wall-clock time "
            "when evaluating heavy models on CPU."
        ),
    )
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 96],
                   help="Forecast horizons in steps (1=15min, 4=1h, 96=1day).")

    p.add_argument(
        "--models", nargs="+",
        default=list(TSFM_MODELS.keys()),
        choices=list(TSFM_MODELS.keys()),
        help="Foundation models to benchmark.",
    )

    p.add_argument("--context_length", type=int, default=None,
                   help="Context window length passed to each foundation model. "
                        "If omitted, each class uses its own default "
                        "(512 for most, 2048 for TimesFM-2.0 / Chronos-Bolt).")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Number of trajectory samples for probabilistic models "
                        "(Chronos, Lag-Llama, Moirai). If omitted, each class "
                        "uses its own default (Chronos: 50, Moirai/Lag-Llama: 100). "
                        "Median is used as the point forecast.")
    p.add_argument("--device", default=None,
                   help="PyTorch device (cpu, mps, cuda). Auto-detected if omitted.")
    p.add_argument("--n_origins", type=int, default=None,
                   help="Cap the number of walk-forward origins evaluated. "
                        "Useful for smoke tests (e.g. --n_origins 5). "
                        "Default None = evaluate the full test window.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Origins per forward pass (batched walk-forward). "
                        "Higher = faster but more GPU memory; 1 disables "
                        "batching and falls back to single-origin predict().")
    p.add_argument("--patch_size", default="auto",
                   help="Patch size for Moirai variants — 'auto' (default), "
                        "or one of 8/16/32/64/128. Ignored by non-Moirai models.")
    p.add_argument("--point_forecast_mode", default="median",
                   choices=["median", "mean"],
                   help="Point-forecast aggregation for TimesFM. "
                        "Ignored by non-TimesFM models.")

    p.add_argument(
        "--energy_tool", default="both",
        choices=["codecarbon", "carbontracker", "both", "nvidia_smi", "all", "none"],
        help=(
            "Energy measurement backend.  "
            "codecarbon: CodeCarbon only.  "
            "carbontracker: CarbonTracker only.  "
            "both: CodeCarbon + CarbonTracker (default, recommended on macOS).  "
            "nvidia_smi: GPU-only via nvidia-smi polling (Linux+NVIDIA).  "
            "all: CodeCarbon + CarbonTracker + nvidia-smi (recommended on ELECTA).  "
            "none: disable all energy tracking."
        ),
    )
    p.add_argument(
        "--gpu_index", default="0",
        help=(
            "GPU index(es) passed to nvidia-smi --id (default '0'). "
            "Use '0,1' for multi-GPU jobs; power from all specified GPUs is summed. "
            "Only relevant when --energy_tool includes nvidia_smi."
        ),
    )

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

    p.add_argument("--results_dir", default=os.environ.get("BENCH_RESULTS", "results"),
                   help="Root directory for output files.")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip forecast plot generation.")
    p.add_argument("--dpi", type=int, default=100)

    return p.parse_args()


# --- Walk-forward evaluation ---

def walk_forward(
    predictor,
    y_all: pd.Series,
    test_start_idx: int,
    horizons: List[int],
    stride: int,
    n_origins: Optional[int] = None,
    batch_size: int = 1,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Evaluate a predictor over the test window using rolling origins."""
    n = len(y_all)
    max_h = max(horizons)

    y_true_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    y_pred_by_h: Dict[int, List[float]] = {h: [] for h in horizons}

    all_origins = list(range(test_start_idx, n - max_h, stride))
    if n_origins is not None:
        all_origins = all_origins[:n_origins]
    n_total = len(all_origins)

    use_batch = batch_size > 1 and hasattr(predictor, "predict_batch")

    def _collect(origin_idx: int, preds: Dict[int, float]) -> None:
        for h in horizons:
            target_idx = origin_idx + h
            if target_idx < n and not np.isnan(preds.get(h, float("nan"))):
                y_true_by_h[h].append(float(y_all.iloc[target_idx]))
                y_pred_by_h[h].append(float(preds[h]))

    if use_batch:
        n_batches = (n_total + batch_size - 1) // batch_size
        print(f"    walk-forward: {n_total} origins, stride={stride}, "
              f"batch={batch_size} ({n_batches} batched calls)"
              + (f" (capped at {n_origins})" if n_origins is not None else ""))
        for bi, chunk_start in enumerate(range(0, n_total, batch_size)):
            if bi % 10 == 0:
                done = min(chunk_start + batch_size, n_total)
                print(f"    batch {bi + 1}/{n_batches} (origin {done}/{n_total})",
                      flush=True)
            chunk = all_origins[chunk_start: chunk_start + batch_size]
            batch_results = predictor.predict_batch(chunk)
            for origin_idx in chunk:
                _collect(origin_idx, batch_results[origin_idx])
    else:
        print(f"    walk-forward: {n_total} origins, stride={stride} (per-origin)"
              + (f" (capped at {n_origins})" if n_origins is not None else ""))
        for i, origin_idx in enumerate(all_origins):
            if i % 50 == 0:
                print(f"    origin {i + 1}/{n_total} (idx={origin_idx})", flush=True)
            _collect(origin_idx, predictor.predict(origin_idx))

    return (
        {h: np.array(v) for h, v in y_true_by_h.items()},
        {h: np.array(v) for h, v in y_pred_by_h.items()},
    )


# --- Metrics ---

def compute_metrics(
    y_true_by_h: Dict[int, np.ndarray],
    y_pred_by_h: Dict[int, np.ndarray],
) -> Dict[int, Dict[str, float]]:
    """Compute MAE, RMSE, and MAPE for each horizon."""
    metrics = {}
    for h in y_true_by_h:
        yt = y_true_by_h[h]
        yp = y_pred_by_h[h]
        if len(yt) == 0:
            metrics[h] = {"MAE": float("nan"), "RMSE": float("nan"),
                          "MAPE": float("nan"), "n": 0}
        else:
            metrics[h] = {
                "MAE":  mae(yt, yp),
                "RMSE": rmse(yt, yp),
                "MAPE": mape(yt, yp),
                "n":    len(yt),
            }
    return metrics


# --- Result logging ---

def log_metrics(
    run_id: str,
    label: str,
    metrics_by_h: Dict[int, Dict[str, float]],
    fit_energy: EnergyResult,
    eval_energy: EnergyResult,
    run_dir: str,
    step_minutes: int = 15,
    bmc_summary: Optional[dict] = None,
    host_start: Optional[dict] = None,
    host_end: Optional[dict] = None,
    rapl_summary: Optional[dict] = None,
    cc_cpu_mode: Optional[str] = None,
    ct_cpu_avg_w: Optional[float] = None,
) -> None:
    """Append one row per horizon to the run's metrics.csv."""
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

    def _r(val, decimals: int = 8):
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
            "fit_cc_energy_kwh":    _r(fit_energy.energy_kwh),
            "fit_cc_emissions_kg":  _r(fit_energy.emissions_kg),
            "fit_duration_s":       _r(fit_energy.duration_s, 2),
            "eval_cc_energy_kwh":   _r(eval_energy.energy_kwh),
            "eval_cc_emissions_kg": _r(eval_energy.emissions_kg),
            "eval_duration_s":      _r(eval_energy.duration_s, 2),
            "fit_ct_energy_kwh":    _r(fit_energy.ct_energy_kwh),
            "fit_ct_emissions_kg":  _r(fit_energy.ct_emissions_kg),
            "eval_ct_energy_kwh":   _r(eval_energy.ct_energy_kwh),
            "eval_ct_emissions_kg": _r(eval_energy.ct_emissions_kg),
            "fit_nv_energy_kwh":    _r(fit_energy.nv_energy_kwh),
            "fit_nv_emissions_kg":  _r(fit_energy.nv_emissions_kg),
            "eval_nv_energy_kwh":   _r(eval_energy.nv_energy_kwh),
            "eval_nv_emissions_kg": _r(eval_energy.nv_emissions_kg),
            "energy_tool":          eval_energy.tool,
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
    """Save a 2-week time-series plot and scatter plot for each horizon."""
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

        fig, ax = plt.subplots(figsize=(14, 4))
        nshow = min(1344, n2)  # 14 days × 96 steps/day
        ax.plot(t_axis[:nshow], yt[:nshow], label="Actual",
                color="steelblue", lw=1)
        ax.plot(t_axis[:nshow], yp[:nshow], label=label,
                color="tomato", lw=1, alpha=0.85)
        ax.set_title(
            f"{label} · h={h} steps ({h * 15} min ahead) — first 14 days of test"
        )
        ax.set_ylabel("Load (MW)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"{label}_h{h}_week.png"), dpi=dpi)
        plt.close(fig)

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


# --- Predictions collector ---

def build_predictions_df(
    model_name: str,
    y_all: pd.Series,
    y_true_by_h: Dict[int, np.ndarray],
    y_pred_by_h: Dict[int, np.ndarray],
    test_start_idx: int,
    stride: int,
) -> pd.DataFrame:
    """Convert walk-forward arrays to a long-format DataFrame."""
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


# --- Per-model runner ---

def _make_meters(
    tag: str,
    cc_dir: str,
    ct_dir: str,
    nv_dir: str,
    cc_enabled: bool,
    ct_enabled: bool,
    nv_enabled: bool,
    gpu_index: str = "0",
) -> Tuple[EnergyMeter, CarbonTrackerMeter, NvidiaSmiMeter]:
    cc = EnergyMeter(tag, cc_dir, enabled=cc_enabled)
    ct = CarbonTrackerMeter(tag, ct_dir, enabled=ct_enabled)
    nv = NvidiaSmiMeter(tag, nv_dir, gpu_index=gpu_index, enabled=nv_enabled)
    # Build the tracker objects now so start() is cheap and their
    # construction never lands inside a chassis-meter window.
    cc.prepare(); ct.prepare()
    return cc, ct, nv


def _combined(
    cc_meter: EnergyMeter,
    ct_meter: CarbonTrackerMeter,
    nv_meter: NvidiaSmiMeter,
) -> EnergyResult:
    cc_res = cc_meter.stop()
    ct_res = ct_meter.stop()
    nv_res = nv_meter.stop()
    return cc_res.with_ct(ct_res).with_nv(nv_res)


def run_model(
    model_name: str,
    y_all: pd.Series,
    y_train: pd.Series,
    test_start_idx: int,
    horizons: List[int],
    stride: int,
    args: argparse.Namespace,
    run_id: str,
    run_dir: str,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Instantiate, load, walk-forward evaluate, log, and plot one TSFM model."""
    host_start = sample_host_conditions()
    cc_enabled = args.energy_tool in ("codecarbon", "both", "all")
    ct_enabled = args.energy_tool in ("carbontracker", "both", "all")
    nv_enabled = args.energy_tool in ("nvidia_smi", "all")
    cc_dir = os.path.join(run_dir, "codecarbon")
    ct_dir = os.path.join(run_dir, "carbontracker")
    nv_dir = os.path.join(run_dir, "nvidia_smi")
    gpu_index = getattr(args, "gpu_index", "0")

    cls = TSFM_MODELS[model_name]
    # Inspect __init__ (walking the MRO) so new variants automatically pick up
    # the matching CLI args without needing per-model branches here.
    import inspect
    accepted: set = set()
    for klass in cls.__mro__:
        if klass is object:
            continue
        try:
            accepted.update(inspect.signature(klass.__init__).parameters.keys())
        except (TypeError, ValueError):
            pass

    common_kwargs: dict = {"horizons": horizons}
    if "device" in accepted and args.device is not None:
        common_kwargs["device"] = args.device
    if "context_length" in accepted and args.context_length is not None:
        common_kwargs["context_length"] = args.context_length
    if "num_samples" in accepted and args.num_samples is not None:
        common_kwargs["num_samples"] = args.num_samples
    if "batch_size" in accepted:
        common_kwargs["batch_size"] = args.batch_size
    if "patch_size" in accepted:
        common_kwargs["patch_size"] = args.patch_size
    if "point_forecast_mode" in accepted:
        common_kwargs["point_forecast_mode"] = args.point_forecast_mode

    predictor = cls(**common_kwargs)

    t_fit_start = time.time()
    predictor.fit(y_train)
    fit_duration_s = time.time() - t_fit_start

    # Zero-shot: no gradient updates, so fit energy is always 0.0
    fit_energy = EnergyResult(
        energy_kwh=0.0,
        emissions_kg=0.0,
        duration_s=round(fit_duration_s, 2),
        tool="zero_shot",
        ct_energy_kwh=0.0,
        ct_emissions_kg=0.0,
        nv_energy_kwh=0.0,
        nv_emissions_kg=0.0,
    )
    print(f"  [{model_name}] model ready in {fit_duration_s:.1f}s "
          f"(fit_kwh=0.0 — zero-shot)")

    predictor.set_series(y_all, test_start_idx)

    eval_tag = f"{run_id}_{model_name}_eval"
    cc_eval, ct_eval, nv_eval = _make_meters(
        eval_tag, cc_dir, ct_dir, nv_dir,
        cc_enabled, ct_enabled, nv_enabled, gpu_index,
    )
    # RAPL samples continuously and is told when phases switch; BMC blocks on
    # its own baseline windows. Starting RAPL first and stopping it last makes
    # the two meters share identical "before" / run / "after" windows. Without
    # --bmc the runner sleeps the RAPL baseline itself so the windows still
    # exist.
    bmc_on = bool(getattr(args, "bmc", False))
    rapl_on = bool(getattr(args, "rapl", False))
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
    rapl_eval.start()                       # RAPL phase "before"
    bmc_eval.start()                        # blocks bmc_baseline_s if --bmc
    if rapl_on and not bmc_on:
        time.sleep(args.rapl_baseline_s)
    rapl_eval.begin_run()                   # RAPL phase "run"
    cc_eval.start()
    ct_eval.start()
    nv_eval.start()
    y_true_by_h, y_pred_by_h = walk_forward(
        predictor, y_all, test_start_idx, horizons, stride,
        n_origins=getattr(args, "n_origins", None),
        batch_size=getattr(args, "batch_size", 1),
    )
    rapl_eval.end_run()                     # RAPL phase "cooldown" then "after"; run = walk-forward + tracker starts only
    eval_energy = _combined(cc_eval, ct_eval, nv_eval)
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

    metrics_by_h = compute_metrics(y_true_by_h, y_pred_by_h)
    for h, m in sorted(metrics_by_h.items()):
        print(
            f"    h={h:3d} ({h * 15:4d}min)  MAE={m['MAE']:.1f}  "
            f"RMSE={m['RMSE']:.1f}  MAPE={m['MAPE']:.2f}%  n={m['n']}"
        )

    if not args.no_plots:
        plot_forecast_sample(
            y_all, y_true_by_h, y_pred_by_h,
            test_start_idx, stride,
            label=model_name, run_dir=run_dir, dpi=args.dpi,
        )

    host_end = sample_host_conditions()
    log_metrics(run_id, model_name, metrics_by_h, fit_energy, eval_energy, run_dir,
                step_minutes=getattr(args, "_step_minutes", 15),
                bmc_summary=bmc_summary,
                host_start=host_start, host_end=host_end,
                rapl_summary=rapl_summary,
                cc_cpu_mode=cc_cpu_mode, ct_cpu_avg_w=ct_cpu_avg_w)

    return metrics_by_h, y_true_by_h, y_pred_by_h


# --- Summary table ---

def print_comparison_table(
    all_metrics: Dict[str, Dict[int, Dict[str, float]]],
    horizons: List[int],
) -> None:
    print("\n" + "=" * 70)
    print("TSFM WALK-FORWARD BENCHMARK RESULTS")
    print("=" * 70)
    print(
        f"{'Model':<14} {'h':>4} {'h(min)':>7}  "
        f"{'MAE':>8}  {'RMSE':>8}  {'MAPE%':>7}  {'n':>6}"
    )
    print("-" * 70)

    for model_name, metrics_by_h in all_metrics.items():
        for h in sorted(horizons):
            m = metrics_by_h.get(h, {})
            print(
                f"{model_name:<14} {h:>4} {h * 15:>7}  "
                f"{m.get('MAE', float('nan')):>8.1f}  "
                f"{m.get('RMSE', float('nan')):>8.1f}  "
                f"{m.get('MAPE', float('nan')):>7.2f}  "
                f"{m.get('n', 0):>6}"
            )
        print()

    print("=" * 70)


# --- Main ---

def main() -> None:
    args = parse_args()
    args._step_minutes = freq_to_minutes(args.freq)
    run_id = "tsfm_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(args.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun folder: {run_dir}")

    print(f"\n[{run_id}] Loading data from {args.data} …")
    df = load_timeseries(
        args.data,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        freq=args.freq,
        delimiter=args.delimiter,
    )
    df = clean_timeseries(df, impute_method="interpolate")
    y_all = df["y"]
    print(f"  Loaded {len(y_all):,} obs from {y_all.index[0]} to {y_all.index[-1]}")

    df_train, df_test = split_last_n_years(df, args.test_years)
    y_train = df_train["y"]
    y_test  = df_test["y"]
    test_start_idx = len(y_train)

    print(f"  Train: {len(y_train):,} obs  ({y_train.index[0]} → {y_train.index[-1]})")
    print(f"  Test : {len(y_test):,} obs  ({y_test.index[0]} → {y_test.index[-1]})")
    print(f"  Horizons: {args.horizons}  Stride: {args.stride}  Models: {args.models}")

    all_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    all_pred_frames: List[pd.DataFrame] = []

    for model_name in args.models:
        print(f"\n{'─' * 60}\nRunning model: {model_name.upper()}\n{'─' * 60}")
        try:
            metrics_by_h, y_true_by_h, y_pred_by_h = run_model(
                model_name=model_name,
                y_all=y_all,
                y_train=y_train,
                test_start_idx=test_start_idx,
                horizons=args.horizons,
                stride=args.stride,
                args=args,
                run_id=run_id,
                run_dir=run_dir,
            )
            all_metrics[model_name] = metrics_by_h
            all_pred_frames.append(
                build_predictions_df(
                    model_name, y_all, y_true_by_h, y_pred_by_h,
                    test_start_idx, args.stride,
                )
            )
        except Exception as exc:
            print(f"  [{model_name}] ERROR: {exc}")
            traceback.print_exc()

    if all_pred_frames:
        predictions_path = os.path.join(run_dir, "predictions.parquet")
        pd.concat(all_pred_frames, ignore_index=True).to_parquet(
            predictions_path, index=False
        )
        print(f"\nPredictions saved → {predictions_path}")

    if all_metrics:
        print_comparison_table(all_metrics, args.horizons)

    print(f"\nDone. Results in {run_dir}/")


if __name__ == "__main__":
    main()
