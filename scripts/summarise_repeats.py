#!/usr/bin/env python3
"""Summarise repeated walk-forward runs produced by scripts/run_repeats.sh.

Reads every  <root>/<model>/rep*/{wf,tsfm}_*/metrics.csv  and reports, per model,
the mean / std / min / max across repeats of:

  * energy  — mean(CodeCarbon, CarbonTracker) eval energy, in uWh per origin
              (kWh / n_predictions * 1e9).  This is the same mean(CC, CT)
              convention used by plot_thesis.py (`mean_kwh`).
  * accuracy — MAE at h=96, in MW.

A second energy column reports fit+eval rather than eval-only.  The two differ
only for the classical models: foundation models are zero-shot, so their fit
energy is logged as exactly 0.0 (run_tsfm_benchmark.py), while lgbm/linear/arima
carry a real training cost.  Read across families with that asymmetry in mind.

Also reported per model, when the columns are present in metrics.csv:

  * eval_duration_s — wall clock of the eval phase, so sub-minute rows are
              visible next to their energy.
  * bmc_*   — whole-chassis BMC measurement (--bmc runs): gross and
              incremental kWh, baseline mean / std W, run mean / peak W,
              run window length in s.
  * rapl_*  — socket CPU measurement via Intel RAPL package-0 + package-1
              (--rapl runs), same statistics as BMC.
  * ct_cpu_avg_w — CarbonTracker's own average CPU watts for the eval
              phase (0.0 = RAPL permission failure, empty = it reported None,
              which happens for sub-second workloads).
  * CodeCarbon measured / modelled split, from the last *_eval row of
              codecarbon/emissions.csv:
                cc_measured_kwh = gpu_energy + cpu_energy  if cc_cpu_mode == RAPL
                                = gpu_energy               otherwise (cpu_load)
                cc_modelled_kwh = energy_consumed - cc_measured_kwh
                measured_fraction = cc_measured_kwh / energy_consumed
              RAM is always a model in CodeCarbon; CPU is a measurement only
              when it read RAPL.  cc_cpu_mode is written by the runners since
              Sept 2026; rows without it (the May runs) were produced without
              RAPL access and are treated as cpu_load, so old and new rows sit
              in one table.

std is the sample standard deviation (ddof=1); it is undefined for n < 2 and
printed as "n/a".

Usage:
    python scripts/summarise_repeats.py                      # newest results/repeats_*
    python scripts/summarise_repeats.py --root results/repeats_20260907_120000
    python scripts/summarise_repeats.py --horizon 4 --csv out.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise repeat runs from scripts/run_repeats.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default=None,
                   help="Repeats root directory. Defaults to the newest results/repeats_*.")
    p.add_argument("--results_dir", default="results",
                   help="Where to look for repeats_* when --root is omitted.")
    p.add_argument("--horizon", type=int, default=96,
                   help="Horizon (in steps) whose MAE is summarised.")
    p.add_argument("--csv", default=None,
                   help="Optional path to also write the per-repeat rows as CSV.")
    return p.parse_args()


def _newest_root(results_dir: str) -> str:
    roots = sorted(glob.glob(os.path.join(results_dir, "repeats_*")))
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        raise SystemExit(
            f"No repeats_* directory found under {results_dir}/. "
            f"Run scripts/run_repeats.sh first, or pass --root."
        )
    return max(roots, key=os.path.getmtime)


def _cc_split(rep_dir: str, cc_cpu_mode: Optional[str]) -> Dict[str, float]:
    """Split CodeCarbon's eval energy into measured and modelled kWh.

    From the last *_eval row of codecarbon/emissions.csv:
      measured = gpu_energy + cpu_energy   if cc_cpu_mode == "RAPL"
               = gpu_energy                otherwise (cpu_load: CPU is a TDP model)
      modelled = energy_consumed - measured   (RAM is always modelled)
      fraction = measured / energy_consumed
    """
    nan = float("nan")
    out = {"cc_measured_kwh": nan, "cc_modelled_kwh": nan, "measured_fraction": nan}
    is_rapl = isinstance(cc_cpu_mode, str) and cc_cpu_mode.strip().upper() == "RAPL"
    for em in sorted(glob.glob(os.path.join(rep_dir, "*", "codecarbon", "emissions.csv"))):
        try:
            e = pd.read_csv(em)
        except Exception:
            continue
        need = ["gpu_energy", "energy_consumed"] + (["cpu_energy"] if is_rapl else [])
        if any(c not in e.columns for c in need):
            continue
        if "project_name" in e.columns:
            ev = e[e["project_name"].astype(str).str.endswith("_eval")]
            if not ev.empty:
                e = ev
        last = e.iloc[-1]
        total = float(last["energy_consumed"])
        measured = float(last["gpu_energy"])
        if is_rapl:
            measured += float(last["cpu_energy"])
        out["cc_measured_kwh"] = measured
        out["cc_modelled_kwh"] = total - measured
        if total > 0:
            out["measured_fraction"] = measured / total
        return out
    return out


def _mean_of_backends(row: pd.Series, cc_col: str, ct_col: str) -> float:
    """mean(CC, CT), skipping backends that are missing or non-positive."""
    vals = []
    for col in (cc_col, ct_col):
        v = row.get(col, np.nan)
        if pd.notna(v) and float(v) > 0:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def collect(root: str, horizon: int) -> pd.DataFrame:
    """One row per repeat."""
    rows: List[dict] = []

    for meta_path in sorted(glob.glob(os.path.join(root, "*", "rep*", "repeat_meta.json"))):
        rep_dir = os.path.dirname(meta_path)
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [warn] unreadable {meta_path}: {exc}")
            continue

        if meta.get("exit_code", 0) != 0:
            print(f"  [warn] {meta.get('model')} rep{meta.get('repeat_index')}: "
                  f"exit_code={meta['exit_code']} — skipped")
            continue

        metrics = sorted(glob.glob(os.path.join(rep_dir, "*", "metrics.csv")))
        if not metrics:
            print(f"  [warn] no metrics.csv under {rep_dir} — skipped")
            continue

        df = pd.read_csv(metrics[0])
        df["model"] = df["model"].str.lower().str.strip()
        model = meta.get("model", df["model"].iloc[0])

        # A run dir can hold several models' rows; keep only this rep's model.
        sub = df[(df["model"] == str(model).lower().strip())
                 & (df["horizon_steps"] == horizon)]
        if sub.empty:
            print(f"  [warn] {model} rep{meta.get('repeat_index')}: "
                  f"no h={horizon} row for model '{model}' — skipped")
            continue
        r = sub.iloc[0]

        n_pred = float(r.get("n_predictions", 0) or 0)
        if n_pred <= 0:
            print(f"  [warn] {model} rep{meta.get('repeat_index')}: n_predictions={n_pred} — skipped")
            continue

        eval_kwh = _mean_of_backends(r, "eval_cc_energy_kwh", "eval_ct_energy_kwh")
        fit_kwh = _mean_of_backends(r, "fit_cc_energy_kwh", "fit_ct_energy_kwh")
        # Zero-shot FMs log fit energy as exactly 0.0, which _mean_of_backends
        # reads as "absent"; a genuinely absent fit is also 0 for this purpose.
        fit_kwh = 0.0 if np.isnan(fit_kwh) else fit_kwh

        def _f(col):
            v = r.get(col)
            try:
                return float(v) if pd.notna(v) else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        cc_cpu_mode = r.get("cc_cpu_mode")
        cc_cpu_mode = str(cc_cpu_mode) if pd.notna(cc_cpu_mode) else None
        cc_split = _cc_split(rep_dir, cc_cpu_mode)

        rows.append({
            "model":            model,
            "repeat_index":     meta.get("repeat_index"),
            "run_id":           meta.get("run_id", ""),
            "kind":             meta.get("kind", ""),
            "phase":            meta.get("phase", "repeat"),
            "n_predictions":    int(n_pred),
            "wall_clock_s":     meta.get("wall_clock_s"),
            "eval_duration_s":  _f("eval_duration_s"),
            "eval_uwh_per_origin":  eval_kwh / n_pred * 1e9,
            "total_uwh_per_origin": (fit_kwh + eval_kwh) / n_pred * 1e9,
            f"MAE_h{horizon}":  float(r["MAE"]),
            # Whole-chassis BMC measurement (present only for --bmc runs).
            "bmc_gross":        _f("bmc_gross_kwh"),
            "bmc_incremental":  _f("bmc_incremental_kwh"),
            "bmc_baseline_w":   _f("bmc_baseline_w"),
            "bmc_baseline_std_w": _f("bmc_baseline_std_w"),
            "bmc_mean_w":       _f("bmc_mean_w"),
            "bmc_peak_w":       _f("bmc_peak_w"),
            "bmc_run_window_s": _f("bmc_run_window_s"),
            # Socket CPU measurement via RAPL (present only for --rapl runs).
            "rapl_gross":       _f("rapl_gross_kwh"),
            "rapl_incremental": _f("rapl_incremental_kwh"),
            "rapl_baseline_w":  _f("rapl_baseline_w"),
            "rapl_baseline_std_w": _f("rapl_baseline_std_w"),
            "rapl_mean_w":      _f("rapl_mean_w"),
            "rapl_peak_w":      _f("rapl_peak_w"),
            "rapl_run_window_s": _f("rapl_run_window_s"),
            # What each tracker's CPU term actually is.
            "cc_cpu_mode":      cc_cpu_mode if cc_cpu_mode is not None else "",
            "ct_cpu_avg_w":     _f("ct_cpu_avg_w"),
            # CodeCarbon measured vs modelled kWh, and their ratio.
            "cc_measured_kwh":  cc_split["cc_measured_kwh"],
            "cc_modelled_kwh":  cc_split["cc_modelled_kwh"],
            "measured_fraction": cc_split["measured_fraction"],
        })

    return pd.DataFrame(rows)


def _stats(values: np.ndarray) -> Dict[str, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n":    len(v),
        "mean": float(np.mean(v)),
        "std":  float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
        "min":  float(np.min(v)),
        "max":  float(np.max(v)),
    }


def _fmt(v: float, width: int, prec: int) -> str:
    return f"{'n/a':>{width}}" if not np.isfinite(v) else f"{v:>{width}.{prec}f}"


def summarise(per_repeat: pd.DataFrame, horizon: int) -> None:
    mae_col = f"MAE_h{horizon}"

    for title, col, unit, prec in [
        (f"Energy — mean(CC, CT) EVAL only", "eval_uwh_per_origin", "uWh / origin", 4),
        (f"Energy — mean(CC, CT) FIT + EVAL", "total_uwh_per_origin", "uWh / origin", 4),
        (f"Accuracy — MAE at h={horizon}", mae_col, "MW", 3),
        ("Eval duration", "eval_duration_s", "s", 2),
        ("BMC — gross chassis energy", "bmc_gross", "kWh", 6),
        ("BMC — incremental over baseline", "bmc_incremental", "kWh", 6),
        ("BMC — baseline power", "bmc_baseline_w", "W", 2),
        ("BMC — baseline power std", "bmc_baseline_std_w", "W", 2),
        ("BMC — run mean power", "bmc_mean_w", "W", 2),
        ("BMC — run peak power", "bmc_peak_w", "W", 2),
        ("BMC — run window", "bmc_run_window_s", "s", 2),
        ("RAPL — gross socket energy (pkg0+pkg1)", "rapl_gross", "kWh", 6),
        ("RAPL — incremental over baseline", "rapl_incremental", "kWh", 6),
        ("RAPL — baseline power", "rapl_baseline_w", "W", 2),
        ("RAPL — baseline power std", "rapl_baseline_std_w", "W", 2),
        ("RAPL — run mean power", "rapl_mean_w", "W", 2),
        ("RAPL — run peak power", "rapl_peak_w", "W", 2),
        ("RAPL — run window", "rapl_run_window_s", "s", 2),
        ("CT CPU avg W (CarbonTracker's own CPU reading)", "ct_cpu_avg_w", "W", 2),
        ("CodeCarbon — measured kWh (gpu [+cpu if RAPL])", "cc_measured_kwh", "kWh", 8),
        ("CodeCarbon — modelled kWh (ram [+cpu if cpu_load])", "cc_modelled_kwh", "kWh", 8),
        ("CodeCarbon — measured fraction (measured/total)", "measured_fraction", "0-1", 4),
    ]:
        if col in per_repeat.columns and per_repeat[col].notna().sum() == 0:
            continue  # nothing measured for this column — skip the empty table
        print()
        print("=" * 86)
        print(f"{title}   [{unit}]")
        print("=" * 86)
        print(f"{'model':<20} {'n':>3} {'mean':>14} {'std':>14} {'min':>14} {'max':>14}")
        print("-" * 86)
        for model, grp in per_repeat.groupby("model", sort=False):
            s = _stats(grp[col].to_numpy(dtype=float))
            print(f"{model:<20} {s['n']:>3} "
                  f"{_fmt(s['mean'], 14, prec)} {_fmt(s['std'], 14, prec)} "
                  f"{_fmt(s['min'], 14, prec)} {_fmt(s['max'], 14, prec)}")
        print("=" * 86)

    # Relative spread is what the variance question is actually about.
    print()
    print("Relative spread  (std / mean, %)")
    print("-" * 86)
    print(f"{'model':<20} {'eval energy':>14} {'fit+eval energy':>18} {f'MAE h={horizon}':>14} "
          f"{'BMC gross':>12} {'BMC increm.':>12} {'RAPL gross':>12} {'RAPL increm.':>12}")
    print("-" * 86)
    for model, grp in per_repeat.groupby("model", sort=False):
        cells = []
        for col in ("eval_uwh_per_origin", "total_uwh_per_origin", mae_col,
                    "bmc_gross", "bmc_incremental", "rapl_gross", "rapl_incremental"):
            if col not in grp.columns:
                cells.append(np.nan)
                continue
            s = _stats(grp[col].to_numpy(dtype=float))
            rel = (s["std"] / s["mean"] * 100.0) if (np.isfinite(s["std"]) and s["mean"]) else np.nan
            cells.append(rel)
        print(f"{model:<20} {_fmt(cells[0], 14, 2)} {_fmt(cells[1], 18, 2)} "
              f"{_fmt(cells[2], 14, 2)} {_fmt(cells[3], 12, 2)} {_fmt(cells[4], 12, 2)} "
              f"{_fmt(cells[5], 12, 2)} {_fmt(cells[6], 12, 2)}")
    print("-" * 86)


def main() -> None:
    args = parse_args()
    root = args.root or _newest_root(args.results_dir)
    print(f"\nRepeats root: {root}")

    per_repeat = collect(root, args.horizon)
    if per_repeat.empty:
        raise SystemExit("No usable repeats found — nothing to summarise.")

    n_models = per_repeat["model"].nunique()
    print(f"Collected {len(per_repeat)} repeat(s) across {n_models} model(s), "
          f"horizon h={args.horizon}")

    n_pred = sorted(per_repeat["n_predictions"].unique())
    if len(n_pred) > 1:
        print(f"  [warn] n_predictions differs across repeats: {n_pred} — "
              f"per-origin figures are not directly comparable")

    print()
    print("Per-repeat rows")
    print("-" * 86)
    print(per_repeat.to_string(index=False))

    summarise(per_repeat, args.horizon)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        per_repeat.to_csv(args.csv, index=False)
        print(f"\nPer-repeat rows written to {args.csv}")


if __name__ == "__main__":
    main()
