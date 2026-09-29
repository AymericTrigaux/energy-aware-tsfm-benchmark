#!/usr/bin/env python3
"""Paper figures for v13, from the September 2026 measurement campaign.

Writes PDF (vector) and PNG (300 dpi) to figures/paper_v13/:

  pareto_all_horizons        MAE against chassis marginal energy, h = 1, 4, 96
  pretraining_amortization   lifetime energy per forecast against forecasts made
  fig_boundary               chassis, socket and tracker energy per model (appendix)
  fig_method_traces          one repeat's power traces and the cool-down curve (appendix)

Inputs are the campaign roots read through scripts/summarise_repeats.collect, so
every number is the one in the campaign table. Page geometry: NeurIPS single
column, 5.5 in text width, Times body font; figures are authored at their
printed size and saved without tight cropping.

Usage:
    python scripts/plot_paper_v13.py            # all four figures
    python scripts/plot_paper_v13.py --only a b
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import summarise_repeats as sr  # noqa: E402
import recompute_pretraining as rp  # noqa: E402

OUT_DIR = os.path.join(ROOT, "figures", "paper_v13")

# ---------------------------------------------------------------- colour and labels
# One hue per family, Okabe and Ito set (colour-vision safe). Every figure draws
# from this dict; members of a family are told apart by direct labels.
FAMILY_COLOR = {
    "Naive":     "#7F7F7F",
    "ARIMA":     "#D55E00",
    "Ridge":     "#0072B2",
    "LightGBM":  "#009E73",
    "Chronos":   "#CC79A7",
    "TimesFM":   "#56B4E9",
    "Moirai":    "#E69F00",
    "Lag-Llama": "#000000",
}
MODEL_FAMILY = {
    "naive": "Naive", "arima": "ARIMA", "linear": "Ridge", "lgbm": "LightGBM",
    "chronos_mini": "Chronos", "chronos_large": "Chronos",
    "chronos_bolt_mini": "Chronos", "chronos_bolt_base": "Chronos",
    "timesfm_200m": "TimesFM", "timesfm_500m": "TimesFM",
    "moirai_small": "Moirai", "moirai_base": "Moirai", "moirai_large": "Moirai",
    "moirai2_small": "Moirai", "lag_llama": "Lag-Llama",
}
DISP = {
    "naive": "Seasonal Naive", "arima": "ARIMA", "linear": "Ridge", "lgbm": "LightGBM",
    "chronos_mini": "Chronos-T5 mini", "chronos_large": "Chronos-T5 large",
    "chronos_bolt_mini": "Chronos-Bolt mini", "chronos_bolt_base": "Chronos-Bolt base",
    "timesfm_200m": "TimesFM 200M", "timesfm_500m": "TimesFM 500M",
    "moirai_small": "Moirai-1.0 small", "moirai_base": "Moirai-1.0 base",
    "moirai_large": "Moirai-1.0 large", "moirai2_small": "Moirai-2.0 small",
    "lag_llama": "Lag-Llama",
}
SHORT = {
    "naive": "Naive", "arima": "ARIMA", "linear": "Ridge", "lgbm": "LightGBM",
    "chronos_mini": "T5 mini", "chronos_large": "T5 large",
    "chronos_bolt_mini": "Bolt mini", "chronos_bolt_base": "Bolt base",
    "timesfm_200m": "TimesFM 200M", "timesfm_500m": "TimesFM 500M",
    "moirai_small": "Moirai-1 small", "moirai_base": "Moirai-1 base",
    "moirai_large": "Moirai-1 large", "moirai2_small": "Moirai-2 small",
    "lag_llama": "Lag-Llama",
}
CLASSICAL = {"naive", "arima", "linear", "lgbm"}
MEASURE_COLOR = {"chassis": "#000000", "socket": "#0072B2", "tracker": "#E69F00"}
INK, INK2, GRID = "#1A1A1A", "#4D4D4D", "#D9D9D9"

# ---------------------------------------------------------------- data sources
PLAN = [
    ("results/repeats_20260925_092952", "both",
     ["naive", "linear", "lgbm", "chronos_bolt_mini", "moirai2_small"]),
    ("results/repeats_20260927_132530", "both", ["arima"]),
    ("results/repeats_20260927_154055", "before",
     ["chronos_bolt_base", "moirai_small", "moirai_base", "moirai_large",
      "timesfm_200m", "chronos_mini", "timesfm_500m", "chronos_large", "lag_llama"]),
]
LAG_LLAMA_CHASSIS_RANGE = (235_746.0, 270_912.0)   # own before-window; explicit 393 W baseline
TRACE_REPEAT = "results/repeats_20260925_092952/chronos_bolt_mini/rep03/tsfm_20260925_104937"
COOLDOWN_RUN = "results/repeats_20260928_112337/lag_llama/rep01/tsfm_20260928_112338"

RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def load_table() -> pd.DataFrame:
    rows = []
    for root, mode, models in PLAN:
        d = {h: _quiet(sr.collect, os.path.join(ROOT, root), h, mode) for h in (1, 4, 96)}
        for m in models:
            g = {h: d[h][d[h].model == m] for h in (1, 4, 96)}
            g96 = g[96]
            n = len(g96)
            sd = (lambda s: float(s.std(ddof=1)) if n > 1 else float("nan"))
            rows.append(dict(
                model=m, n=n,
                mae_h1=g[1].MAE_h1.mean(), mae_h4=g[4].MAE_h4.mean(), mae_h96=g96.MAE_h96.mean(),
                chassis=g96.bmc_incremental_uwh_per_origin.mean(),
                chassis_std=sd(g96.bmc_incremental_uwh_per_origin),
                socket=g96.rapl_incremental_uwh_per_origin.mean(),
                socket_std=sd(g96.rapl_incremental_uwh_per_origin),
                tracker=g96.eval_uwh_per_origin.mean(), tracker_std=sd(g96.eval_uwh_per_origin),
            ))
    T = pd.DataFrame(rows).set_index("model")
    # Seasonal Naive sits below the chassis meter's resolution; plot it at its socket value.
    T["x_plot"] = T["chassis"]
    T.loc["naive", "x_plot"] = T.loc["naive", "socket"]
    return T


# ---------------------------------------------------------------- helpers
def no_dashes(fig):
    """Refuse to save a figure whose text contains an em or en dash."""
    bad = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
           if "—" in t.get_text() or "–" in t.get_text()]
    if bad:
        raise ValueError(f"em/en dash in figure text: {bad}")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    no_dashes(fig)
    png = os.path.join(OUT_DIR, name + ".png")
    pdf = os.path.join(OUT_DIR, name + ".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  saved {os.path.relpath(png, ROOT)} and .pdf")


def repel_labels(fig, ax, texts, points, max_shift_pt=9.0, iters=300):
    """Nudge overlapping labels apart vertically (small steps) and draw a hairline
    leader when a label ends more than 3 pt from its anchor."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    shifts = np.zeros(len(texts))
    for _ in range(iters):
        moved = False
        boxes = [t.get_window_extent(r).expanded(1.04, 1.12) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    d = 0.8 * fig.dpi / 72  # 0.8 pt
                    up, dn = (i, j) if boxes[i].y0 >= boxes[j].y0 else (j, i)
                    for k, s in ((up, +d), (dn, -d)):
                        if abs(shifts[k] + s) <= max_shift_pt * fig.dpi / 72:
                            shifts[k] += s
                            x, y = texts[k].get_position()
                            texts[k].set_position((x, y + s / (fig.dpi / 72)))
                            moved = True
                    boxes = [t.get_window_extent(r).expanded(1.04, 1.12) for t in texts]
        if not moved:
            break
    for t, (px, py) in zip(texts, points):
        x, y = t.get_position()
        if abs(y) > 3.0:
            ax.annotate("", xy=(px, py), xycoords="data", xytext=(x, y), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=2.5))


def place_labels(fig, ax, anchors, names, fontsize=5.6, max_leader_pt=14.0):
    """Greedy label placement: try offsets around each point, nearest first, and
    keep the first that overlaps neither an existing label nor another marker.
    A hairline leader is drawn when the label sits more than 5 pt away."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ppi = fig.dpi / 72.0
    # marker boxes in display space
    pt_boxes = []
    for (x, y) in anchors:
        px, py = ax.transData.transform((x, y))
        pt_boxes.append(matplotlib.transforms.Bbox.from_bounds(px - 3 * ppi, py - 3 * ppi, 6 * ppi, 6 * ppi))
    cands = []
    for dist in (4, 7, 10, 14, 19, 25, 32, 40):
        angs = ((20, -35, 160, -145, 70, -80, 110, -110, 0, 180, 45, -45, 135, -135) if dist <= 14
                else (70, 110, 55, 125, 90, 40, 140, 20, 160))
        for ang in angs:
            a = np.deg2rad(ang)
            dx, dy = dist * np.cos(a), dist * np.sin(a)
            ha = "left" if dx > 0.5 else "right" if dx < -0.5 else "center"
            va = "bottom" if dy > 0.5 else "top" if dy < -0.5 else "center"
            cands.append((dx, dy, ha, va, dist))
    placed = []
    axbox = ax.get_window_extent(r)
    order = sorted(range(len(anchors)), key=lambda i: (np.log10(max(anchors[i][0], 1e-3)), anchors[i][1]))
    for i in order:
        x, y = anchors[i]
        chosen = None
        for dx, dy, ha, va, dist in cands:
            t = ax.annotate(names[i], (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, ha=ha, va=va, color=INK, zorder=6,
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
            bb = t.get_window_extent(r).expanded(1.06, 1.15)
            ok = (bb.x0 >= axbox.x0 - 2 and bb.x1 <= axbox.x1 + 2 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1 + 1
                  and not any(bb.overlaps(q) for q in placed)
                  and not any(bb.overlaps(pb) for j, pb in enumerate(pt_boxes) if j != i))
            if ok:
                chosen = (t, bb, dist, dx, dy)
                break
            t.remove()
        if chosen is None:   # fall back to the nearest position, accepting overlap
            dx, dy, ha, va, dist = cands[0]
            t = ax.annotate(names[i], (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, ha=ha, va=va, color=INK, zorder=6)
            chosen = (t, t.get_window_extent(r), dist, dx, dy)
        t, bb, dist, dx, dy = chosen
        placed.append(bb)
        if dist > 5:
            ax.annotate("", xy=(x, y), xycoords="data", xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.35, color=INK2, shrinkA=0, shrinkB=2.5), zorder=5)


def family_legend(fig, families, extra=(), loc="lower center", ncol=5, y=0.0):
    handles = [Line2D([0], [0], marker="s", ls="none", ms=5, mfc=FAMILY_COLOR[f], mec="none", label=f)
               for f in families]
    handles += list(extra)
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=(0.5, y), ncol=ncol,
               handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2)


# ---------------------------------------------------------------- figure a
def fig_pareto(T: pd.DataFrame):
    panels = [("mae_h1", "h = 1 (15 min)"), ("mae_h4", "h = 4 (1 h)"), ("mae_h96", "h = 96 (24 h)")]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04)
    lo, hi = LAG_LLAMA_CHASSIS_RANGE
    for ax, (col, title) in zip(axes, panels):
        pts = {m: (T.loc[m, "x_plot"], T.loc[m, col]) for m in T.index}
        front = [m for m in pts if not any(
            (pts[o][0] <= pts[m][0] and pts[o][1] <= pts[m][1] and pts[o] != pts[m]) for o in pts if o != m)]
        texts, anchors = [], []
        for m in T.index:
            x, y = pts[m]
            c = FAMILY_COLOR[MODEL_FAMILY[m]]
            mk = "o" if m in CLASSICAL else "^"
            if m == "lag_llama":
                ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="butt", zorder=3)
                ax.plot([lo], [y], marker=mk, ms=4.5, mfc=c, mec="white", mew=0.6, ls="none", zorder=4)
                x = lo
            elif m == "naive":
                ax.plot([x], [y], marker=mk, ms=4.5, mfc="white", mec=c, mew=1.0, ls="none", zorder=4)
            else:
                if T.loc[m, "n"] > 1 and np.isfinite(T.loc[m, "chassis_std"]):
                    ax.errorbar([x], [y], xerr=[[T.loc[m, "chassis_std"]], [T.loc[m, "chassis_std"]]],
                                fmt="none", ecolor=c, elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
                ax.plot([x], [y], marker=mk, ms=4.5, mfc=c, mec="white", mew=0.6, ls="none", zorder=4)
            if m in front:
                ax.plot([x], [y], marker="o", ms=9, mfc="none", mec=INK, mew=0.7, ls="none", zorder=5)
            texts.append(SHORT[m]); anchors.append((x, y))
        ax.set_xscale("log")
        ax.set_xlim(0.03, 4e6)
        ymin, ymax = T[col].min(), T[col].max()
        ax.set_ylim(ymin - 0.10 * (ymax - ymin), ymax + 0.16 * (ymax - ymin))
        ax.set_title(title, loc="left", pad=3)
        ax.grid(True, which="major"); ax.grid(False, which="minor")
        ax.tick_params(length=2.5)
        place_labels(fig, ax, anchors, texts)
    axes[0].set_ylabel("MAE (MW)")
    axes[1].set_xlabel("Chassis marginal energy per forecast origin (uWh, log scale)")
    axes[0].annotate("open marker: Naive at its socket value,\nchassis marginal below meter resolution",
                     xy=(0.97, 0.88), xycoords="axes fraction", fontsize=5.0, color=INK2,
                     ha="right", va="top")
    extra = [Line2D([0], [0], marker="o", ls="none", ms=4.5, mfc="#BBBBBB", mec="white", label="classical"),
             Line2D([0], [0], marker="^", ls="none", ms=4.5, mfc="#BBBBBB", mec="white", label="foundation"),
             Line2D([0], [0], marker="o", ls="none", ms=8, mfc="none", mec=INK, mew=0.7, label="Pareto frontier")]
    family_legend(fig, list(FAMILY_COLOR), extra=extra, ncol=6, y=-0.01)
    fig.set_constrained_layout_pads(rect=(0, 0.12, 1, 1)) if False else None
    fig.get_layout_engine().set(rect=(0, 0.11, 1, 1))
    save(fig, "pareto_all_horizons")


# ---------------------------------------------------------------- figure b
def rule_table():
    """Pretraining energy per foundation model: old constant, Rule A, Rule B.

    Rule A: parameter-linear, anchored on the Chronos-T5 large disclosure
            (294.7 kWh for 708,963,328 parameters, 0.4157 kWh per M parameters).
    Rule B: parameter-linear, anchored on the nearest other row(s) of the same
            family (mean kWh per M parameters of those rows); undefined when the
            family has no other row.
    """
    P, E = rp.PARAMS, rp.CURRENT_KWH
    rate = {m: E[m] / (P[m] / 1e6) for m in E}
    rate_a = rate["chronos_large"]
    fam_rows = {"Chronos": ["chronos_mini", "chronos_large"],
                "Moirai": ["moirai_small", "moirai_base", "moirai_large"]}
    anchors_b = {
        "chronos_bolt_mini": ["chronos_large"], "chronos_bolt_base": ["chronos_large"],
        "chronos_mini": ["chronos_large"], "chronos_large": ["chronos_mini"],
        "moirai2_small": ["moirai_base", "moirai_large"], "moirai_small": ["moirai_base", "moirai_large"],
        "moirai_base": ["moirai_large"], "moirai_large": ["moirai_base"], "lag_llama": [],
    }
    out = {}
    for m in E:
        a = P[m] / 1e6 * rate_a
        b = P[m] / 1e6 * np.mean([rate[x] for x in anchors_b[m]]) if anchors_b[m] else float("nan")
        out[m] = (E[m], a, b)
    return out


def fig_amortization(T: pd.DataFrame, rules):
    N = np.logspace(3, 12, 400)
    marg = {m: T.loc[m, "x_plot"] * 1e-9 for m in T.index}   # kWh per origin
    scen = [("A", 35_040), ("B", 3_504_000), ("C", 35_040_000)]
    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 1))
    curves = {}
    for m in ["naive", "arima", "linear", "lgbm"]:
        c = FAMILY_COLOR[MODEL_FAMILY[m]]
        ax.plot(N, np.full_like(N, marg[m]), color=c, lw=(1.8 if m == "lgbm" else 1.1),
                alpha=(1 if m == "lgbm" else 0.9), zorder=(5 if m == "lgbm" else 3))
        curves[m] = marg[m]
    fm = [m for m in T.index if m not in CLASSICAL and m in rules]   # TimesFM has no estimate: omitted
    for m in fm:
        c = FAMILY_COLOR[MODEL_FAMILY[m]]
        old, a, b = rules[m]
        if m in rp.FLAGGED:          # the three rows where the accounting rules disagree
            pre, pre_hi = min(a, b), max(a, b)
        else:                        # everything else keeps the constants used so far
            pre = pre_hi = old
        if m == "lag_llama":
            lo, hi = (v * 1e-9 for v in LAG_LLAMA_CHASSIS_RANGE)
            ax.fill_between(N, pre / N + lo, pre / N + hi, color=c, alpha=0.22, lw=0, zorder=2)
            ax.plot(N, pre / N + lo, color=c, lw=1.0, ls="--", zorder=3)
            curves[m] = pre / N[-1] + hi
        elif abs(pre_hi - pre) > 1e-9:
            ax.fill_between(N, pre / N + marg[m], pre_hi / N + marg[m], color=c, alpha=0.22, lw=0, zorder=2)
            ax.plot(N, pre / N + marg[m], color=c, lw=1.0, ls="--", zorder=3)
            ax.plot(N, pre_hi / N + marg[m], color=c, lw=1.0, ls="--", zorder=3)
            curves[m] = pre_hi / N[-1] + marg[m]
        else:
            ax.plot(N, pre / N + marg[m], color=c, lw=1.0, ls="--", zorder=3)
            curves[m] = pre / N[-1] + marg[m]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e3, 1e12)
    ax.set_ylim(3e-11, 3.0)
    for sc, per_year in scen:
        for mult, tag in ((1, "1 yr"), (30, "30 yr")):
            nval = per_year * mult
            ax.axvline(nval, color=INK2, ls=":", lw=0.6, alpha=0.7, zorder=1)
            ax.annotate(f"{sc} {tag}", (nval, 2e-10), xytext=(1.5, 0), textcoords="offset points",
                        rotation=90, va="bottom", ha="left", fontsize=5.4, color=INK2,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    # direct labels at the right edge, nudged apart
    texts, anchors = [], []
    for m, yv in sorted(curves.items(), key=lambda kv: kv[1]):
        t = ax.annotate(DISP[m], (N[-1], yv), xytext=(2.5, 0), textcoords="offset points",
                        fontsize=5.6, va="center", color=INK, annotation_clip=False)
        texts.append(t); anchors.append((N[-1], yv))
    ax.set_xlabel("Forecasts made over the model's lifetime, N (log scale)")
    ax.set_ylabel("Energy per forecast (kWh, log scale)")
    ax.grid(True, which="major"); ax.grid(False, which="minor")
    ax.tick_params(length=2.5)
    fig.canvas.draw()
    repel_labels(fig, ax, texts, anchors, max_shift_pt=9.0)
    extra = [Line2D([0], [0], color=INK2, lw=1.1, label="classical: chassis marginal"),
             Line2D([0], [0], color=INK2, lw=1.0, ls="--", label="foundation: pretraining / N + marginal"),
             Patch(facecolor="#BBBBBB", alpha=0.6, label="band: Rule A to Rule B, or baseline range")]
    fig.legend(handles=extra, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3,
               handletextpad=0.5, columnspacing=1.2)
    save(fig, "pretraining_amortization")


# ---------------------------------------------------------------- figure c
def fig_boundary(T: pd.DataFrame):
    order = T.sort_values("chassis").index.tolist()          # ascending by chassis marginal
    fig, ax = plt.subplots(figsize=(3.6, 5.0))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.985, bottom=0.165)
    h = 0.26
    ypos = np.arange(len(order))[::-1] * 1.0                   # smallest at the top
    lo, hi = LAG_LLAMA_CHASSIS_RANGE
    for i, m in enumerate(order):
        y = ypos[i]
        vals = [("chassis", T.loc[m, "chassis"], T.loc[m, "chassis_std"]),
                ("socket", T.loc[m, "socket"], T.loc[m, "socket_std"]),
                ("tracker", T.loc[m, "tracker"], T.loc[m, "tracker_std"])]
        for k, (key, v, s) in enumerate(vals):
            yy = y + (1 - k) * h
            if m == "lag_llama" and key == "chassis":
                ax.barh(yy, lo, height=h * 0.92, color=MEASURE_COLOR[key], lw=0)
                ax.barh(yy, hi - lo, left=lo, height=h * 0.92, facecolor="white",
                        edgecolor=MEASURE_COLOR[key], hatch="////", lw=0.5)
                continue
            v = max(v, 1e-3)
            ax.barh(yy, v, height=h * 0.92, color=MEASURE_COLOR[key], lw=0)
            if T.loc[m, "n"] > 1 and np.isfinite(s) and s > 0:
                ax.errorbar([v], [yy], xerr=[[min(s, v * 0.999)], [s]], fmt="none", ecolor="white",
                            elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([DISP[m] for m in order], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(8e-4, 8e5)
    ax.set_xlabel("Energy per forecast origin (uWh, log scale)")
    ax.grid(True, axis="x", which="major"); ax.grid(False, axis="y"); ax.grid(False, which="minor")
    ax.tick_params(length=2.5)
    ax.set_ylim(-0.65, len(order) - 0.35)
    handles = [Patch(facecolor=MEASURE_COLOR["chassis"], label="chassis marginal (BMC)"),
               Patch(facecolor=MEASURE_COLOR["socket"], label="socket marginal (RAPL)"),
               Patch(facecolor=MEASURE_COLOR["tracker"], label="tracker gross (CC/CT mean)"),
               Patch(facecolor="white", edgecolor=INK, hatch="////", label="Lag-Llama baseline range")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2,
               handlelength=1.2, handletextpad=0.5, columnspacing=0.9, fontsize=6)
    save(fig, "fig_boundary")


# ---------------------------------------------------------------- figure d
def fig_method_traces():
    d = os.path.join(ROOT, TRACE_REPEAT)
    b = pd.read_csv(os.path.join(d, "bmc_power.csv"))
    r = pd.read_csv(os.path.join(d, "rapl_power.csv"))
    nv = pd.read_csv(glob.glob(os.path.join(d, "nvidia_smi", "*gpu_power.csv"))[0])
    t0 = b[b.phase == "run"].timestamp_s.iloc[0]
    t_end = b[b.phase == "run"].timestamp_s.iloc[-1]
    nv["elapsed_s"] = nv.timestamp_s - t0
    xlim = (b.elapsed_s.min() - 1, b.elapsed_s.max() + 1)
    PH = {"baseline_before": ("before", "#EBEBEB"), "before": ("before", "#EBEBEB"),
          "run": ("run", "#FFF3D6"), "baseline_after": ("after", "#EBEBEB"), "after": ("after", "#EBEBEB")}

    fig, axes = plt.subplots(4, 1, figsize=(3.6, 5.4), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.03)

    def shade(ax, trace, run_end_rel):
        for ph, (lab, col) in PH.items():
            s = trace[trace.phase == ph]
            if s.empty:
                continue
            x0, x1 = s.elapsed_s.min(), s.elapsed_s.max()
            if ph == "run":
                x0, x1 = 0.0, run_end_rel
            ax.axvspan(x0, x1, color=col, lw=0, zorder=0)

    def panel(ax, trace, col, ylabel, color, baseline_phases, run_end_rel):
        shade(ax, trace, run_end_rel)
        base = trace[trace.phase.isin(baseline_phases)][col].mean()
        ax.plot(trace.elapsed_s, trace[col], color=color, lw=0.9, zorder=3)
        ax.axhline(base, color=INK2, ls="--", lw=0.6, zorder=2)
        run = trace[(trace.elapsed_s >= 0) & (trace.elapsed_s <= run_end_rel)]
        ax.fill_between(run.elapsed_s, base, run[col], where=run[col] > base, facecolor="none",
                        edgecolor=color, hatch="////", lw=0, zorder=1)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim); ax.tick_params(length=2.5)
        ax.grid(False)
        return base

    run_end_rel = t_end - t0
    b_base = panel(axes[0], b, "power_w", "Chassis (W)", MEASURE_COLOR["chassis"], ["baseline_before", "baseline_after"], run_end_rel)
    r_base = panel(axes[1], r, "sum_w", "Sockets (W)", MEASURE_COLOR["socket"], ["before", "after"], run_end_rel)
    ax = axes[2]
    for ph, (lab, col) in PH.items():
        s = b[b.phase == ph]
        if not s.empty:
            x0, x1 = (0.0, run_end_rel) if ph == "run" else (s.elapsed_s.min(), s.elapsed_s.max())
            ax.axvspan(x0, x1, color=col, lw=0, zorder=0)
    ax.plot(nv.elapsed_s, nv.power_w, color="#009E73", lw=0.7, zorder=3)
    ax.set_ylabel("GPU (W)"); ax.set_xlim(*xlim); ax.tick_params(length=2.5); ax.grid(False)
    ax.annotate("sampled during the run only", xy=(0.02, 0.88), xycoords="axes fraction", fontsize=5.6, color=INK2)
    axes[2].set_xlabel("Time from run start (s)")
    for ax, lab in zip(axes[:3], ("before", "run", "after")):
        pass
    axes[0].annotate("before", xy=(-58, 0.90), xycoords=("data", "axes fraction"), fontsize=5.6, color=INK2)
    axes[0].annotate("run", xy=(1.5, 0.90), xycoords=("data", "axes fraction"), fontsize=5.6, color=INK2)
    axes[0].annotate("after", xy=(run_end_rel + 4, 0.90), xycoords=("data", "axes fraction"), fontsize=5.6, color=INK2)
    axes[0].annotate("baseline mean", xy=(xlim[1] - 2, b_base), xytext=(-2, 2), textcoords="offset points",
                     ha="right", fontsize=5.6, color=INK2)
    axes[0].set_title("Chronos-Bolt mini, one repeat", loc="left", pad=3)

    # panel 4: Lag-Llama cool-down curve, time relative to the end of the run
    c = pd.read_csv(os.path.join(COOLDOWN_RUN, "bmc_power.csv")) if os.path.isabs(COOLDOWN_RUN) \
        else pd.read_csv(os.path.join(ROOT, COOLDOWN_RUN, "bmc_power.csv"))
    t_re = c[c.phase == "run"].timestamp_s.iloc[-1]
    c["rel"] = c.timestamp_s - t_re
    c = c[c.rel >= -120]
    ax = axes[3]
    for ph, col in (("run", "#FFF3D6"), ("cooldown", "#E4F0F8"), ("baseline_after", "#EBEBEB")):
        s = c[c.phase == ph]
        if not s.empty:
            ax.axvspan(s.rel.min(), s.rel.max(), color=col, lw=0, zorder=0)
    ax.plot(c.rel, c.power_w, color=MEASURE_COLOR["chassis"], lw=0.9, zorder=3)
    ax.set_xlim(-120, c.rel.max() + 1)
    ax.set_ylabel("Chassis (W)")
    ax.set_xlabel("Time from run end (s)")
    ax.tick_params(length=2.5); ax.grid(False)
    for lab, x in (("run", -115), ("cool-down", 3), ("after", 123)):
        ax.annotate(lab, xy=(x, 0.90), xycoords=("data", "axes fraction"), fontsize=5.6, color=INK2)
    ax.set_title("Lag-Llama, end of run and cool-down", loc="left", pad=3)
    save(fig, "fig_method_traces")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["a", "b", "c", "d"], default=["a", "b", "c", "d"])
    args = p.parse_args()
    plt.rcParams.update(RC)
    T = load_table()
    print("inputs (uWh per origin, MAE in MW):")
    print(T[["n", "mae_h1", "mae_h4", "mae_h96", "chassis", "chassis_std", "socket", "tracker", "x_plot"]].round(3).to_string())
    rules = rule_table()
    print("\nPretraining energy, kWh: old constant | Rule A | Rule B")
    print("  Rule A: parameter-linear, anchored on Chronos-T5 large (294.7 kWh, 708,963,328 parameters, 0.4157 kWh per M parameters).")
    print("  Rule B: parameter-linear, anchored on the nearest other row(s) of the same family; n/a when the family has no other row.")
    for m, (old, a, b) in rules.items():
        print(f"  {DISP[m]:20s} old={old:7.2f}  A={a:7.2f}  B={'n/a' if np.isnan(b) else f'{b:7.2f}'}")
    print("  TimesFM 200M / 500M: no estimate, omitted from the figure.")
    if "a" in args.only: fig_pareto(T)
    if "b" in args.only: fig_amortization(T, rules)
    if "c" in args.only: fig_boundary(T)
    if "d" in args.only: fig_method_traces()


if __name__ == "__main__":
    main()
