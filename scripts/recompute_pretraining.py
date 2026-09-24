#!/usr/bin/env python3
"""Recompute the three flagged pretraining-energy constants from the anchor formula.

READ-ONLY. This script prints numbers and writes nothing. The constants in
plot_thesis.py (PRETRAIN_KWH, ~line 1243) are deliberately left untouched.

The anchor formula, as specified:

    E [kWh] = t [h] x (8 x 400 W + 700 W) x 1.2 x 1e-3
            = t x 4.68 kWh/h

i.e. eight 400 W accelerators plus a 700 W host, PUE 1.2. Scaling to a different
model is then meant to follow parameter count and step count relative to the
Chronos-T5 large disclosure.

The three flagged rows are chronos_bolt_mini (80.0), chronos_bolt_base (130.0)
and moirai2_small (45.0) kWh.

WHAT THIS SCRIPT FOUND, stated up front so the tables are read correctly: the
five existing anchor rows are NOT mutually consistent under any single
(params x steps) rule. Inverting the formula per row shows the implied energy
per million parameters spanning a factor of ~35 across the anchors, and moving
in opposite directions within the two families. So "recompute from the same
formula" has no unique answer; the script reports several candidate rules and
lets the reader choose. See the CAVEATS section it prints last.

Usage:
    python scripts/recompute_pretraining.py
"""
from __future__ import annotations

from typing import Dict, Optional

# --- Anchor formula ---------------------------------------------------------

N_ACCEL = 8
ACCEL_W = 400.0
HOST_W = 700.0
PUE = 1.2

KWH_PER_HOUR = (N_ACCEL * ACCEL_W + HOST_W) * PUE * 1e-3  # 4.68 kWh per wall-clock hour


# --- Inputs -----------------------------------------------------------------
# Parameter counts are measured from the pinned checkpoints
# (see docs/checkpoint_revisions.md), not copied from any dict in this repo.
PARAMS: Dict[str, int] = {
    "chronos_mini":       20_456_192,
    "chronos_large":     708_963_328,
    "chronos_bolt_mini":  21_236_096,
    "chronos_bolt_base": 205_292_928,
    "moirai_small":       13_827_528,
    "moirai_base":        91_357_728,
    "moirai_large":      310_970_624,
    "moirai2_small":      11_387_208,
    "lag_llama":           2_449_299,
}

# Training step counts. These are EXTERNAL claims, not measurements.
# None = not published in a form comparable to the Chronos-T5 disclosure.
# Anything not None should be re-checked against the paper before being trusted;
# the whole step-scaling branch below depends on it.
STEPS: Dict[str, Optional[int]] = {
    "chronos_mini":      200_000,   # UNVERIFIED: Chronos paper, all T5 sizes
    "chronos_large":     200_000,   # UNVERIFIED: Chronos paper, all T5 sizes
    "chronos_bolt_mini": None,      # not published as steps (reported in tokens)
    "chronos_bolt_base": None,      # not published as steps (reported in tokens)
    "moirai_small":      None,
    "moirai_base":       None,
    "moirai_large":      None,
    "moirai2_small":     None,      # not published
    "lag_llama":         None,
}

# Current constants, plot_thesis.py:1243-1249.
CURRENT_KWH: Dict[str, float] = {
    "chronos_bolt_mini":  80.0,
    "chronos_bolt_base": 130.0,
    "chronos_mini":       37.4,
    "chronos_large":     294.7,
    "moirai_small":        3.0,
    "moirai_base":       190.0,
    "moirai_large":      650.0,
    "moirai2_small":      45.0,
    "lag_llama":          18.7,
}

FLAGGED = ["chronos_bolt_mini", "chronos_bolt_base", "moirai2_small"]
ANCHORS = ["chronos_mini", "chronos_large", "moirai_small",
           "moirai_base", "moirai_large", "lag_llama"]

ANCHOR_MODEL = "chronos_large"   # the disclosure named in the task
FAMILY_ANCHOR = {
    "chronos_bolt_mini": ["chronos_large"],
    "chronos_bolt_base": ["chronos_large"],
    # Moirai-1.0 base and large agree closely on kWh per M-param; small does not.
    "moirai2_small":     ["moirai_base", "moirai_large"],
}


def _rule(width: int = 96) -> None:
    print("-" * width)


def implied_hours(kwh: float) -> float:
    return kwh / KWH_PER_HOUR


def kwh_per_mparam(model: str) -> float:
    return CURRENT_KWH[model] / (PARAMS[model] / 1e6)


def table_anchors() -> None:
    print()
    print("=" * 96)
    print("TABLE 1 - Anchor rows inverted through the formula  (E = t x 4.68 kWh/h)")
    print("=" * 96)
    print(f"{'model':<20} {'params':>14} {'E current':>11} {'implied t':>11} "
          f"{'kWh / M-param':>15} {'steps':>10}")
    print(f"{'':<20} {'':>14} {'(kWh)':>11} {'(GPU-h)':>11} {'':>15} {'':>10}")
    _rule()
    for m in ANCHORS:
        steps = STEPS[m]
        print(f"{m:<20} {PARAMS[m]:>14,} {CURRENT_KWH[m]:>11.1f} "
              f"{implied_hours(CURRENT_KWH[m]):>11.2f} {kwh_per_mparam(m):>15.4f} "
              f"{(f'{steps:,}' if steps else 'unknown'):>10}")
    _rule()
    ratios = [kwh_per_mparam(m) for m in ANCHORS]
    print(f"kWh per M-param across anchors: min={min(ratios):.4f}  max={max(ratios):.4f}  "
          f"spread={max(ratios) / min(ratios):.1f}x")
    print("A single params-proportional rule would make this column constant. It is not.")


def table_rule_reproduction() -> None:
    """Apply the specified rule back to the anchors. If it cannot reproduce them,
    it cannot be the rule that produced them."""
    print()
    print("=" * 96)
    print(f"TABLE 2 - Does the specified rule reproduce the anchors?")
    print(f"          E_pred = E[{ANCHOR_MODEL}] x (P / P_anchor) x (S / S_anchor)")
    print("=" * 96)
    p_anchor = PARAMS[ANCHOR_MODEL]
    e_anchor = CURRENT_KWH[ANCHOR_MODEL]
    s_anchor = STEPS[ANCHOR_MODEL]
    print(f"{'model':<20} {'E current':>11} {'E predicted':>13} {'ratio':>9} {'step term':>12}")
    _rule()
    for m in ANCHORS:
        param_term = PARAMS[m] / p_anchor
        if STEPS[m] is not None and s_anchor:
            step_term = STEPS[m] / s_anchor
            step_note = f"{step_term:.3f}"
        else:
            step_term = 1.0
            step_note = "assumed 1"
        pred = e_anchor * param_term * step_term
        ratio = pred / CURRENT_KWH[m] if CURRENT_KWH[m] else float("nan")
        flag = "  <-- reproduced" if 0.95 <= ratio <= 1.05 else ""
        print(f"{m:<20} {CURRENT_KWH[m]:>11.1f} {pred:>13.2f} {ratio:>9.3f} "
              f"{step_note:>12}{flag}")
    _rule()
    print("Only the anchor itself is reproduced. The rule as specified does not")
    print("generate the other four rows, so it is not the rule they came from.")


def table_flagged() -> None:
    print()
    print("=" * 96)
    print("TABLE 3 - The three flagged rows: old value vs recomputed, per candidate rule")
    print("=" * 96)

    p_anchor = PARAMS[ANCHOR_MODEL]
    e_anchor = CURRENT_KWH[ANCHOR_MODEL]

    print(f"{'model':<20} {'params':>14} {'OLD kWh':>9} {'RULE A':>9} {'RULE B':>9} {'RULE C':>9}")
    _rule()
    for m in FLAGGED:
        # RULE A: params-linear off the Chronos-T5 large disclosure (as specified).
        a = e_anchor * (PARAMS[m] / p_anchor)

        # RULE B: params-linear off the nearest same-family anchor(s).
        fam = FAMILY_ANCHOR[m]
        rate = sum(kwh_per_mparam(f) for f in fam) / len(fam)
        b = rate * (PARAMS[m] / 1e6)

        # RULE C: params x steps. Needs a published step count for m.
        c = "n/a" if STEPS[m] is None else f"{e_anchor * (PARAMS[m] / p_anchor) * (STEPS[m] / STEPS[ANCHOR_MODEL]):.2f}"

        print(f"{m:<20} {PARAMS[m]:>14,} {CURRENT_KWH[m]:>9.1f} "
              f"{a:>9.2f} {b:>9.2f} {str(c):>9}")
    _rule()
    print("RULE A  params-linear, anchored on chronos_large (the named disclosure)")
    print("RULE B  params-linear, anchored on the nearest same-family row(s):")
    for m in FLAGGED:
        fam = FAMILY_ANCHOR[m]
        rate = sum(kwh_per_mparam(f) for f in fam) / len(fam)
        print(f"          {m:<20} <- {', '.join(fam)}  ({rate:.4f} kWh/M-param)")
    print("RULE C  params x steps, anchored on chronos_large - NOT COMPUTABLE for any")
    print("        flagged row: none publishes a step count comparable to Chronos-T5's.")

    print()
    print("Formula inputs behind every number above:")
    _rule()
    print(f"  accelerators      {N_ACCEL} x {ACCEL_W:.0f} W = {N_ACCEL * ACCEL_W:.0f} W")
    print(f"  host              {HOST_W:.0f} W")
    print(f"  PUE               {PUE}")
    print(f"  => energy rate    {KWH_PER_HOUR:.2f} kWh per wall-clock hour")
    print(f"  anchor model      {ANCHOR_MODEL}  "
          f"(P={PARAMS[ANCHOR_MODEL]:,}, E={CURRENT_KWH[ANCHOR_MODEL]} kWh, "
          f"S={STEPS[ANCHOR_MODEL]:,} steps [UNVERIFIED])")
    for m in FLAGGED:
        print(f"  {m:<18} P={PARAMS[m]:,}  S={STEPS[m] if STEPS[m] else 'unpublished'}  "
              f"implied t at OLD value = {implied_hours(CURRENT_KWH[m]):.2f} GPU-h")


def caveats() -> None:
    print()
    print("=" * 96)
    print("CAVEATS - read before adopting any number above")
    print("=" * 96)
    print("""
1. The five anchor rows are mutually inconsistent. Energy per million parameters
   ranges from 0.22 (moirai_small) to 7.64 (lag_llama) kWh/M-param, a 35x spread.
   Within families it disagrees in opposite directions: Chronos scales SUBLINEARLY
   with size (mini 1.83 vs large 0.42 kWh/M-param) while Moirai-1.0 base and large
   agree closely (2.08 vs 2.09) with small an order of magnitude below them (0.22).
   No single (params x steps) rule fits all five.

2. The rule named in the task does not reproduce the anchors. Scaling chronos_large
   (294.7 kWh) down to chronos_mini by parameter count gives 8.5 kWh; the table says
   37.4. Chronos-T5 mini and large were trained for the same number of steps, so the
   step term cannot close that gap.

3. Step counts are unpublished for all three flagged models. Chronos-Bolt reports
   training volume in tokens, not steps; Moirai-2.0 publishes neither. This is
   plausibly why these three rows were flagged in the first place. RULE C is
   therefore not computable, and the step term in RULE A is silently assumed to be 1.

4. The 200,000-step figure for Chronos-T5 is marked UNVERIFIED in this script. It is
   recalled from the Chronos paper, not read from a source in this repository. Check
   it before it carries any weight.

5. The provenance of the five anchor constants themselves is unknown. There is no
   derivation comment, script, or citation anywhere in the repository - only a note
   pointing at "thesis Table B.6 note c", a document not in this repo. These numbers
   cannot be independently reconstructed from what is here.

RECOMMENDATION: do not adopt a single recomputed value on the strength of this
script. Either recover the original derivation from thesis Table B.6, or drop the
amortisation curves for models whose pretraining energy is not estimable - which is
exactly what plot_thesis.py already does for TimesFM.
""".rstrip())


def main() -> None:
    print()
    print("#" * 96)
    print("# Pretraining-energy recomputation - READ-ONLY, no constants are modified")
    print("#" * 96)
    table_anchors()
    table_rule_reproduction()
    table_flagged()
    caveats()
    print()


if __name__ == "__main__":
    main()
