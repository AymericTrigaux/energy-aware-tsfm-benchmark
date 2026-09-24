"""Dataset preparation — Elia and ETTh1.

Usage:
    python prepare_data.py elia               # Prepare Elia 15-min electricity load
    python prepare_data.py etth1              # Download and prepare ETTh1 dataset
    python prepare_data.py etth1 --source github   # Force GitHub source
    python prepare_data.py all                # Prepare both datasets

Replaces the two separate prepare_dataset.py and prepare_etth1.py scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# --- Elia ---

RAW_ELIA  = Path("data/raw/Data Elia Load.csv")
OUT_ELIA  = Path("data/processed/elia_load_15min.csv")
FREQ_ELIA = "15min"


def prepare_elia() -> None:
    """Read raw Elia CSV, standardise columns, fill gaps, write processed CSV."""
    print(f"Reading {RAW_ELIA} …")
    df = pd.read_csv(RAW_ELIA, delimiter=";")

    timestamp_col = "Datetime"
    target_col    = "Total Load"

    df[timestamp_col] = (
        pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        .dt.tz_localize(None)
    )
    if df[timestamp_col].isna().any():
        bad = df[df[timestamp_col].isna()].head(5)
        raise ValueError(f"Unparseable timestamps found. Example:\n{bad}")

    df = (df[[timestamp_col, target_col]]
          .rename(columns={timestamp_col: "datetime", target_col: "totalload"})
          .sort_values("datetime")
          .set_index("datetime"))

    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq(FREQ_ELIA)
    df["totalload"] = df["totalload"].interpolate(method="time").ffill().bfill()

    OUT_ELIA.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(OUT_ELIA, index=False)
    print(f"Saved {OUT_ELIA} with {len(df):,} rows.")


# --- ETTh1 ---

OUT_ETTH1 = Path("data/processed/etth1.csv")
GITHUB_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)


def _fetch_huggingface() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("ett", "h1", split="test", trust_remote_code=True)
    rec = ds[-1]
    target = list(rec["target"])
    start  = pd.Timestamp(rec["start"])
    idx    = pd.date_range(start=start, periods=len(target), freq="h")
    return pd.DataFrame({"datetime": idx, "ot": target})


def _fetch_github() -> pd.DataFrame:
    df = pd.read_csv(GITHUB_URL)
    df = df.rename(columns={"date": "datetime", "OT": "ot"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime", "ot"]]


def prepare_etth1(source: str = "auto", out: Path = OUT_ETTH1) -> None:
    """Download ETTh1 and write to data/processed/etth1.csv."""
    if source == "github":
        print(f"Fetching ETTh1 from GitHub: {GITHUB_URL}")
        df = _fetch_github()
    elif source == "huggingface":
        print("Fetching ETTh1 from HuggingFace (dataset='ett', config='h1') …")
        df = _fetch_huggingface()
    else:
        try:
            print("Fetching ETTh1 from HuggingFace (dataset='ett', config='h1') …")
            df = _fetch_huggingface()
        except Exception as exc:
            print(f"  HuggingFace failed ({type(exc).__name__}: {exc}); "
                  f"falling back to GitHub.")
            df = _fetch_github()

    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("h")
    df["ot"] = df["ot"].interpolate(method="time").ffill().bfill()

    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out, index=False)
    print(f"Saved {out} with {len(df):,} rows "
          f"({df.index[0]} → {df.index[-1]}).")


# --- CLI ---

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset", choices=["elia", "etth1", "all"],
                   help="Which dataset to prepare.")
    p.add_argument("--source", choices=["auto", "huggingface", "github"],
                   default="auto",
                   help="ETTh1 source (ignored for elia).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path override (ETTh1 only).")
    args = p.parse_args()

    if args.dataset in ("elia", "all"):
        prepare_elia()
    if args.dataset in ("etth1", "all"):
        prepare_etth1(
            source=args.source,
            out=args.out or OUT_ETTH1,
        )


if __name__ == "__main__":
    main()
