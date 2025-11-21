"""Plot comparison between our best run and a teammate's metrics.

Usage example:
python scripts/compare_vs_teammate.py \
  --ours logs/lightning/version_25/metrics.csv \
  --theirs-species 0.80 --theirs-genus 0.78 --theirs-family 0.82 \
  --out reports/compare_vs_teammate.png

The script reads our metrics CSV, finds the max macro-F1 for family/genus/species,
and plots side-by-side bars against the teammate's provided numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def load_best_metrics(metrics_path: Path) -> Dict[str, float]:
    df = pd.read_csv(metrics_path)
    val_df = df.dropna(subset=["val/macro_f1_species"])
    return {
        "family": float(val_df["val/macro_f1_family"].max()),
        "genus": float(val_df["val/macro_f1_genus"].max()),
        "species": float(val_df["val/macro_f1_species"].max()),
    }


def plot_comparison(ours: Dict[str, float], theirs: Dict[str, float], out_path: Path) -> None:
    labels = ["family", "genus", "species"]
    ours_vals = [ours[l] for l in labels]
    theirs_vals = [theirs[l] for l in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], ours_vals, width, label="ours", color="#4C84FF")
    ax.bar([i + width / 2 for i in x], theirs_vals, width, label="teammate", color="#FFB74C")

    ax.set_ylabel("Macro-F1")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Ours vs Teammate (macro-F1)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for idx, val in enumerate(ours_vals):
        ax.text(idx - width / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for idx, val in enumerate(theirs_vals):
        ax.text(idx + width / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare our metrics with teammate's")
    parser.add_argument("--ours", type=Path, default=Path("logs/lightning/version_25/metrics.csv"))
    parser.add_argument("--theirs-family", type=float, required=True)
    parser.add_argument("--theirs-genus", type=float, required=True)
    parser.add_argument("--theirs-species", type=float, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/compare_vs_teammate.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ours = load_best_metrics(args.ours)
    theirs = {
        "family": args.theirs_family,
        "genus": args.theirs_genus,
        "species": args.theirs_species,
    }
    plot_comparison(ours, theirs, args.out)
    print(f"Saved comparison plot to {args.out}")
    print(f"Ours: {ours}")
    print(f"Theirs: {theirs}")


if __name__ == "__main__":
    main()
