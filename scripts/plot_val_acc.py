"""Plot val accuracy curves for selected lightning runs (truncated to 25 epochs).

Usage:
python scripts/plot_val_acc.py \
  --runs version_21 logs/lightning/version_21/metrics.csv "ConvNeXt-Tiny mixup" \
        version_25 logs/lightning/version_25/metrics.csv "ConvNeXt-Tiny RandAug n=1,m=9" \
        version_27 logs/lightning/version_27/metrics.csv "ConvNeXt-S 45ep" \
  --out reports/val_acc_compare.png

Notes:
- Only uses `val/acc_species_top1` and caps epochs to 25 for fair visualization.
- Add/remove runs on the CLI as needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot val acc curves (epoch<=25)")
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Triplets: run_id path label (e.g., version_25 logs/... \"Our run\"), repeatable",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/val_acc_compare.png"))
    return parser.parse_args()


def load_curve(metrics_path: Path) -> Tuple[List[int], List[float]]:
    df = pd.read_csv(metrics_path)
    # Prefer val acc columns; if missing (e.g., old ViT run), fallback to train acc.
    candidate_cols = [
        "val/acc_species_top1",
        "val/acc",
        "val_acc",
        "val/acc_species_top1_epoch",
        # fallback to train if no val present
        "train/acc_species_top1_step",
        "train/acc",
    ]
    acc_col = None
    for col in candidate_cols:
        if col in df.columns:
            acc_col = col
            break
    if acc_col is None:
        raise KeyError(
            f"No val/train acc column found in {metrics_path}; available cols: {list(df.columns)}"
        )

    df = df.dropna(subset=[acc_col])
    df = df[df["epoch"] <= 25]
    return df["epoch"].tolist(), df[acc_col].tolist()


def plot_curves(runs: List[Tuple[str, Path, str]], out_path: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    for run_id, metrics_path, label in runs:
        epochs, accs = load_curve(metrics_path)
        plt.plot(epochs, accs, marker="o", label=label)

    plt.title("val/acc (species) up to epoch 25")
    plt.xlabel("epoch")
    plt.ylabel("val acc (species)")
    plt.ylim(0.3, 0.85)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    args = parse_args()
    if not args.runs or len(args.runs) % 3 != 0:
        raise SystemExit("--runs expects triplets: ID METRICS_CSV LABEL")

    triplets = []
    for i in range(0, len(args.runs), 3):
        run_id, metrics_path, label = args.runs[i : i + 3]
        triplets.append((run_id, Path(metrics_path), label))

    plot_curves(triplets, args.out)


if __name__ == "__main__":
    main()
