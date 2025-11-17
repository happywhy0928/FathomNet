#!/usr/bin/env python3
"""Resolve taxonomy for all species present in the current splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.taxonomy_resolver import resolve_lineage

DEFAULT_SPLITS = ["train.csv", "val.csv", "test.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build configs/taxonomy_map.json")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory with CSV splits")
    parser.add_argument(
        "--output",
        default="configs/taxonomy_map.json",
        help="Destination JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    if not splits_dir.is_absolute():
        splits_dir = REPO_ROOT / splits_dir

    species = set()
    for file_name in DEFAULT_SPLITS:
        csv_path = splits_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        species.update(df["species"].dropna().astype(str).str.strip())

    species_list = sorted(species)
    print(f"[taxonomy] Resolving {len(species_list)} species")

    records = []
    family_hits = 0
    for name in species_list:
        lineage = resolve_lineage(name)
        if lineage.get("family"):
            family_hits += 1
        records.append(lineage)
        print(
            f"[taxonomy] {name} -> genus={lineage.get('genus') or 'NA'}, family={lineage.get('family') or 'NA'}"
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    coverage = family_hits / max(1, len(records)) * 100
    print(
        f"[taxonomy] Wrote {len(records)} entries to {out_path.relative_to(REPO_ROOT)}"
        f" | family coverage={coverage:.1f}%"
    )


if __name__ == "__main__":
    main()
