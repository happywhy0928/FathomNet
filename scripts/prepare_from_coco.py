#!/usr/bin/env python3
"""Convert COCO manifests into normalized CSV splits and label encoders."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.kaggle_ingest import COL_ORDER, run_ingest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CSVs from COCO annotations")
    parser.add_argument("--num-workers", type=int, default=16, dest="num_workers")
    parser.add_argument("--min-per-class", type=int, default=50, dest="min_per_class")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing images")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def ensure_external_assets(external_dir: Path) -> dict[str, Path]:
    assets = {
        "train": external_dir / "dataset_train.json",
        "test": external_dir / "dataset_test.json",
        "download": external_dir / "download.py",
    }
    missing = [path for path in assets.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing external assets: " + ", ".join(str(p) for p in missing))
    return assets


def run_external_downloader(script: Path, manifest: Path, output_dir: Path, workers: int) -> None:
    cmd = [sys.executable, str(script), str(manifest), str(output_dir), "-n", str(workers)]
    print(f"[prepare_from_coco] Running {' '.join(cmd)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def infer_majority_species(manifest: Path, image_rel_root: Path, repo_root: Path) -> pd.DataFrame:
    with manifest.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    categories = {cat["id"]: cat.get("name") or cat.get("title") for cat in data.get("categories", [])}
    image_to_counts = defaultdict(Counter)
    for ann in data.get("annotations", []):
        cat_name = categories.get(ann.get("category_id"))
        if cat_name:
            image_to_counts[ann.get("image_id")][cat_name] += 1

    rows = []
    missing_files = 0
    images = data.get("images", [])
    for img in tqdm(images, desc="Consolidating annotations", unit="img"):
        image_id = img.get("id")
        if image_id not in image_to_counts:
            continue
        species = image_to_counts[image_id].most_common(1)[0][0]
        url = img.get("coco_url") or img.get("url")
        file_name = Path(url).name if url else (img.get("file_name") or img.get("path"))
        if not file_name:
            continue
        rel_path = (image_rel_root / Path(file_name).name).as_posix()
        abs_path = (repo_root / rel_path).resolve()
        if not abs_path.is_file():
            missing_files += 1
            continue
        rows.append(
            {
                "image": rel_path,
                "family": None,
                "genus": species.split()[0] if species else None,
                "species": species,
                "concept": species,
            }
        )

    if missing_files:
        print(f"[prepare_from_coco] Skipped {missing_files} entries without local files")

    df = pd.DataFrame(rows, columns=COL_ORDER).drop_duplicates(subset="image")
    if df.empty:
        raise RuntimeError("No labeled rows generated from manifest")
    print(f"[prepare_from_coco] Manifest {manifest.name} -> {len(df)} rows")
    return df


def main() -> None:
    args = parse_args()
    external_dir = REPO_ROOT / "external" / "fathomnet-2025"
    assets = ensure_external_assets(external_dir)

    img_dir = REPO_ROOT / "data" / "kaggle" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    reuse_existing = args.skip_download or any(img_dir.iterdir())
    if reuse_existing:
        reason = "flag set" if args.skip_download else "images already present"
        print(f"[prepare_from_coco] Skipping download step ({reason})")
    else:
        run_external_downloader(assets["download"], assets["train"], img_dir, args.num_workers)
        run_external_downloader(assets["download"], assets["test"], img_dir, args.num_workers)

    image_rel_root = Path("data") / "kaggle" / "images"
    df = infer_majority_species(assets["train"], image_rel_root, REPO_ROOT)
    splits_dir = REPO_ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = splits_dir / "all_from_coco.csv"
    df.to_csv(combined_csv, index=False)
    print(f"[prepare_from_coco] Wrote consolidated CSV to {combined_csv.relative_to(REPO_ROOT)}")

    run_ingest(
        in_csv=combined_csv,
        img_root=image_rel_root,
        out_dir=splits_dir,
        min_per_class=args.min_per_class,
        seed=2025,
        save_encoders=True,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
