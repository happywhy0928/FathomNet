#!/usr/bin/env python3
"""Download real FathomNet imagery referenced by COCO manifests."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download COCO-referenced imagery")
    parser.add_argument("--train", required=True, help="Path to dataset_train.json")
    parser.add_argument("--test", required=True, help="Path to dataset_test.json")
    parser.add_argument("--out", required=True, help="Destination directory for images")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent download workers")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_manifest(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    entries = []
    for image in data.get("images", []):
        url = image.get("coco_url") or image.get("url") or image.get("flickr_url")
        if not url:
            continue
        entries.append({"id": image.get("id"), "file_name": image.get("file_name"), "url": url})
    return entries


def infer_filename(entry: Dict, used: set[str]) -> str:
    url_name = Path(urlparse(entry["url"]).path).name
    base = url_name or (entry.get("file_name") or f"image_{entry['id']}.jpg")
    candidate = base
    idx = 1
    while candidate in used:
        candidate = f"{entry['id']}_{idx}_{base}"
        idx += 1
    used.add(candidate)
    return candidate


def validate_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_one(url: str, dest: Path) -> Tuple[bool, str]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return False, f"request failed: {exc}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(resp.content)
    if not validate_image(tmp):
        tmp.unlink(missing_ok=True)
        return False, "invalid image"
    tmp.replace(dest)
    return True, ""


def main() -> None:
    args = parse_args()
    manifests = [resolve_path(args.train), resolve_path(args.test)]
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict] = []
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        subset = load_manifest(manifest)
        entries.extend(subset)
        print(f"[download_real] Loaded {len(subset)} entries from {manifest.relative_to(REPO_ROOT)}")

    used_names: set[str] = set()
    tasks: List[Dict] = []
    skipped = 0
    for entry in entries:
        filename = infer_filename(entry, used_names)
        dest = out_dir / filename
        if validate_image(dest):
            skipped += 1
            continue
        tasks.append({"url": entry["url"], "dest": dest})

    print(
        f"[download_real] Total entries={len(entries)} | pending={len(tasks)} | already valid={skipped}"
    )

    downloaded = 0
    failed = 0
    errors: List[str] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_map = {pool.submit(download_one, task["url"], task["dest"]): task for task in tasks}
            for idx, future in enumerate(as_completed(future_map), start=1):
                ok, msg = future.result()
                if ok:
                    downloaded += 1
                else:
                    failed += 1
                    errors.append(f"{future_map[future]['url']}: {msg}")
                if idx % 50 == 0:
                    print(
                        f"[download_real] Progress {idx}/{len(tasks)} | downloaded={downloaded} | failed={failed}"
                    )

    failure_rate = failed / max(1, len(entries)) * 100
    print(
        f"[download_real] Summary -> downloaded={downloaded}, skipped={skipped}, failed={failed} ({failure_rate:.2f}% )"
    )
    if errors:
        print("[download_real] Sample errors:")
        for line in errors[:5]:
            print("   -", line)
    if failure_rate > 1.0:
        print("[download_real] Warning: failure rate exceeds 1%", file=sys.stderr)


if __name__ == "__main__":
    main()
