#!/usr/bin/env python3
"""Download a public labeled fake-vs-genuine fashion dataset from Kaggle.

Default dataset slug is the public "Fake Brand Data" dataset. Kaggle credentials
are required by kagglehub. Put them in ~/.kaggle/kaggle.json or use Kaggle's
supported environment authentication.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import kagglehub

DEFAULT_DATASET = "prashanthgolla/fake-brand-data"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.dataset} ...")
    src = Path(kagglehub.dataset_download(args.dataset)).resolve()
    print(f"Downloaded to: {src}")
    target = out / src.name
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    if src.is_dir():
        shutil.copytree(src, target)
    else:
        shutil.copy2(src, target)
    print(f"Copied dataset to: {target}")


if __name__ == "__main__":
    main()
