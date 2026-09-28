#!/usr/bin/env python3
from __future__ import annotations
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}



def infer_label(path: Path) -> int | None:
    for part in path.parts:
        name = part.lower()

        if name.endswith("_genuine"):
            return 0

        if name.endswith("_fake"):
            return 1

    return None


def valid_image(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    src, out = Path(args.src), Path(args.out)
    images = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS and valid_image(p)]
    labeled = [(p, infer_label(p)) for p in images]
    labeled = [(p, y) for p, y in labeled if y is not None]
    if len(labeled) < 20:
        raise SystemExit(
            f"Only {len(labeled)} labeled images found. "
            "The source dataset must expose real/genuine and fake/counterfeit labels "
            "in folder or filename structure."
        )
    paths = [str(p) for p, _ in labeled]
    y = [label for _, label in labeled]
    train_p, tmp_p, train_y, tmp_y = train_test_split(paths, y, test_size=0.30, stratify=y, random_state=args.seed)
    val_p, test_p, val_y, test_y = train_test_split(tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=args.seed)
    if out.exists():
        shutil.rmtree(out)
    for split, items in {"train": zip(train_p, train_y), "val": zip(val_p, val_y), "test": zip(test_p, test_y)}.items():
        for src_path, label in items:
            cls = "fake" if label == 1 else "real"
            dst_dir = out / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            src_file = Path(src_path)
            # Prefix with a hash-like deterministic counter to avoid collisions.
            dst = dst_dir / f"{src_file.parent.name}_{src_file.name}"
            if dst.exists():
                dst = dst_dir / f"{abs(hash(src_path))}_{src_file.name}"
            shutil.copy2(src_file, dst)
    print(f"Prepared {len(train_p)} train / {len(val_p)} val / {len(test_p)} test images in {out}")
    for split, labels in [("train", train_y), ("val", val_y), ("test", test_y)]:
        print(split, {"real": labels.count(0), "fake": labels.count(1)})


if __name__ == "__main__":
    main()
