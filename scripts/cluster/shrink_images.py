"""
shrink_images.py

Makes a copy of a YOLO dataset (images/ + labels/ + data.yaml) with every image
shrunk so its long side is at most --max-side px. Labels are normalised (0-1)
coordinates, so they are copied unchanged.

Why: Ultralytics resizes every image to imgsz (1600) the moment it loads it, so
full-resolution Mothbox photos (9248x6944, ~15 MB JPEG) only cost CPU time. On the
6-CPU cluster VM, decoding them starved the GPU: ~12 min/epoch at full size vs
~1 min/epoch from a 2048 px copy, with the same model input (2026-10-15).
2048 leaves room for a future imgsz=2048 experiment.

Re-running skips images already written.

Usage (on the cluster):
    python3 shrink_images.py \
        --src /mnt/ami-training/detector/sources/incoming-andy/mothbox_mbd11 \
        --dst /mnt/ami-training/detector/sources/incoming-andy/mothbox_mbd11_2048
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def shrink(p, src, dst, max_side):
    out = dst / "images" / p.relative_to(src / "images")
    if out.exists():
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(out, quality=95)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    imgs = sorted(p for p in (args.src / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"{len(imgs)} images to check", flush=True)
    work = partial(shrink, src=args.src, dst=args.dst, max_side=args.max_side)
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for k, n in enumerate(pool.map(work, imgs, chunksize=8), 1):
            done += n
            if k % 100 == 0 or k == len(imgs):
                print(f"  [{k}/{len(imgs)}] {done} newly shrunk", flush=True)
    shutil.copytree(args.src / "labels", args.dst / "labels", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.cache"))
    if (args.src / "data.yaml").exists():
        shutil.copy(args.src / "data.yaml", args.dst / "data.yaml")
    print(f"done -> {args.dst}")


if __name__ == "__main__":
    main()
