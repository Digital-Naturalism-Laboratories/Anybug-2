"""
fieldguide_sam_boxes.py

Tightens Fieldguide's loose square user-crop boxes with SAM: each photo's
square is given to SAM 2.1 as a box prompt, the largest connected piece of the
returned mask is outlined, and cv2.minAreaRect of it becomes a rotated box.

Each result is flagged keep=True only when the mask looks like one subject:
  - SAM returned a mask,
  - the mask is in <= --max-pieces connected pieces (masks that grab textured
    walls/bark come back in dozens to hundreds of pieces), and
  - the rotated box covers < --max-ratio of the square (a box nearly as big as
    the square means SAM outlined the background).
On a 50-photo hand check (2026-10-13) this kept 28, of which 2 were bad.

Progress is banked per photo to <out>/sam_boxes.jsonl — re-running resumes.

Usage:
    python3 fieldguide_sam_boxes.py \
        --data  /Users/automeris/Desktop/Megabug/fg-arthropods-v1 \
        --sam   /Users/automeris/Desktop/Megabug/models/sam2.1_b.pt \
        --out   /Users/automeris/Desktop/Megabug/fieldguide_sam
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def sam_box(model, bgr, square):
    """(rotated box (4,2), outline (N,2), pieces, ratio) or None if SAM found nothing."""
    masks = model(bgr, bboxes=[square], verbose=False)[0].masks
    if masks is None or len(masks.data) == 0:
        return None
    mask = masks.data[0].cpu().numpy().astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    biggest = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    rect = cv2.minAreaRect(cv2.findNonZero(biggest))
    contours, _ = cv2.findContours(biggest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
    x0, y0, x1, y1 = square
    ratio = rect[1][0] * rect[1][1] / max((x1 - x0) * (y1 - y0), 1e-6)
    return cv2.boxPoints(rect), outline, n - 1, ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="fg-arthropods-v1 folder")
    ap.add_argument("--sam", required=True, help="SAM weights, e.g. sam2.1_b.pt")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--groups", nargs="*", default=["moth", "butterfly", "other_arthropod"])
    ap.add_argument("--max-pieces", type=int, default=12)
    ap.add_argument("--max-ratio", type=float, default=0.8)
    ap.add_argument("--limit", type=int, help="max photos (testing)")
    args = ap.parse_args()

    from ultralytics import SAM
    model = SAM(args.sam)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "sam_boxes.jsonl"
    done = set()
    if log_path.exists():
        for line in open(log_path):
            try:
                done.add(json.loads(line)["image_key"])
            except json.JSONDecodeError:
                continue  # last line mid-write

    rows = [r for r in map(json.loads, open(args.data / "index.jsonl")) if r["group"] in args.groups]
    work = [r for r in rows if r["image_key"] not in done][:args.limit]
    print(f"{len(done)} photos already done, {len(work)} to go", flush=True)

    t0, kept = time.time(), 0
    with open(log_path, "a") as log:
        for k, r in enumerate(work, 1):
            img = ImageOps.exif_transpose(Image.open(args.data / r["image_key"])).convert("RGB")
            rec = {"image_key": r["image_key"], "group": r["group"], "square": r["bbox_xyxy"],
                   "w": r["width"], "h": r["height"], "keep": False}
            if img.size != (r["width"], r["height"]):
                rec["error"] = f"size {img.size} vs index"
            else:
                res = sam_box(model, np.ascontiguousarray(np.asarray(img)[:, :, ::-1]), r["bbox_xyxy"])
                if res is None:
                    rec["error"] = "no mask"
                else:
                    box, outline, pieces, ratio = res
                    rec.update(obb=np.round(box.astype(float), 1).tolist(), outline=outline.astype(int).tolist(),
                               pieces=int(pieces), ratio=round(float(ratio), 3),
                               keep=bool(pieces <= args.max_pieces and ratio < args.max_ratio))
            kept += rec["keep"]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if k % 100 == 0 or k == len(work):
                el = time.time() - t0
                print(f"  [{k}/{len(work)}] {r['group']:16s} kept {kept}/{k} ({kept / k:.0%})  "
                      f"{el / k:4.2f}s/photo  ETA {el / k * (len(work) - k) / 60:5.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
