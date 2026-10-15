"""
yolofast_to_yolo_obb.py

Converts YOLO-Fast-dataset.zip (257 focus-stacked macro photos, one
axis-aligned box per photo) into a Mothbox-style YOLO OBB folder for the
mixed "other backgrounds" mashup only:

    <out>/other/images/YF_<name>.jpg   <out>/other/labels/YF_<name>.txt

Their axis-aligned boxes are kept as unrotated OBBs. --refit swaps in the
rotated box of the Mothbox detection overlapping each one best (IoU >= --min-iou),
but checked on 2026-10-13 with MBD-1-1 that made boxes worse (cut legs, heads,
half a beetle — macro portraits are far from trap photos), so it is off by
default. PNGs are re-saved as JPEG (quality 95) to save ~2 GB.

Progress is banked per image to <out>/manifest.jsonl — re-running resumes.

Usage:
    python3 yolofast_to_yolo_obb.py \
        --zip   /Users/automeris/Desktop/Megabug/YOLO-Fast-dataset.zip \
        --out   /Users/automeris/Desktop/Megabug/yolofast_yolo_obb
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from flatbug_find_missed import aabb, iou_matrix  # noqa: E402
from flatbug_to_yolo_obb import safe_stem  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--refit", action="store_true", help="re-fit boxes with a detector (see above)")
    ap.add_argument("--model", help="detector for --refit")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--min-iou", type=float, default=0.5)
    args = ap.parse_args()

    if args.refit:
        from ultralytics import YOLO
        model = YOLO(args.model)

    root = args.out / "other"
    for sub in ("images", "labels"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["source"] for l in open(manifest) if l.strip()}

    z = zipfile.ZipFile(args.zip)
    images = sorted(n for n in z.namelist() if n.lower().endswith(".png") and n not in done)
    print(f"{len(done)} images already converted, {len(images)} to go", flush=True)

    with open(manifest, "a") as log:
        for k, n in enumerate(images, 1):
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(n)))).convert("RGB")
            W, H = img.size
            label = "labels/" + Path(n).stem + ".txt"
            theirs = [list(map(float, l.split()[1:])) for l in z.read(label).decode().splitlines()
                      if len(l.split()) == 5]

            det = np.zeros((0, 4, 2))
            if args.refit:
                r = model.predict(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]), imgsz=args.imgsz,
                                  conf=args.conf, verbose=False)[0]
                if r.obb is not None and len(r.obb):
                    det = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)

            lines, fits = [], []
            for cx, cy, w, h in theirs:
                box = np.array([[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]])
                iou = iou_matrix(aabb(det), box)[:, 0] if len(det) else np.zeros(0)
                if len(iou) and iou.max() >= args.min_iou:
                    pts = det[int(iou.argmax())] / [W, H]
                    fits.append(round(float(iou.max()), 3))
                else:
                    x0, y0, x1, y1 = box[0] / [W, H, W, H]
                    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                    fits.append(None)
                pts = np.clip(pts, 0, 1)  # Ultralytics rejects a label file with any coord outside 0..1
                lines.append("0 " + " ".join(f"{v:.6f}" for v in pts.ravel()))

            name = f"YF_{safe_stem(n)}"
            img.save(root / "images" / f"{name}.jpg", quality=95)
            (root / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
            log.write(json.dumps({"source": n, "name": name, "size": [W, H], "refit_iou": fits}) + "\n")
            log.flush()
            if k % 25 == 0 or k == len(images):
                print(f"  [{k}/{len(images)}]", flush=True)

    rows = [json.loads(l) for l in open(manifest)]
    fits = [f for r in rows for f in r["refit_iou"]]
    refit = [f for f in fits if f is not None]
    print(f"\n{len(rows)} images, {len(fits)} boxes: {len(refit)} re-fitted with the detector "
          + (f"(median IoU {np.median(refit):.2f}), " if refit else "")
          + f"{len(fits) - len(refit)} kept as their unrotated box")


if __name__ == "__main__":
    sys.exit(main())
