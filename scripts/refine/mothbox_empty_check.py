"""
mothbox_empty_check.py

Checks the Mothbox photos whose label file is EMPTY (used as "background" in
training). Some aren't empty at all (found 2026-10-15: a test "background" with a
moth, flies and a swallowtail), which teaches the detector to ignore real insects.

A detector (MBD-1-2 by default) is run over each empty-label photo and EVERY
detection becomes its own review tile, in the same format as the flat-bug
missed-creature review:

  <id>_s.jpg   tight crop (6x the box, >= 96 px) with the detection in pink
  <id>_b.jpg   wide crop  (18x the box, >= 480 px) for context

Open it with review_missed.py:
  green (click)        = a real creature -> becomes a label in that photo
  red (right-click)    = not a creature (debris, smudge, sheet texture)

Each detection keeps its rotated box ("pts", in photo pixels), so the green ones
can be written straight into the photo's label file afterwards.

--dets-from reuses detections saved by an earlier run (the old whole-photo version
of this script stored them under dets[0].boxes) instead of running the model again.

Usage:
    python3 mothbox_empty_check.py \
        --data  /Users/automeris/Desktop/2026_09_17_YOLOOBB_TRAINING \
        --model /Users/automeris/Documents/GitHub/Mothbot_Process/trained_models/MBD-1-2.pt \
        --out   /Users/automeris/Desktop/Megabug/mothbox_empty_review
    python3 review_missed.py --dir /Users/automeris/Desktop/Megabug/mothbox_empty_review \
        --cols 5 --max-rows 2 --max-size 100000 --title "Mothbox empty-photo detections"
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from flatbug_find_missed import save_crop  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def empty_photos(root):
    """[(split, image path)] for every image whose label file is missing or empty."""
    out = []
    for split_dir in sorted((root / "images").iterdir()):
        if not split_dir.is_dir():
            continue
        for img in sorted(split_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = root / "labels" / split_dir.name / (img.stem + ".txt")
            if not lbl.exists() or not lbl.read_text().strip():
                out.append((split_dir.name, img))
    return out


def load_saved(path):
    """{"split/name": [(pts, conf)]} from an earlier run's detections.jsonl."""
    saved = {}
    for line in open(path):
        r = json.loads(line)
        saved[r["file"]] = r["dets"][0].get("boxes", [])
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="YOLO dataset root (images/<split>, labels/<split>)")
    ap.add_argument("--model", help="detector weights (not needed with --dets-from)")
    ap.add_argument("--dets-from", type=Path, help="reuse detections from an earlier run's detections.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--conf", type=float, default=0.1, help="lowest detection confidence kept")
    args = ap.parse_args()

    saved = load_saved(args.dets_from) if args.dets_from else None
    if saved is None:
        from ultralytics import YOLO
        model = YOLO(args.model, task="obb")

    (args.out / "crops").mkdir(parents=True, exist_ok=True)
    log_path = args.out / "detections.jsonl"
    done = set()
    if log_path.exists():
        for line in open(log_path):
            try:
                done.add(json.loads(line)["file"])
            except json.JSONDecodeError:
                continue  # last line mid-write
    work = [(s, p) for s, p in empty_photos(args.data) if f"{s}/{p.name}" not in done]
    print(f"{len(done)} photos already done, {len(work)} empty-label photos to check", flush=True)

    t0, n_tiles = time.time(), 0
    with open(log_path, "a") as log:
        for k, (split, path) in enumerate(work, 1):
            key = f"{split}/{path.name}"
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            if saved is not None:
                found = [(p, c) for p, c in saved.get(key, []) if c >= args.conf]
            else:
                r = model.predict(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]), imgsz=args.imgsz,
                                  conf=args.conf, max_det=5000, verbose=False)[0]
                found = ([(p.tolist(), float(c)) for p, c in
                          zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy())]
                         if r.obb is not None and len(r.obb) else [])
            stem = hashlib.sha1(key.encode()).hexdigest()[:8]
            dets = []
            for i, (pts, conf) in enumerate(sorted(found, key=lambda d: -d[1])):
                p = np.asarray(pts, float)
                e1, e2 = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])
                L, cap = float(max(e1, e2)), max(img.size)
                cid = f"MBE-{split}-{stem}-{i:03d}"
                save_crop(img, p, [], min(max(6 * L, 96), cap), 240, args.out / "crops" / f"{cid}_s.jpg")
                save_crop(img, p, [], min(max(18 * L, 480), cap), 720, args.out / "crops" / f"{cid}_b.jpg")
                dets.append({"id": cid, "pts": np.round(p, 1).tolist(), "conf": round(conf, 3),
                             "long": round(L, 1), "short": round(float(min(e1, e2)), 1), "matched": False,
                             "caption": f"{conf:.2f} · {L:.0f}px · {split} · {path.stem}"})
                n_tiles += 1
            log.write(json.dumps({"subset": split, "code": split, "file": key, "path": str(path),
                                  "w": img.width, "h": img.height, "dets": dets}) + "\n")
            log.flush()
            if k % 25 == 0 or k == len(work):
                el = time.time() - t0
                print(f"  [{k}/{len(work)}] {split:5s} {n_tiles} tiles  ETA {el / k * (len(work) - k) / 60:4.1f} min",
                      flush=True)


if __name__ == "__main__":
    sys.exit(main())
