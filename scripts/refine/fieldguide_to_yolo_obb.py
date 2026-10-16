"""
fieldguide_to_yolo_obb.py

Converts the Fieldguide photos Andrew APPROVED in the SAM box review into a
Mothbox-style YOLO OBB folder (class 0 = creature) for the Mixed set:

    <out>/other/images/FG-<group>_<sha>.jpg
    <out>/other/labels/...txt        rotated box from SAM (cv2.minAreaRect of its mask)
    <out>/other/labels_seg/...txt    SAM's outline (YOLO-seg format)
    <out>/manifest.jsonl             source image_key, taxon, licence class per output

Only photos marked green ("insect") in fieldguide_sam_review/decisions.json are
used. Red ones (bad box, or other unlabelled insects in the photo), unreviewed
ones and photos SAM's keep-rule dropped are all left out.

Images that need no change are hard-linked (no extra disk); EXIF-rotated ones are
saved upright (JPEG q95), since index/SAM coordinates are in upright pixels.

Usage:
    python3 fieldguide_to_yolo_obb.py \
        --data   /Users/automeris/Desktop/Megabug/fg-arthropods-v1 \
        --sam    /Users/automeris/Desktop/Megabug/fieldguide_sam \
        --review /Users/automeris/Desktop/Megabug/fieldguide_sam_review \
        --out    /Users/automeris/Desktop/Megabug/fieldguide_yolo_obb
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from flatbug_find_missed import FIELDGUIDE_CODES  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def norm_line(pts, w, h):
    xy = np.clip(np.asarray(pts, float) / [w, h], 0, 1)  # Ultralytics rejects coords outside 0..1
    return "0 " + " ".join(f"{v:.6f}" for v in xy.ravel())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--sam", required=True, type=Path)
    ap.add_argument("--review", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    root = args.out / "other"
    for sub in ("images", "labels", "labels_seg"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    decisions = json.loads((args.review / "decisions.json").read_text())
    id_to_key = {}
    for line in open(args.review / "detections.jsonl"):
        r = json.loads(line)
        id_to_key[r["dets"][0]["id"]] = r["file"]
    approved = {id_to_key[i] for i, v in decisions.items() if v["label"] == "insect" and i in id_to_key}
    sam = {r["image_key"]: r for r in map(json.loads, open(args.sam / "sam_boxes.jsonl"))}
    meta = {r["image_key"]: r for r in map(json.loads, open(args.data / "index.jsonl"))}
    print(f"{len(approved)} approved photos", flush=True)

    rotated = 0
    with open(args.out / "manifest.jsonl", "w") as man:
        for key in sorted(approved):
            rec, m = sam[key], meta[key]
            name = f"FG-{FIELDGUIDE_CODES.get(rec['group'], 'FG')}_{m['sha256'][:16]}"
            src = args.data / key
            dst = root / "images" / f"{name}.jpg"
            img = Image.open(src)
            if img.getexif().get(274) not in (None, 1):
                if not dst.exists():
                    ImageOps.exif_transpose(img).convert("RGB").save(dst, quality=95)
                rotated += 1
            elif not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
            w, h = rec["w"], rec["h"]
            (root / "labels" / f"{name}.txt").write_text(norm_line(rec["obb"], w, h) + "\n")
            (root / "labels_seg" / f"{name}.txt").write_text(norm_line(rec["outline"], w, h) + "\n")
            man.write(json.dumps({"name": name, "image_key": key, "group": rec["group"],
                                  "taxon": m.get("taxon_name"), "order": m.get("order"),
                                  "license_class": m.get("license_class")}) + "\n")
    print(f"wrote {len(approved)} images ({rotated} re-saved upright) -> {root}")


if __name__ == "__main__":
    sys.exit(main())
