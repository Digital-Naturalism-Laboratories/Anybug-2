"""
mothbox_apply_empty_review.py

Turns the Mothbox empty-label review (mothbox_empty_check.py + review_missed.py)
into corrected label files. Every detection Andrew marked green ("insect") becomes
a box in its photo's label file; red and unreviewed detections are ignored.
Photos with no green detections stay empty (they really are background).

Before writing, approved boxes in the same photo that mostly overlap (IoU of their
axis-aligned bounds > --dedupe-iou) are merged by keeping the more confident one,
and any ids in --drop are removed. --drop is for the detector putting one box
around two neighbouring insects that it ALSO boxed individually: that box overlaps
each single box too little for the IoU rule to catch (2026-10-15: test photo
adeptTurca_2024_09_02__04_34_05, merged box MBE-test-6025fea4-008).

Writes ONLY the changed label files, in YOLO OBB format normalised to the photo
size, to <out>/labels/<split>/<stem>.txt, plus <out>/changes.json. Because the
coordinates are normalised, the same files fit the full-size set and the 2048 px
shrunk copy. Copy them over a COPY of the dataset (see TRAINING.md); the original
dataset is not touched.

Usage:
    python3 mothbox_apply_empty_review.py \
        --review /Users/automeris/Desktop/Megabug/mothbox_empty_review \
        --out    /Users/automeris/Desktop/Megabug/mothbox_mbd12_label_patch \
        --drop MBE-test-6025fea4-008
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def aabb_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ax0, ay0, ax1, ay1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    bx0, by0, bx1, by1 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", required=True, type=Path, help="mothbox_empty_check.py output with decisions.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--drop", nargs="*", default=[], help="detection ids to leave out even if approved")
    ap.add_argument("--dedupe-iou", type=float, default=0.5)
    args = ap.parse_args()

    decisions = json.loads((args.review / "decisions.json").read_text())
    changes = []
    for line in open(args.review / "detections.jsonl"):
        r = json.loads(line)
        keep = sorted((d for d in r["dets"]
                       if decisions.get(d["id"], {}).get("label") == "insect" and d["id"] not in args.drop),
                      key=lambda d: -d["conf"])
        if not keep:
            continue
        final, merged = [], []
        for d in keep:
            dup = next((f for f in final if aabb_iou(d["pts"], f["pts"]) > args.dedupe_iou), None)
            (merged if dup else final).append(d)
        split, name = r["file"].split("/", 1)
        stem = Path(name).stem
        w, h = r["w"], r["h"]
        lines = ["0 " + " ".join(f"{v:.6f}" for v in np.clip(np.asarray(d["pts"]) / [w, h], 0, 1).ravel())
                 for d in final]
        out = args.out / "labels" / split / f"{stem}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        changes.append({"file": r["file"], "boxes": len(final), "ids": [d["id"] for d in final],
                        "merged_duplicates": [d["id"] for d in merged]})

    (args.out / "changes.json").write_text(json.dumps(changes, indent=1))
    n_boxes = sum(c["boxes"] for c in changes)
    n_dup = sum(len(c["merged_duplicates"]) for c in changes)
    by_split = {}
    for c in changes:
        by_split[c["file"].split("/")[0]] = by_split.get(c["file"].split("/")[0], 0) + 1
    print(f"{len(changes)} label files fixed ({by_split}), {n_boxes} boxes added; "
          f"{n_dup} overlapping duplicates merged, {len(args.drop)} id(s) dropped by hand -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
