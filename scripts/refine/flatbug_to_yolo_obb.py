"""
flatbug_to_yolo_obb.py

Converts the flat-bug dataset into Mothbox-style YOLO OBB folders (class
0 = creature), ready to pass as a --source to collect_yolo_training_data.py:

    <out>/flat/images/*.jpg   <out>/flat/labels/*.txt   <out>/flat/labels_seg/*.txt
    <out>/other/...

  - flat  : plain/flat backgrounds (sheets, trays, dishes, sticky cards, ...)
  - other : NHM-beetles-crops, DIRT, PeMaToEuroPep, ArTaxOr
  Flat-only mashup = Mothbox + flat. Mixed mashup = Mothbox + flat + other.

What it does:
  - Each flat-bug outline becomes a rotated box via cv2.minAreaRect (the same
    method Ultralytics uses for seg -> OBB). Outlines are kept too, in YOLO-seg
    format, under labels_seg/ (collect_yolo_training_data.py ignores that folder).
  - Adds the creatures marked "insect" in review_missed.py (decisions.json).
    Every other detection from flatbug_find_missed.py is ignored.
  - Images are written upright (EXIF orientation applied — PeMaToEuroPep has
    99 rotated photos whose outlines are in upright coordinates).
  - Subsets whose median creature would be smaller than --target-px at the
    model's 1600 px input are cut into the fewest evenly spaced tiles of about
    the size that gives that scale (Mothbox median is ~25 px at imgsz 1600),
    neighbours overlapping by --overlap. Creatures cut
    by a tile edge are kept (clipped) if at least --min-visible of their outline
    area is inside the tile.
  - Images flat-bug left with no annotations are skipped (we can't tell
    "empty" from "not annotated").
  - Images that need no change are hard-linked from flatbug-dataset (no extra
    disk space); tiles and re-rotated photos are new JPEGs (quality 95).
  - Files are named FB-<code>_<original name>[_x<X>_y<Y>].jpg so results can be
    broken down by source.

Progress is banked per source image to <out>/manifest.jsonl — re-running resumes.

Usage:
    python3 flatbug_to_yolo_obb.py \
        --flatbug /Users/automeris/Desktop/Megabug/flatbug-dataset \
        --review  /Users/automeris/Desktop/Megabug/flatbug_missed \
        --out     /Users/automeris/Desktop/Megabug/flatbug_yolo_obb
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from flatbug_find_missed import load_short_codes, load_subset  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

OTHER_BACKGROUNDS = {"NHM-beetles-crops", "DIRT", "PeMaToEuroPep", "ArTaxOr"}
MODEL_INPUT = 1600


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def clip_to_window(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman: clip a polygon (N,2) to an axis-aligned window."""
    def clip(pts, inside, cross):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            if inside(b):
                if not inside(a):
                    out.append(cross(a, b))
                out.append(b)
            elif inside(a):
                out.append(cross(a, b))
        return out

    def at_x(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def at_y(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    pts = [tuple(p) for p in poly]
    for inside, cross in ((lambda p: p[0] >= x0, at_x(x0)), (lambda p: p[0] <= x1, at_x(x1)),
                          (lambda p: p[1] >= y0, at_y(y0)), (lambda p: p[1] <= y1, at_y(y1))):
        if not pts:
            break
        pts = clip(pts, inside, cross)
    return np.array(pts, np.float32).reshape(-1, 2)


def even_windows(W, H, tile, overlap):
    """Fewest evenly spaced tiles of about `tile` px per axis, neighbours overlapping
    by `overlap` of the image side. Tile size flexes so no two tiles nearly coincide."""
    def axis(n_px):
        n = int(np.ceil(n_px / tile))
        if n <= 1:
            return [(0, n_px)]
        side = min(n_px, int(np.ceil(n_px / n * (1 + overlap))))
        return [(int(round(s)), int(round(s)) + side) for s in np.linspace(0, n_px - side, n)]
    return [(x0, y0, x1, y1) for y0, y1 in axis(H) for x0, x1 in axis(W)]


def area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def label_lines(polys, window, min_visible):
    """OBB + seg label lines for one output image (window = x0,y0,x1,y1)."""
    x0, y0, x1, y1 = window
    W, H = x1 - x0, y1 - y0
    obb, seg = [], []
    for poly in polys:
        full = area(poly)
        clipped = clip_to_window(poly, x0, y0, x1, y1)
        if len(clipped) < 3 or area(clipped) < min_visible * max(full, 1e-6):
            continue
        local = (clipped - [x0, y0]).astype(np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(local)) / [W, H]
        box = np.clip(box, 0, 1)  # Ultralytics rejects a whole label file with any coord outside 0..1
        obb.append("0 " + " ".join(f"{v:.6f}" for v in box.ravel()))
        seg.append("0 " + " ".join(f"{v:.6f}" for v in np.clip(local / [W, H], 0, 1).ravel()))
    return obb, seg


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def load_added(review_dir):
    """{(subset, file): [pts (4,2)]} for detections marked creature in review."""
    dec = json.loads((review_dir / "decisions.json").read_text())
    added = defaultdict(list)
    for line in open(review_dir / "detections.jsonl"):
        r = json.loads(line)
        for d in r["dets"]:
            if not d["matched"] and dec.get(d.get("id"), {}).get("label") == "insect":
                added[(r["subset"], r["file"])].append(np.array(d["pts"], np.float32))
    return added


def subset_tile_size(anns_by_image, target_px):
    """Tile side (px) giving a median creature of target_px at the model input, or None."""
    sizes, longest = [], []
    for im, anns in anns_by_image:
        longest.append(max(im["width"], im["height"]))
        sizes += [max(a["bbox"][2], a["bbox"][3]) for a in anns]
    if not sizes:
        return None
    at_input = np.median(sizes) * MODEL_INPUT / np.median(longest)
    if at_input >= target_px:
        return None
    return int(np.median(sizes) * MODEL_INPUT / target_px)


def safe_stem(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("_")


def convert_image(job, args):
    subset, code, im, anns, extra, tile, group = job
    root = args.out / group
    src = args.flatbug / subset / im["file_name"]
    polys = [np.array(s, np.float32).reshape(-1, 2) for a in anns for s in a["segmentation"][:1] if len(s) >= 6]
    polys += extra
    W, H = im["width"], im["height"]
    base = f"FB-{code}_{safe_stem(im['file_name'])}"

    img = Image.open(src)
    rotated = img.getexif().get(274) not in (None, 1)
    windows = [(0, 0, W, H)] if tile is None else even_windows(W, H, tile, args.overlap)
    if rotated or len(windows) > 1:
        img = ImageOps.exif_transpose(img).convert("RGB")
        assert img.size == (W, H), f"{subset}/{im['file_name']}: {img.size} vs json {(W, H)}"

    outputs = []
    for (x0, y0, x1, y1) in windows:
        name = base if len(windows) == 1 else f"{base}_x{x0}_y{y0}"
        obb, seg = label_lines(polys, (x0, y0, x1, y1), args.min_visible)
        img_out = root / "images" / f"{name}.jpg"
        if len(windows) == 1 and not rotated:
            # Untouched bytes: hard-link (no extra disk space), copy if on another volume.
            img_out.unlink(missing_ok=True)
            try:
                os.link(src, img_out)
            except OSError:
                shutil.copyfile(src, img_out)
        else:
            img.crop((x0, y0, x1, y1)).save(img_out, quality=95)
        (root / "labels" / f"{name}.txt").write_text("\n".join(obb) + ("\n" if obb else ""))
        (root / "labels_seg" / f"{name}.txt").write_text("\n".join(seg) + ("\n" if seg else ""))
        outputs.append({"name": name, "window": [x0, y0, x1, y1], "boxes": len(obb)})
    return {"subset": subset, "file": im["file_name"], "group": group, "tile": tile,
            "flatbug_labels": len(anns), "added_labels": len(extra), "outputs": outputs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flatbug", required=True, type=Path)
    ap.add_argument("--review", required=True, type=Path, help="flatbug_find_missed.py output (decisions.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-px", type=float, default=25,
                    help="tile subsets whose median creature is smaller than this at imgsz 1600")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-visible", type=float, default=0.25,
                    help="keep a tile-cut creature if this share of its outline is inside")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subsets", nargs="*", help="only these subset folders (testing)")
    args = ap.parse_args()

    for group in ("flat", "other"):
        for sub in ("images", "labels", "labels_seg"):
            (args.out / group / sub).mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in open(manifest):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line mid-write
            done.add((r["subset"], r["file"]))

    codes = load_short_codes(args.flatbug)
    added = load_added(args.review)
    print(f"{sum(len(v) for v in added.values())} reviewed creatures to add")

    jobs, skipped_empty = [], 0
    subsets = sorted(p.name for p in args.flatbug.iterdir() if (p / "instances_default.json").exists())
    if args.subsets:
        subsets = [s for s in subsets if s in args.subsets]
    for subset in subsets:
        items = load_subset(args.flatbug / subset)
        group = "other" if subset in OTHER_BACKGROUNDS else "flat"
        tile = subset_tile_size([(im, a) for im, a in items if a], args.target_px)
        n = 0
        for im, anns in items:
            if not anns:
                skipped_empty += 1
                continue
            n += 1
            if (subset, im["file_name"]) not in done:
                extra = added.get((subset, im["file_name"]), [])
                jobs.append((subset, codes.get(subset, subset[:3].upper()), im, anns, extra, tile, group))
        print(f"  {subset:26s} {group:5s} {n:5d} images" + (f"  tiled at {tile}px" if tile else ""))
    print(f"{len(done)} images already converted, {len(jobs)} to go "
          f"({skipped_empty} unannotated images skipped)", flush=True)

    t0 = time.time()
    with open(manifest, "a") as log, ThreadPoolExecutor(args.workers) as pool:
        for k, rec in enumerate(pool.map(lambda j: convert_image(j, args), jobs), 1):
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if k % 200 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  [{k}/{len(jobs)}] {rec['subset']:24s} ETA {el / k * (len(jobs) - k) / 60:5.1f} min",
                      flush=True)

    summarize(manifest)


def summarize(manifest):
    stats = defaultdict(lambda: [0, 0, 0, 0])
    for line in open(manifest):
        r = json.loads(line)
        s = stats[(r["group"], r["subset"])]
        s[0] += len(r["outputs"]); s[1] += sum(o["boxes"] for o in r["outputs"])
        s[2] += r["flatbug_labels"]; s[3] += r["added_labels"]
    print(f"\n{'group':6s} {'subset':26s} out-imgs  boxes  (flatbug + added)")
    for group in ("flat", "other"):
        tot = [0, 0, 0, 0]
        for (g, s), v in sorted(stats.items()):
            if g == group:
                print(f"{g:6s} {s:26s} {v[0]:7d} {v[1]:7d}  ({v[2]} + {v[3]})")
                tot = [a + b for a, b in zip(tot, v)]
        print(f"{group:6s} {'TOTAL':26s} {tot[0]:7d} {tot[1]:7d}  ({tot[2]} + {tot[3]})\n")


if __name__ == "__main__":
    sys.exit(main())
