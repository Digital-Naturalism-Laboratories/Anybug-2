"""
fieldguide_sam_review_prep.py

Turns the kept SAM boxes from fieldguide_sam_boxes.py into a review folder
that review_missed.py can open: one tile per photo, centred on the user's
square, with the user square in green, SAM's outline in yellow and the
rotated box that would become the label in pink. With --missed, the caption
says "⚠ N extra?" when the Mothbox detector found confident insects outside the
square, and the shift-click view (the whole photo) circles them in cyan.

In review: green (click) = good box, red (right-click) = bad box.

Crops already written are skipped, so re-running resumes (--force redraws them).

Usage:
    python3 fieldguide_sam_review_prep.py \
        --sam  /Users/automeris/Desktop/Megabug/fieldguide_sam \
        --data /Users/automeris/Desktop/Megabug/fg-arthropods-v1 \
        --out  /Users/automeris/Desktop/Megabug/fieldguide_sam_review \
        --missed /Users/automeris/Desktop/Megabug/fieldguide_missed
    python3 review_missed.py --dir /Users/automeris/Desktop/Megabug/fieldguide_sam_review \
        --cols 5 --max-rows 2 --max-size 100000 --title "Fieldguide SAM box review" \
        --legend "Judge the <b style='color:#ff28c8'>pink box</b>: does it fit the insect? (yellow = SAM outline, green = user's square) · click = good box (green) · right-click = bad box (red) · Enter = save marks · A / N = mark rest good / bad · U = undo · ←/→ pages"
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from flatbug_find_missed import FIELDGUIDE_CODES  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def tile(img, rec, pad, out_px):
    """Crop around the user square (padded by `pad` of its side) with all three outlines drawn."""
    x0, y0, x1, y1 = rec["square"]
    p = pad * max(x1 - x0, y1 - y0)
    c = (int(x0 - p), int(y0 - p), int(x1 + p), int(y1 + p))
    t = img.crop(c).resize((out_px, out_px), Image.LANCZOS)
    s = out_px / (c[2] - c[0])
    to = lambda pts: [((x - c[0]) * s, (y - c[1]) * s) for x, y in pts]
    d = ImageDraw.Draw(t)
    d.rectangle([(x0 - c[0]) * s, (y0 - c[1]) * s, (x1 - c[0]) * s, (y1 - c[1]) * s], outline=(0, 230, 110), width=1)
    d.line(to(rec["outline"] + rec["outline"][:1]), fill=(255, 220, 0), width=1)
    d.polygon(to(rec["obb"]), outline=(255, 40, 200), width=6)
    return t


def full_view(img, rec, extras, out_px=1400, square=False):
    """Whole photo (long side out_px) with the square, SAM outline, pink box and,
    in cyan, the detector's confident finds outside the square (possible unlabelled insects).
    square=True pads it onto a black out_px x out_px canvas (for grid thumbnails)."""
    s = out_px / max(img.size)
    lw = max(2, round(6 * out_px / 1400))
    t = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    to = lambda pts: [(x * s, y * s) for x, y in pts]
    d = ImageDraw.Draw(t)
    x0, y0, x1, y1 = rec["square"]
    d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(0, 230, 110), width=2)
    d.line(to(rec["outline"] + rec["outline"][:1]), fill=(255, 220, 0), width=2)
    d.polygon(to(rec["obb"]), outline=(255, 40, 200), width=max(lw, 3))
    for pts in extras:
        cx, cy = np.mean(pts, axis=0) * s
        r = max(14, 0.75 * max(np.ptp(np.array(pts), axis=0)) * s)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 220, 255), width=max(lw, 3))
    if square:
        canvas = Image.new("RGB", (out_px, out_px))
        canvas.paste(t, ((out_px - t.width) // 2, (out_px - t.height) // 2))
        return canvas
    return t


def load_extras(missed_dir, min_conf, min_px, max_total):
    """{image_key: [pts]} of confident detections outside the user square, from a
    flatbug_find_missed.py --format fieldguide run. Photos with more than max_total
    unmatched detections are left out: those are background texture (mesh, stucco)."""
    extras = {}
    for line in open(missed_dir / "detections.jsonl"):
        r = json.loads(line)
        unmatched = [d for d in r["dets"] if not d["matched"]]
        if len(unmatched) <= max_total:
            extras[r["file"]] = [d["pts"] for d in unmatched if d["conf"] >= min_conf and d["long"] >= min_px]
    return extras


def prep(args, rec, meta, extras):
    code = FIELDGUIDE_CODES.get(rec["group"], "FG")
    cid = f"FGSAM-{code}-{hashlib.sha1(rec['image_key'].encode()).hexdigest()[:10]}"
    small, big = args.out / "crops" / f"{cid}_s.jpg", args.out / "crops" / f"{cid}_b.jpg"
    if args.force or not (small.exists() and big.exists()):
        img = ImageOps.exif_transpose(Image.open(args.data / rec["image_key"])).convert("RGB")
        # Whole photo as the thumbnail (Andrew 2026-10-15): unlabelled insects anywhere in the
        # frame must be visible in the grid, not just around the user's square.
        full_view(img, rec, extras, 480, square=True).save(small, quality=85)
        full_view(img, rec, extras).save(big, quality=85)
    box = np.array(rec["obb"])
    e1, e2 = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])
    return {"subset": rec["group"], "code": code, "file": rec["image_key"], "path": rec["image_key"],
            # "conf" only orders the review page (its default sort is confidence ↓):
            # photos flagged with possible extra insects come first, most extras first.
            "dets": [{"id": cid, "pts": rec["obb"], "conf": 0.9 + 0.01 * min(len(extras), 9) if extras else 0.5,
                      "matched": False,
                      "long": round(float(max(e1, e2)), 1), "short": round(float(min(e1, e2)), 1),
                      "caption": (f"⚠ {len(extras)} extra? · " if extras else "")
                                 + f"{meta.get('taxon_name') or '?'} · {rec['pieces']}pc · {rec['ratio']:.2f}"}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam", required=True, type=Path, help="fieldguide_sam_boxes.py output folder")
    ap.add_argument("--data", required=True, type=Path, help="fg-arthropods-v1 folder")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="redraw crops that already exist")
    ap.add_argument("--missed", type=Path, help="flatbug_find_missed.py --format fieldguide output, "
                    "to flag photos where the detector found other insects")
    ap.add_argument("--extra-conf", type=float, default=0.5)
    ap.add_argument("--extra-px", type=float, default=20)
    ap.add_argument("--extra-max-total", type=int, default=20,
                    help="ignore photos with more unmatched detections than this (texture)")
    args = ap.parse_args()

    (args.out / "crops").mkdir(parents=True, exist_ok=True)
    meta = {r["image_key"]: r for r in map(json.loads, open(args.data / "index.jsonl"))}
    extras = load_extras(args.missed, args.extra_conf, args.extra_px, args.extra_max_total) if args.missed else {}
    kept = [r for r in map(json.loads, open(args.sam / "sam_boxes.jsonl")) if r["keep"]]
    print(f"{len(kept)} kept SAM boxes", flush=True)

    t0, out = time.time(), []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, rec in enumerate(pool.map(lambda r: prep(args, r, meta[r["image_key"]], extras.get(r["image_key"], [])), kept), 1):
            out.append(rec)
            if k % 500 == 0 or k == len(kept):
                el = time.time() - t0
                print(f"  [{k}/{len(kept)}] ETA {el / k * (len(kept) - k) / 60:4.1f} min", flush=True)
    # Written whole at the end (crops are the resumable part).
    tmp = args.out / "detections.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in out))
    tmp.replace(args.out / "detections.jsonl")


if __name__ == "__main__":
    sys.exit(main())
