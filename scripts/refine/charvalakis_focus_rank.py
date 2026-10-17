"""
charvalakis_focus_rank.py

Ranks the Charvalakis et al. sticky-sheet photos (Google Drive multi-part zips,
20241023_Config_file/{train,val}/{images,labels}) by how in-focus the SHEET is,
skipping infrared (grayscale) frames, and builds a review folder that
review_missed.py can open sorted sharpest-first:

  <id>_s.jpg   native-resolution crop from the middle of the sheet (judge focus here)
  <id>_b.jpg   the whole frame

Sharpness is measured ON THE INSECTS: for every class-0 (insect) box the
authors drew, the variance of the Laplacian of that box (padded 25%, grayscale,
light 3x3 blur so sensor/JPEG noise doesn't count), at half resolution; a frame's
score is the median over its insects. (A first version scored the middle of the
frame, but knocked-sideways cameras then ranked "sharpest" because the trees and
tray edges have lots of detail, and water droplets scored high too.) Frames with
no labelled insects can't be judged and are left out of the ranking. Each tile is
a native-resolution crop centred on the frame's densest cluster of insects.
A tile's "conf" is its sharpness percentile (1.0 = sharpest), so the page's
default sort shows the sharpest first.

In review (optional): green = usable focus, red = too blurry.

Reads straight from the zips (no extraction). Writes <out>/frames.jsonl (every
frame: trap, light/dark, time, IR flag, sharpness, label count) and the review files.

Usage:
    python3 charvalakis_focus_rank.py \
        --zips "/Users/automeris/Desktop/Mushbug_Datasets/20241023_Config_file-20261016T144408Z-1-*.zip" \
        --out  /Users/automeris/Desktop/Megabug/charvalakis_focus_review
"""

import argparse
import glob
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

NAME = re.compile(r"Ins_LMP_([A-Z]+\d*)_+([LD])_+(\d{8})_(\d{6})")


def is_infrared(img):
    a = np.asarray(img.resize((64, 48))).astype(float)
    return (np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 4


def insect_sharpness(img, boxes):
    """Median over insect boxes (normalised cx, cy, w, h) of Laplacian variance inside each box."""
    g = cv2.GaussianBlur(np.asarray(img.convert("L"), np.float32), (3, 3), 0)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    H, W = g.shape
    vals = []
    for cx, cy, w, h in boxes:
        bw, bh = max(w * W * 1.25, 6), max(h * H * 1.25, 6)
        x0, y0 = int(max(0, cx * W - bw / 2)), int(max(0, cy * H - bh / 2))
        x1, y1 = int(min(W, cx * W + bw / 2)), int(min(H, cy * H + bh / 2))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            vals.append(float(lap[y0:y1, x0:x1].var()))
    return float(np.median(vals)) if vals else None


def work(job):
    zpath, names, out, label_zip = job
    z, recs, zcache = zipfile.ZipFile(zpath), [], {}
    for n in names:
        data = z.read(n)
        img = Image.open(io.BytesIO(data))
        W, H = img.size
        img.draft("RGB", (W // 2, H // 2))  # fast half-size JPEG decode
        img = img.convert("RGB")
        lbl = n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        lz = label_zip.get(lbl)
        if lz is None:
            rows = []
        else:
            zl = zcache.setdefault(lz, zipfile.ZipFile(lz))
            rows = [r.split() for r in zl.read(lbl).decode().splitlines() if r.strip()]
        insects = [tuple(map(float, r[1:5])) for r in rows if r[0] == "0" and len(r) == 5]
        m = NAME.search(n.rsplit("/", 1)[-1])
        rec = {"name": n.rsplit("/", 1)[-1], "split": n.split("/")[1], "zip": Path(zpath).name, "w": W, "h": H,
               "trap": m.group(1) if m else "?", "light": m.group(2) if m else "?",
               "date": m.group(3) if m else "?", "time": m.group(4) if m else "?",
               "ir": bool(is_infrared(img)),
               "n_insects": len(insects), "n_other": sum(r[0] == "1" for r in rows)}
        rec["sharp"] = None if rec["ir"] else insect_sharpness(img, insects)
        if rec["sharp"] is not None:
            cid = "CHV-" + hashlib.sha1(n.encode()).hexdigest()[:12]
            rec["id"] = cid
            # native-resolution crop (900x900 native = 450x450 in the half-size decode),
            # centred on the median insect position
            cx = int(np.median([b[0] for b in insects]) * img.width)
            cy = int(np.median([b[1] for b in insects]) * img.height)
            half = 225
            img.crop((cx - half, cy - half, cx + half, cy + half)).resize((480, 480), Image.LANCZOS).save(
                out / "crops" / f"{cid}_s.jpg", quality=88)
            big = img.copy()
            big.thumbnail((1600, 1600))
            big.save(out / "crops" / f"{cid}_b.jpg", quality=85)
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", required=True, help="glob of the multi-part zips (one complete download)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()
    (args.out / "crops").mkdir(parents=True, exist_ok=True)

    zips = sorted(glob.glob(args.zips))
    label_zip = {n: zp for zp in zips for n in zipfile.ZipFile(zp).namelist() if n.endswith(".txt")}
    jobs = []
    for zp in zips:
        names = sorted(n for n in zipfile.ZipFile(zp).namelist() if n.lower().endswith(".jpg") and "/images/" in n)
        jobs += [(zp, names[i:i + args.chunk], args.out, label_zip) for i in range(0, len(names), args.chunk)]
    total = sum(len(j[1]) for j in jobs)
    print(f"{total} frames in {len(jobs)} chunks", flush=True)

    t0, recs = time.time(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(pool.map(work, jobs), 1):
            recs += r
            if k % 10 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  [{len(recs)}/{total}] ETA {el / len(recs) * (total - len(recs)) / 60:4.1f} min", flush=True)

    vis = sorted((r for r in recs if r.get("sharp") is not None), key=lambda r: r["sharp"])
    for i, r in enumerate(vis):
        r["sharp_pct"] = round((i + 1) / len(vis), 4)
    with open(args.out / "frames.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    with open(args.out / "detections.jsonl", "w") as f:
        for r in vis:
            ins = r["n_insects"]
            f.write(json.dumps({"subset": r["trap"], "code": r["trap"], "file": r["name"], "dets": [{
                "id": r["id"], "matched": False, "conf": r["sharp_pct"], "long": float(r["w"]), "short": float(r["h"]),
                "caption": f"sharp {r['sharp_pct']:.0%} ({r['sharp']:.0f}) · {r['trap']} {r['light']} "
                           f"{r['date']} {r['time'][:4]} · {ins} insects"}]}) + "\n")
    print(f"{len(recs)} frames: {sum(r['ir'] for r in recs)} infrared skipped, "
          f"{sum(1 for r in recs if not r['ir'] and r['sharp'] is None)} visible with no insects, "
          f"{len(vis)} ranked -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
