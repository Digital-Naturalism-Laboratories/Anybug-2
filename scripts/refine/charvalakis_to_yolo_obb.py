"""
charvalakis_to_yolo_obb.py

Converts a hand-picked slice of the Charvalakis et al. sticky-sheet time-lapse
(Google Drive multi-part zips, 20241023_Config_file/{train,val}/{images,labels})
into a Mothbox-style YOLO OBB folder for the FLAT set:

    <out>/flat/images/CHV-<trap>-<L|D>-<date>__t<HHMMSS>.jpg
    <out>/flat/labels/...txt      class 0 (insect) as unrotated OBB, class 1 (debris) dropped
    <out>/manifest.jsonl

Selection (Andrew, 2026-10-16: labels are good, image quality is low, so take only a
handful for variety):
  - infrared frames dropped;
  - one frame per trap-hour (the sharpest, by charvalakis_focus_rank.py, which scores
    sharpness inside the authors' insect boxes);
  - only the top --top-frac of those trap-hour scenes by sharpness (default 7%);
  - at most --per-day frames per trap-day (sticky sheets change slowly, so frames
    from the same night are near-duplicates), keeping the sharpest.

The "__t<time>" suffix makes collect_yolo_training_data.py keep all frames of one
trap-day in the same split. Images are shrunk to --max-side (default 2048; training
runs at 1600 and Ultralytics resizes on load anyway).

Usage:
    python3 charvalakis_to_yolo_obb.py \
        --zips "/Users/automeris/Desktop/Mushbug_Datasets/20241023_Config_file-20261016T144408Z-1-*.zip" \
        --frames /Users/automeris/Desktop/Megabug/charvalakis_focus_review/frames.jsonl \
        --out    /Users/automeris/Desktop/Megabug/charvalakis_yolo_obb
"""

import argparse
import collections
import glob
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", required=True)
    ap.add_argument("--frames", required=True, type=Path, help="frames.jsonl from charvalakis_focus_rank.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-frac", type=float, default=0.07)
    ap.add_argument("--per-day", type=int, default=2)
    ap.add_argument("--max-side", type=int, default=2048)
    args = ap.parse_args()

    frames = [json.loads(l) for l in open(args.frames)]
    best = {}
    for r in frames:
        if r.get("sharp") is None or r["trap"] == "?":
            continue
        k = (r["trap"], r["light"], r["date"], r["time"][:2])
        if k not in best or r["sharp"] > best[k]["sharp"]:
            best[k] = r
    scenes = sorted(best.values(), key=lambda r: -r["sharp"])
    top = scenes[:round(len(scenes) * args.top_frac)]
    per_day, chosen = collections.Counter(), []
    for r in top:  # sharpest first
        day = (r["trap"], r["light"], r["date"])
        if per_day[day] < args.per_day:
            per_day[day] += 1
            chosen.append(r)
    print(f"{len(scenes)} trap-hour scenes -> top {args.top_frac:.0%} = {len(top)} -> "
          f"{len(chosen)} after max {args.per_day}/trap-day ({len(per_day)} trap-days)", flush=True)

    zips = sorted(glob.glob(args.zips))
    where = {n: z for z in zips for n in zipfile.ZipFile(z).namelist()}
    zc = {}
    root = args.out / "flat"
    for sub in ("images", "labels"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    with open(args.out / "manifest.jsonl", "w") as man:
        for r in chosen:
            img_n = f"20241023_Config_file/{r['split']}/images/{r['name']}"
            lab_n = img_n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            zi = zc.setdefault(where[img_n], zipfile.ZipFile(where[img_n]))
            zl = zc.setdefault(where[lab_n], zipfile.ZipFile(where[lab_n]))
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(zi.read(img_n)))).convert("RGB")
            img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
            name = f"CHV-{r['trap']}-{r['light']}-{r['date']}__t{r['time']}"
            img.save(root / "images" / f"{name}.jpg", quality=95)
            lines = []
            for row in zl.read(lab_n).decode().splitlines():
                p = row.split()
                if len(p) == 5 and p[0] == "0":
                    cx, cy, w, h = map(float, p[1:])
                    x0, y0, x1, y1 = max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)
                    lines.append(f"0 {x0:.6f} {y0:.6f} {x1:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x0:.6f} {y1:.6f}")
            (root / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
            n_boxes += len(lines)
            man.write(json.dumps({"name": name, "source": img_n, "sharp_pct": r["sharp_pct"],
                                  "insects": len(lines)}) + "\n")
    print(f"wrote {len(chosen)} images, {n_boxes} insect boxes -> {root}")


if __name__ == "__main__":
    sys.exit(main())
