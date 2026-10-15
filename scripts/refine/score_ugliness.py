"""
score_ugliness.py

Adds Mothbot Process's "ugly score" (core/blur.py blur_fields) to every
unmatched detection found by flatbug_find_missed.py, so review_missed.py can
sort candidates sharpest-first or ugliest-first.

Each detection is cut from the original photo exactly as Process's detect step
cuts its patches (_crop_obb_fast), so the scores are on the same scale as the
ones shown in Mothbot Classify. The saved review crops can't be used: they
include context and have the pink box drawn on them.

Progress is banked per image to <dir>/ugliness.jsonl — re-running resumes.

Usage:
    python3 score_ugliness.py \
        --dir     /Users/automeris/Desktop/Megabug/flatbug_missed \
        --flatbug /Users/automeris/Desktop/Megabug/flatbug-dataset \
        --process /Users/automeris/Documents/GitHub/Mothbot_Process
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def score_image(args, rec):
    from core.blur import blur_fields
    from pipeline.detect import _crop_obb_fast

    rel = rec.get("path") or f'{rec["subset"]}/{rec["file"]}'  # older flat-bug runs have no "path"
    img = ImageOps.exif_transpose(Image.open(args.data / rel)).convert("RGB")
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    out = {}
    for d in rec["dets"]:
        if d["matched"]:
            continue
        f = blur_fields(_crop_obb_fast(bgr, d["pts"]))
        out[d["id"]] = {"ugly": f["blur_score"], "ugly_winner": f["blur_winner"],
                        "ugly_type": f["blur_type"]}
    return rec["subset"], rec["file"], out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path, help="output folder of flatbug_find_missed.py")
    ap.add_argument("--data", "--flatbug", dest="data", required=True, type=Path,
                    help="dataset root the detection run used")
    ap.add_argument("--process", required=True, help="Mothbot_Process repo (for core.blur)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    sys.path.insert(0, args.process)
    from core.blur import BLUR_METHOD

    out_path = args.dir / "ugliness.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line mid-write
            if r.get("method") == BLUR_METHOD:
                done.add((r["subset"], r["file"]))

    work = []
    for line in open(args.dir / "detections.jsonl"):
        r = json.loads(line)
        if (r["subset"], r["file"]) not in done and any(not d["matched"] for d in r["dets"]):
            work.append(r)
    total = len(work)
    print(f"{len(done)} images already scored, {total} to go ({BLUR_METHOD})", flush=True)

    t0, n = time.time(), 0
    with open(out_path, "a") as log, ThreadPoolExecutor(args.workers) as pool:
        for k, (subset, file, scores) in enumerate(pool.map(lambda r: score_image(args, r), work), 1):
            log.write(json.dumps({"subset": subset, "file": file, "method": BLUR_METHOD,
                                  "scores": scores}) + "\n")
            log.flush()
            n += len(scores)
            if k % 50 == 0 or k == total:
                el = time.time() - t0
                print(f"  [{k}/{total}] {subset:24s} {n} patches  "
                      f"ETA {el / k * (total - k) / 60:5.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
