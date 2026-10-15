"""
flatbug_find_missed.py

Runs a Mothbox OBB detector over the flat-bug dataset (or, with --format
fieldguide, the fg-arthropods Fieldguide set) and finds detections that do NOT
overlap any existing annotation — candidate creatures the annotators skipped
(flat-bug had a ~32 px minimum-size rule; Fieldguide has one square user-crop
box per photo, so every other creature in the frame is unlabelled).

For every unmatched detection it saves two review crops (tight + wide
context), with the detection drawn in pink and flat-bug's own outlines in
green. Review them with review_missed.py.

Large images (e.g. pinoy2023 at 14039x10200) are run as one full-frame pass
plus overlapping tiles, so small creatures are not shrunk away by imgsz.

Progress is banked per image to detections.jsonl — re-running the same
command resumes where it stopped.

Usage:
    python3 flatbug_find_missed.py \
        --flatbug /Users/automeris/Desktop/Megabug/flatbug-dataset \
        --model  /Users/automeris/Documents/GitHub/Mothbot_Process/trained_models/MBD-1-1.pt \
        --out    /Users/automeris/Desktop/Megabug/flatbug_missed

    # Quick test on a couple of subsets, 5 images each
    python3 flatbug_find_missed.py ... --subsets AMI-traps Diopsis --limit 5

    # Fieldguide (subsets = its groups; mollusk_worm skipped by default)
    python3 flatbug_find_missed.py --format fieldguide \
        --data  /Users/automeris/Desktop/Megabug/fg-arthropods-v1 \
        --model /Users/automeris/Documents/GitHub/Mothbot_Process/trained_models/MBD-1-1.pt \
        --out   /Users/automeris/Desktop/Megabug/fieldguide_missed
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_short_codes(flatbug):
    codes = {}
    meta = flatbug / "metadata.csv"
    if meta.exists():
        with open(meta, newline="") as f:
            for row in csv.DictReader(f):
                codes[row["dataset"]] = row["short_name"]
    return codes


def load_subset(subset_dir):
    """Return [(image_record, [annotations])] for images whose file exists."""
    coco = json.loads((subset_dir / "instances_default.json").read_text())
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    present = {p.name for p in subset_dir.iterdir()}
    return [(im, by_img.get(im["id"], []))
            for im in coco["images"] if im["file_name"] in present]


FIELDGUIDE_CODES = {"moth": "MOTH", "butterfly": "BFLY", "other_arthropod": "ARTH", "mollusk_worm": "MOLL"}


def load_fieldguide(root):
    """{group: [(image_record, [annotation])]} from a fg-arthropods index.jsonl.
    The one square user-crop box per photo becomes a flat-bug-style annotation;
    file_name is the image's path relative to root."""
    groups = {}
    for line in open(root / "index.jsonl"):
        r = json.loads(line)
        x0, y0, x1, y1 = r["bbox_xyxy"]
        ann = {"bbox": [x0, y0, x1 - x0, y1 - y0], "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]}
        im = {"file_name": r["image_key"], "width": r["width"], "height": r["height"]}
        groups.setdefault(r["group"], []).append((im, [ann]))
    return groups


def load_work(args):
    """[(subset, code, image path relative to --data, image_record, annotations)]"""
    if args.format == "fieldguide":
        groups = load_fieldguide(args.data)
        subsets = sorted(g for g in groups if g in (args.subsets or set(groups) - {"mollusk_worm"}))
        return [(g, FIELDGUIDE_CODES.get(g, g[:4].upper()), im["file_name"], im, anns)
                for g in subsets for im, anns in groups[g][:args.limit]]
    codes = load_short_codes(args.data)
    subsets = sorted(p.name for p in args.data.iterdir() if (p / "instances_default.json").exists())
    if args.subsets:
        subsets = [s for s in subsets if s in args.subsets]
    return [(s, codes.get(s, s[:3].upper()), f"{s}/{im['file_name']}", im, anns)
            for s in subsets for im, anns in load_subset(args.data / s)[:args.limit]]


# ---------------------------------------------------------------------------
# Inference (full frame + tiles)
# ---------------------------------------------------------------------------

def tile_windows(W, H, tile, overlap):
    """Overlapping tile x tile windows covering the image."""
    step = int(tile * (1 - overlap))

    def starts(n):
        if n <= tile:
            return [0]
        s = list(range(0, n - tile, step))
        return s + [n - tile]

    return [(x, y, min(x + tile, W), min(y + tile, H))
            for y in starts(H) for x in starts(W)]


def run_model(model, arrays, args):
    """Predict on a list of BGR arrays → list of (pts (N,4,2), conf (N,))."""
    out = []
    for i in range(0, len(arrays), args.batch):
        res = model.predict(arrays[i:i + args.batch], imgsz=args.imgsz,
                            conf=args.conf, max_det=args.max_det,
                            device=args.device, verbose=False)
        for r in res:
            if r.obb is None or len(r.obb) == 0:
                out.append((np.zeros((0, 4, 2)), np.zeros(0)))
            else:
                out.append((r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2),
                            r.obb.conf.cpu().numpy()))
    return out


def aabb(pts):
    return np.concatenate([pts.min(1), pts.max(1)], axis=1)


def nms(boxes, scores, iou_thr):
    import torch
    import torchvision
    keep = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32),
                               torch.tensor(scores, dtype=torch.float32), iou_thr)
    return keep.numpy()


def detect(model, rgb, args):
    H, W = rgb.shape[:2]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    windows = [(0, 0, W, H)]
    if max(W, H) > args.tile * args.tile_trigger:
        windows += tile_windows(W, H, args.tile, args.overlap)
    preds = run_model(model, [bgr[y0:y1, x0:x1] for x0, y0, x1, y1 in windows], args)

    all_pts, all_conf = [], []
    for wi, ((x0, y0, x1, y1), (pts, conf)) in enumerate(zip(windows, preds)):
        if len(conf) == 0:
            continue
        pts = pts + [x0, y0]
        if wi > 0:
            # Drop tile detections cut off by an interior tile edge — the
            # neighbouring tile (or the full-frame pass) sees them whole.
            b, m = aabb(pts), 3
            cut = (((b[:, 0] <= x0 + m) & (x0 > 0)) | ((b[:, 1] <= y0 + m) & (y0 > 0)) |
                   ((b[:, 2] >= x1 - m) & (x1 < W)) | ((b[:, 3] >= y1 - m) & (y1 < H)))
            pts, conf = pts[~cut], conf[~cut]
        all_pts.append(pts)
        all_conf.append(conf)
    if not all_pts:
        return np.zeros((0, 4, 2)), np.zeros(0), len(windows)
    pts, conf = np.concatenate(all_pts), np.concatenate(all_conf)
    if len(windows) > 1 and len(conf):
        keep = nms(aabb(pts), conf, 0.5)
        pts, conf = pts[keep], conf[keep]
    return pts, conf, len(windows)


# ---------------------------------------------------------------------------
# Matching against flat-bug annotations
# ---------------------------------------------------------------------------

def iou_matrix(a, b):
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(det_pts, anns, iou_thr):
    """Return (det_matched bool[N], gt_matched bool[M]).

    A detection counts as already annotated if its box overlaps a flat-bug
    box (IoU > iou_thr) or its centre lies inside a flat-bug outline."""
    N, M = len(det_pts), len(anns)
    det_m, gt_m = np.zeros(N, bool), np.zeros(M, bool)
    if N == 0 or M == 0:
        return det_m, gt_m
    gt_box = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                        a["bbox"][1] + a["bbox"][3]] for a in anns])
    polys = [[np.array(s, np.float32).reshape(-1, 2) for s in a["segmentation"] if len(s) >= 6]
             for a in anns]
    iou = iou_matrix(aabb(det_pts), gt_box)
    hit = iou > iou_thr
    centres = det_pts.mean(1)
    for i, (cx, cy) in enumerate(centres):
        inside = np.where((gt_box[:, 0] <= cx) & (cx <= gt_box[:, 2]) &
                          (gt_box[:, 1] <= cy) & (cy <= gt_box[:, 3]))[0]
        for j in inside:
            if any(cv2.pointPolygonTest(p, (float(cx), float(cy)), False) >= 0 for p in polys[j]):
                hit[i, j] = True
    return hit.any(1), hit.any(0)


# ---------------------------------------------------------------------------
# Review crops
# ---------------------------------------------------------------------------

def save_crop(img, pts, anns, side, out_px, path):
    cx, cy = pts.mean(0)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    crop = img.crop((x0, y0, x0 + int(side), y0 + int(side))).resize((out_px, out_px), Image.LANCZOS)
    s = out_px / int(side)
    d = ImageDraw.Draw(crop)
    for a in anns:
        bx, by, bw, bh = a["bbox"]
        if bx > x0 + side or by > y0 + side or bx + bw < x0 or by + bh < y0:
            continue
        for seg in a["segmentation"]:
            if len(seg) >= 6:
                d.polygon([((x - x0) * s, (y - y0) * s) for x, y in zip(seg[0::2], seg[1::2])],
                          outline=(0, 230, 110), width=1)
    d.polygon([((x - x0) * s, (y - y0) * s) for x, y in pts], outline=(255, 40, 200), width=2)
    crop.save(path, quality=85)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--flatbug", dest="data", required=True, type=Path,
                    help="dataset root (flat-bug folder, or fg-arthropods-v1 with --format fieldguide)")
    ap.add_argument("--format", choices=["flatbug", "fieldguide"], default="flatbug")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subsets", nargs="*", help="only these subset folders")
    ap.add_argument("--limit", type=int, help="max images per subset (testing)")
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max-det", type=int, default=3000)
    ap.add_argument("--tile", type=int, default=3200, help="tile side in original pixels")
    ap.add_argument("--tile-trigger", type=float, default=1.25,
                    help="tile images whose long side exceeds tile*this")
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    crops = args.out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "detections.jsonl"
    done = set()
    if log_path.exists():
        for line in open(log_path):
            r = json.loads(line)
            done.add((r["subset"], r["file"]))
    (args.out / "run_args.json").write_text(json.dumps({**vars(args), "data": str(args.data),
                                                        "out": str(args.out)}, indent=1))

    prefix = "FG" if args.format == "fieldguide" else "FB"
    work = [w for w in load_work(args) if (w[0], w[3]["file_name"]) not in done]
    total = len(work)
    print(f"{len(done)} images already banked, {total} to go "
          f"(model {Path(args.model).name}, imgsz {args.imgsz}, conf {args.conf})")

    t0, n_cand = time.time(), 0
    with open(log_path, "a") as log:
        for k, (subset, code, rel, im, anns) in enumerate(work, 1):
            path = args.data / rel
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            if img.size != (im["width"], im["height"]):
                print(f"  ! size mismatch {subset}/{im['file_name']} {img.size} vs json — skipped")
                continue
            pts, conf, n_win = detect(model, np.asarray(img), args)
            det_m, gt_m = match(pts, anns, args.match_iou)

            key = hashlib.sha1(f"{subset}/{im['file_name']}".encode()).hexdigest()[:8]
            dets = []
            for i in range(len(conf)):
                p = pts[i]
                e1, e2 = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])
                d = {"pts": np.round(p, 1).tolist(), "conf": round(float(conf[i]), 3),
                     "long": round(float(max(e1, e2)), 1), "short": round(float(min(e1, e2)), 1),
                     "matched": bool(det_m[i])}
                if not det_m[i]:
                    d["id"] = f"{prefix}-{code}-{key}-{i:04d}"
                    # Cap context at the image size — full-frame passes on huge
                    # images can produce card-sized junk boxes.
                    L, cap = d["long"], max(img.size)
                    save_crop(img, p, anns, min(max(6 * L, 96), cap), 200, crops / f"{d['id']}_s.jpg")
                    save_crop(img, p, anns, min(max(18 * L, 480), cap), 720, crops / f"{d['id']}_b.jpg")
                    n_cand += 1
                dets.append(d)
            log.write(json.dumps({"subset": subset, "code": code, "file": im["file_name"], "path": rel,
                                  "w": im["width"], "h": im["height"], "windows": n_win,
                                  "n_gt": len(anns), "gt_found": int(gt_m.sum()),
                                  "dets": dets}) + "\n")
            log.flush()

            if k % 10 == 0 or k == total:
                el = time.time() - t0
                eta = el / k * (total - k)
                print(f"  [{k}/{total}] {subset:24s} {el/k:5.2f}s/img  "
                      f"candidates so far {n_cand}  ETA {eta/60:5.1f} min", flush=True)

    summarize(log_path)


def summarize(log_path):
    stats = {}
    for line in open(log_path):
        r = json.loads(line)
        s = stats.setdefault(r["subset"], [0, 0, 0, 0])
        s[0] += 1; s[1] += r["n_gt"]; s[2] += r["gt_found"]
        s[3] += sum(not d["matched"] for d in r["dets"])
    print(f"\n{'subset':26s} imgs   labels  found-by-model  unmatched-dets")
    for k in sorted(stats):
        n, g, f, u = stats[k]
        print(f"{k:26s} {n:5d} {g:7d}  {f/max(g,1):6.1%}         {u:6d}")


if __name__ == "__main__":
    sys.exit(main())
