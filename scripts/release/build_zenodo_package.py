"""
build_zenodo_package.py

Assembles the openly shareable Anybug-2 release folder from the local Megabug
working files (hard links, so it takes no extra disk until zipped):

  data/Anybug2-Open-core/       Mothbox (fixed labels) + flat-bug + open-licence Fieldguide + YOLO-Fast
                                + Charvalakis sticky sheets,
                                images/{train,val,test}, labels/..., data.yaml
  metadata/manifest.csv         one row per image: split, file, source, subset, group, licence,
                                attribution/credit, original id/DOI, number of boxes
  metadata/fieldguide_open_attribution.csv, metadata/flatbug_subsets.csv
  metadata/stats.json           counts and box-size statistics

The split of every image is exactly the one used to train and test the released
models (Mixed_Open_v1), so results are reproducible.

Usage:
    python3 build_zenodo_package.py --megabug ~/Desktop/Megabug --out ~/Desktop/Megabug/zenodo_Anybug-2 \
        --flatbug-licences flatbug_subset_licences.json
"""

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

OPEN_FG = {"cc0", "cc-by", "cc-by-sa"}
FLAT_SUBSETS_OTHER = {"NHM-beetles-crops", "DIRT", "PeMaToEuroPep", "ArTaxOr"}


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--megabug", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--flatbug-licences", required=True, type=Path, help="JSON from the Zenodo API per subset")
    args = ap.parse_args()
    M, O = args.megabug, args.out

    src = M / "Mixed_Open_v1"
    fb_lic = {r["short_name"]: r for r in json.loads(args.flatbug_licences.read_text())}
    code_to_subset = {r["short_name"]: r["subset"] for r in fb_lic.values()}
    fb_man = {}
    for line in open(M / "flatbug_yolo_obb/manifest.jsonl"):
        r = json.loads(line)
        for o in r["outputs"]:
            fb_man[o["name"]] = (r["subset"], r["file"], o["window"])
    fg_man = {json.loads(l)["name"]: json.loads(l) for l in open(M / "fieldguide_yolo_obb/manifest.jsonl")}
    fg_idx = {json.loads(l)["image_key"]: json.loads(l) for l in open(M / "fg-arthropods-v1/index.jsonl")}
    yf_src = {json.loads(l)["name"]: json.loads(l)["source"] for l in open(M / "yolofast_yolo_obb/manifest.jsonl")}

    rows, fg_attr = [], []
    for split in ("train", "val", "test"):
        for img in sorted((src / "images" / split).iterdir()):
            stem, lbl = img.stem, src / "labels" / split / f"{img.stem}.txt"
            boxes = [l for l in lbl.read_text().splitlines() if l.strip()]
            if stem.startswith("YF_"):
                part, row = "data/Anybug2-Open-core", dict(
                    source="YOLO-Fast (Entomoscope 2.0)", subset="", group="other",
                    licence="CC-BY-4.0", credit="Shirali, Hossein",
                    original=f"YOLO-Fast-dataset.zip: {yf_src[stem]}", doi="10.5281/zenodo.21892852")
            elif stem.startswith("CHV-"):
                # licensed CC-BY-4.0 by the authors for this release (email to Andrew, 2026-11)
                part, row = "data/Anybug2-Open-core", dict(
                    source="Charvalakis et al. 2026 (sticky sheets)", subset=stem.split("-")[1], group="flat",
                    licence="CC-BY-4.0", credit="Charvalakis, Spoelstra et al. (NIOO-KNAW)",
                    original=f"time-lapse frame {stem[4:]} (from the authors)", doi="10.1002/3070-5436.70004")
            elif stem.startswith("FB-"):
                code = stem[3:].split("_", 1)[0]
                subset, orig, window = fb_man.get(stem, (code_to_subset.get(code, code), "", None))
                lic = fb_lic.get(code, {})
                part, row = "data/Anybug2-Open-core", dict(
                    source="flat-bug", subset=subset, group="other" if subset in FLAT_SUBSETS_OTHER else "flat",
                    licence=(lic.get("license") or "cc-by-4.0").upper(), credit=lic.get("creators", ""),
                    original=orig + (f" (tile {window})" if window and stem.endswith(tuple("0123456789")) and "_x" in stem else ""),
                    doi=lic.get("doi", ""))
            elif stem.startswith("FG-"):
                m = fg_man[stem]
                ix = fg_idx[m["image_key"]]
                if m["license_class"] not in OPEN_FG:
                    raise SystemExit(f"non-open Fieldguide photo in Mixed_Open_v1: {stem}")
                part, row = "data/Anybug2-Open-core", dict(
                    source="Fieldguide (fg-arthropods-v1)", subset=m["group"], group="other",
                    licence=m["license_class"].upper(), credit=ix.get("license") or "",
                    original=f"photo_id {ix['photo_id']} (stored at source size)", doi="")
                fg_attr.append(dict(file=f"{img.name}", photo_id=ix["photo_id"], licence_class=m["license_class"],
                                    licence_text=ix.get("license") or "", taxon=ix.get("taxon_name") or "",
                                    order=ix.get("order") or "", sha256=ix["sha256"]))
            else:
                part, row = "data/Anybug2-Open-core", dict(
                    source="Mothbox Training Dataset 1-1 (labels corrected)", subset="", group="flat",
                    licence="CC-BY-4.0", credit="Quitmeyer, Kit; Quitmeyer, Andrew (Mothbox)",
                    original=f"{img.name} (full resolution)", doi="10.5281/zenodo.23140320")
            link(img, O / part / "images" / split / img.name)
            link(lbl, O / part / "labels" / split / lbl.name)
            rows.append(dict(split=split, file=img.name, part=part, boxes=len(boxes), **row))

    for part in ("data/Anybug2-Open-core",):
        (O / part / "data.yaml").write_text(
            "# Ultralytics YOLO OBB dataset; no 'path:' so it works wherever it is unpacked\n"
            "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: creature\n")

    meta = O / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    with open(meta / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(meta / "fieldguide_open_attribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fg_attr[0].keys()))
        w.writeheader(); w.writerows(fg_attr)
    with open(meta / "flatbug_subsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "short_name", "doi", "license", "creators"])
        w.writeheader(); w.writerows(fb_lic.values())

    stats = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        stats[(r["part"], r["source"])]["images_" + r["split"]] += 1
        stats[(r["part"], r["source"])]["boxes_" + r["split"]] += r["boxes"]
    # box long side at the 1600 px training scale
    sizes = collections.defaultdict(list)
    for r in rows:
        if r["part"] != "data/Anybug2-Open-core":
            continue
        for l in (O / r["part"] / "labels" / r["split"] / (Path(r["file"]).stem + ".txt")).read_text().splitlines():
            p = l.split()
            if len(p) == 9:
                q = np.array(list(map(float, p[1:]))).reshape(4, 2)
                e = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]))
                sizes[r["source"]].append(e * 1600)
    out = {"by_part_and_source": {f"{k[0]} | {k[1]}": dict(v) for k, v in stats.items()},
           "box_long_side_px_at_1600_approx": {k: {"p10": round(float(np.percentile(v, 10))), "median": round(float(np.median(v))),
                                                   "p90": round(float(np.percentile(v, 90)))} for k, v in sizes.items()}}
    (meta / "stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
