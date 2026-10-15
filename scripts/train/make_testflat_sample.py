"""
make_testflat_sample.py

Builds the hard-linked source folders for TestFlat_150MB, a ~150 MB sample of
the Flat set used to test the whole cluster workflow before real training:
6 Mothbox photos (3-60 boxes each) + 2 random images from each of the 19
flat-bug "flat" subsets (images < 6 MB where possible). seed 150.

Then run collect_yolo_training_data.py on the two folders it makes:
    python3 collect_yolo_training_data.py \
        --source <out>/mothbox --source <out>/flatbug_flat --output TestFlat_150MB

(Run 2026-10-14 against 2026_MOTHBOTYOLO_FULL; the Mothbox set used for real
training is 2026_09_17_YOLOOBB_TRAINING / Zenodo MBD-1.1.)
"""

import argparse
import collections
import os
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mothbox", required=True, type=Path, help="YOLO OBB folder with images/<split>, labels/<split>")
    ap.add_argument("--flatbug-flat", required=True, type=Path, help="flatbug_yolo_obb/flat")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-mothbox", type=int, default=6)
    ap.add_argument("--per-subset", type=int, default=2)
    ap.add_argument("--seed", type=int, default=150)
    args = ap.parse_args()
    random.seed(args.seed)
    for d in ("mothbox/images", "mothbox/labels", "flatbug_flat/images", "flatbug_flat/labels"):
        (args.out / d).mkdir(parents=True, exist_ok=True)

    pairs = []
    for split in ("train", "val", "test"):
        for im in (args.mothbox / "images" / split).iterdir():
            lb = args.mothbox / "labels" / split / (im.stem + ".txt")
            if lb.exists() and 3 <= len(lb.read_text().split("\n")) <= 60:
                pairs.append((im, lb))
    for im, lb in random.sample(pairs, args.n_mothbox):
        os.link(im, args.out / "mothbox/images" / im.name)
        os.link(lb, args.out / "mothbox/labels" / lb.name)

    by_code = collections.defaultdict(list)
    for im in (args.flatbug_flat / "images").iterdir():
        by_code[im.name.split("_")[0]].append(im)
    for code, ims in sorted(by_code.items()):
        small = [i for i in ims if i.stat().st_size < 6e6]
        if len(small) < args.per_subset:
            small = ims
        for im in random.sample(small, min(args.per_subset, len(small))):
            os.link(im, args.out / "flatbug_flat/images" / im.name)
            os.link(args.flatbug_flat / "labels" / (im.stem + ".txt"), args.out / "flatbug_flat/labels" / (im.stem + ".txt"))


if __name__ == "__main__":
    main()
