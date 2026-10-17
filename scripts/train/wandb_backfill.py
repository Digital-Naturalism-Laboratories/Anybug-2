"""
wandb_backfill.py

Uploads an already-finished Ultralytics training run folder to Weights & Biases,
for runs trained before W&B logging was switched on (no retraining needed):

  config   <- args.yaml (every training setting) + --notes/--tags
  history  <- results.csv, one step per epoch (losses, precision, recall, mAP, lr)
  media    <- the run's plots (PR/F1 curves, confusion matrix, results.png, labels.jpg, ...)
  summary  <- final/best epoch metrics, energy_report.json (if present) as energy/*,
              and any test-split scores given with --test
  artifact <- weights/best.pt (and best.onnx) with --weights

The run is tagged "backfilled" so it's clear it wasn't logged live.

Usage (on the GPU machine, after `wandb login --host=https://api.forge.coreweave.com`):
    python3 wandb_backfill.py --run /mnt/ami-training/andy/runs/flat_s1600 \
        --project anybug-2 --entity moth-ai --tags flat s1600 --weights \
        --test mothbox_test=0.897,0.892,0.937,0.673 flatbug_test=0.916,0.927,0.963,0.732
    (--test NAME=P,R,mAP50,mAP50-95; repeatable)
"""

import argparse
import csv
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="Ultralytics run folder (has results.csv, args.yaml)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--name", default=None, help="W&B run name (default: the folder name)")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--notes", default=None)
    ap.add_argument("--test", nargs="*", default=[], metavar="NAME=P,R,mAP50,mAP50-95")
    ap.add_argument("--weights", action="store_true", help="also upload weights/best.pt (+ best.onnx) as an artifact")
    args = ap.parse_args()

    import wandb
    import yaml

    run_dir = args.run
    config = yaml.safe_load((run_dir / "args.yaml").read_text()) if (run_dir / "args.yaml").exists() else {}
    with open(run_dir / "results.csv") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]

    run = wandb.init(entity=args.entity, project=args.project, name=args.name or run_dir.name,
                     job_type="train", tags=["backfilled", *args.tags], notes=args.notes,
                     config={**config, "backfilled_from": str(run_dir)})

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for r in rows:
        epoch = int(float(r.get("epoch", 0)))
        log = {k: x for k, v in r.items() if k != "epoch" and (x := num(v)) is not None}
        run.log(log, step=epoch)

    if rows:
        key = "metrics/mAP50-95(B)"
        best = max(rows, key=lambda r: num(r.get(key)) or -1)
        for k, v in best.items():
            if (x := num(v)) is not None:
                run.summary[f"best_epoch/{k}"] = x
        run.summary["epochs_trained"] = len(rows)

    for p in sorted(list(run_dir.glob("*.png")) + list(run_dir.glob("*.jpg"))):
        run.log({f"plots/{p.stem}": wandb.Image(str(p))})

    energy = run_dir / "energy_report.json"
    if energy.exists():
        import json
        for k, v in json.loads(energy.read_text()).items():
            if isinstance(v, (int, float, str)):
                run.summary[f"energy/{k}"] = v

    for t in args.test:
        name, vals = t.split("=", 1)
        p, r, m50, m5095 = map(float, vals.split(","))
        for k, v in (("precision", p), ("recall", r), ("mAP50", m50), ("mAP50-95", m5095)):
            run.summary[f"test/{name}/{k}"] = v

    if args.weights:
        art = wandb.Artifact(name=f"{(args.name or run_dir.name).replace('/', '-')}_model", type="model")
        for w in ("best.pt", "best.onnx"):
            if (run_dir / "weights" / w).exists():
                art.add_file(str(run_dir / "weights" / w))
        run.log_artifact(art, aliases=["best"])

    print(f"backfilled {len(rows)} epochs -> {run.url}")
    run.finish()


if __name__ == "__main__":
    sys.exit(main())
