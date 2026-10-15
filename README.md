# Anybug-2

Training data and recipe for the next Mothbox creature detector: our own Mothbox
rotated-box (OBB) dataset merged with other groups' insect datasets, cleaned up to the
Mothbox labelling convention (every creature in frame boxed, legs and antennae included,
one class `creature`).

We're building four training sets:

| Set | What |
|---|---|
| **Mothbox** | our MBD-1.1 dataset alone (baseline / upgrade of MBD-1-1) |
| **Flat** | Mothbox + other datasets on plain/flat backgrounds |
| **Mixed** | Flat + naturalistic / other backgrounds |
| **TestFlat_150MB** | a ~150 MB sample of Flat for testing the cluster workflow |

## Where to look

- **[DATASETS.md](DATASETS.md):** every dataset we considered: source, DOI, licence, format, what was
  wrong with it, how we refined it, which group it's in, and where the output lives. Also what we skipped and why.
- **[REFINEMENT_TOOLS.md](REFINEMENT_TOOLS.md):** the scripts and the little review web app we used to
  find missed creatures, tighten boxes with SAM and review everything fast.
- **[TRAINING.md](TRAINING.md):** cluster setup, gotchas, commands, parameter reasoning.
- **`records/`:** the refinement record:
  - review decisions (`flatbug/missed_creature_decisions.json`, `fieldguide/sam_box_review_decisions…json`);
  - run arguments;
  - copies of source metadata (flat-bug `metadata.csv`, Fieldguide README/spec);
  - results.

## Scripts

```
scripts/refine/   find missed creatures, ugliness scoring, SAM box tightening, review web app, converters
scripts/train/    collect_yolo_training_data.py, train_yolo_obb.py, make_testflat_sample.py
scripts/cluster/  env.sh for the GPU VM, shrink_images.py
```

`scripts/train/collect_yolo_training_data.py` and `train_yolo_obb.py` were copied on 2026-10-15 from
`Mothbox/AI/training_scripts/` (which still has its own copies; keep the two in sync or edit here).
`scripts/refine/*` were written in `Mothbox/AI/training_scripts/megabug/`, which was removed on
2026-10-15; this repo is now their only home.

## Data locations

- **Local working folder:** `~/Desktop/Megabug/`. Downloads, intermediate runs (`flatbug_missed/`,
  `fieldguide_missed/`, `fieldguide_sam/`, `fieldguide_sam_review/`), converted outputs (`*_yolo_obb/`) and
  `models/sam2.1_b.pt`.
- **Cluster:** `ami-workspace-02-gpu:/mnt/ami-training/` (see TRAINING.md).

Large data is not in this repo (see `.gitignore`).
