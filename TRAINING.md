# Training

## The sets

| Set | Sources | Status |
|---|---|---|
| **Mothbox** | Mothbox MBD-1.1 (its own train/val/test split) | trained `s` and `m` at 1600 — `s` wins (see records/results) |
| **TestFlat_150MB** | 6 Mothbox photos + 2 per flat-bug flat subset (`scripts/train/make_testflat_sample.py`) | built; used to test the cluster workflow |
| **Flat** | Mothbox (fixed labels) + flat-bug `flat` + Charvalakis sticky sheets | **built: `Flat_v1`** (see below) |
| **Mixed** | Flat sources + flat-bug `other` + YOLO-Fast + Fieldguide | **built: `Mixed_v1`** (see below) |

Flat and Mixed are built with `scripts/train/collect_yolo_training_data.py`. That script:
- re-splits sources 80/10/10;
- keeps tiles of one photo (`_x<X>_y<Y>`) in the same split;
- writes a `data.yaml` with **no `path:`**, so the folder can be copied anywhere.

Note that it re-splits Mothbox too. To compare against MBD-1-1 on the same test photos, add an
option that keeps a source's existing split, or evaluate on the Mothbox test split separately.

## Flat_v1 (built 2026-10-16)

`~/Desktop/Megabug/Flat_v1`, 3.2 GB, all images shrunk to ≤2048 px (training runs at 1600 and Ultralytics resizes on load).

| Source | Images | Boxes | Split |
|---|---|---|---|
| Mothbox, fixed labels (`mothbox_mbd12_2048`) | 1,695 | 14,693 | **its own** (`--keep-split`): 1,356 / 169 / 170 |
| flat-bug `flat` (`flatbug_flat_2048`) | 4,166 | 107,457 | 80/10/10 (tiles grouped) |
| Charvalakis sticky sheets (`charvalakis_yolo_obb/flat`) | 75 | 9,985 | 80/10/10 (trap-days grouped) |
| **Flat_v1 total** | **5,935** (4,754 / 594 / 587) | **132,130** (107,766 / 10,341 / 14,023) | |

Built with:
```
python3 collect_yolo_training_data.py -s mothbox_mbd12_2048 -s flatbug_flat_2048 -s charvalakis_yolo_obb/flat \
  --keep-split mothbox_mbd12_2048 -o Flat_v1
```

Then one fix: the photo `lanzarViper_2026_05_21__01_08_04_HDR0` is byte-identical in the original Mothbox **train and
val** splits (a leak in MBD-1.1 itself), so its val copy was removed. Checks: every image has a label file; no group spans
two splits; the Mothbox photos are all in their original split. 28 Mothbox OBB lines (26 photos) have a corner up to
1.039, just off the image edge. They're left as-is, since Ultralytics accepts them and they are what MBD-1-2 used.

**Counting note:** some Mothbox label files have no trailing newline, so `cat labels/*.txt | grep -c .` undercounts
(13,303 instead of 14,693). Count with `awk 'NF' labels/*.txt | wc -l`. Earlier box counts quoted for the Mothbox set
(13,102 / 13,303) were undercounts made the same way.

**To compare with MBD-1-1 / MBD-1-2,** evaluate Flat models on the Mothbox test split (`mothbox_mbd12_2048/data.yaml`,
`split=test`), which is the same 170 photos.

## Mixed_v1 (built 2026-10-17)

`~/Desktop/Megabug/Mixed_v1`, 7.7 GB, all images ≤2048 px. Same build as Flat_v1 plus the three "other" sources
(each shrunk with `shrink_images.py` first): `flatbug_other_2048` (3,175 images / 20,493 boxes), `yolofast_other_2048`
(257 / 257), `fieldguide_other_2048` (5,408 / 5,408).

| Source | Train | Val | Test |
|---|---|---|---|
| Mothbox (own split) | 1,356 | 168 | 170 |
| flat-bug (flat + other) | 5,876 | 726 | 739 |
| Fieldguide | 4,303 | 558 | 547 |
| YOLO-Fast | 214 | 20 | 23 |
| Charvalakis sticky sheets | 59 | 5 | 11 |
| **Total images** | **11,808** | **1,477** | **1,490** |
| **Boxes** | 126,284 | 15,284 | 16,720 (158,288 total) |

```
python3 collect_yolo_training_data.py -s mothbox_mbd12_2048 -s flatbug_flat_2048 -s charvalakis_yolo_obb/flat \
  -s flatbug_other_2048 -s yolofast_other_2048 -s fieldguide_other_2048 --keep-split mothbox_mbd12_2048 -o Mixed_v1
```

Then the duplicate `lanzarViper_2026_05_21__01_08_04_HDR0` val copy was removed. Checks passed: labels for every image,
all lines 9-column, no group spans splits, Mothbox photos in their original split.

## Mixed_Open_v1 (built 2026-10-20) — the publishable version

Fieldguide's terms allow training ML models on its photos, so Mixed_v1 is fine for our own models. To **publish** the
mashup, though, only openly licensed images may be redistributed. Mixed_Open_v1 = **Mixed_v1 minus every Fieldguide photo
whose `license_class` isn't cc0 / cc-by / cc-by-sa** (Fieldguide README's redistribution rule). Everything else, including
each image's split, is identical, so results are comparable on the shared photos.

- **Fieldguide kept:** 121 of 5,408 (16 cc0, 69 cc-by, 36 cc-by-sa); 97 train / 10 val / 14 test.
  5,287 photos and 5,287 boxes removed. Exclusion list: `records/fieldguide/mixed_open_v1_exclude.txt` (paths relative to the
  dataset root).
- **Totals:** 9,488 images (7,602 / 929 / 957), 153,001 boxes (122,078 / 14,736 / 16,187).
- **Local:** `~/Desktop/Megabug/Mixed_Open_v1` (hard links into Mixed_v1, no extra disk).
- **Cluster:** built server-side the same way: `cp -al Mixed_v1 Mixed_Open_v1`, then remove the listed files.
- **Before publishing, also check the other sources' terms:**
  - YOLO-Fast (no licence recorded with the archive);
  - the Charvalakis sticky sheets (shared privately by the authors);
  - flat-bug (Creative Commons per subset DOI);
  - the Mothbox data (our own Zenodo record).

## Cluster: ami-workspace-02-gpu

- **Access:** `ssh ami-workspace-02-gpu` (via the `ami-bastion` jump host; key `~/.ssh/andy`).
- **GPU:** NVIDIA H100L-1-24C, a MIG 1g.24gb vGPU slice. About 23.8 GB of memory and about 1/7 of an H100's
  compute. It's shared (another user's BioCLIP process was using ~0.5 GB). Driver 580, CUDA 13.0. 6 CPUs.
- **Disks:** the root disk `/` (which holds `~`) is only 20 GB and was full. Everything lives on the
  shared mount `/mnt/ami-training` (1 TB):
  - scripts + venv + runs: `/mnt/ami-training/andy/`
  - data: `/mnt/ami-training/detector/sources/incoming-andy/`
- **One-time setup:**
  ```
  python3 -m venv /mnt/ami-training/andy/venv
  # copy scripts/cluster/env.sh to /mnt/ami-training/andy/env.sh, then:
  source /mnt/ami-training/andy/env.sh
  pip install -U pip && pip install ultralytics      # pulls torch cu130
  ```
- **Every session:** `tmux new -s <name>` (or `tmux attach -t <name>`), then `source /mnt/ami-training/andy/env.sh`.
  - Detach with Ctrl-b d. Turn on scrolling with `tmux set -g mouse on`.
  - Don't press Ctrl-C in the training window unless you mean to stop it.
- **Uploading from the Mac** (Andrew's home internet is unstable; re-run to resume):
  ```
  rsync -av --exclude 'runs/' --exclude '__pycache__/' --exclude '.DS_Store' \
    <this repo>/scripts/ ami-workspace-02-gpu:/mnt/ami-training/andy/
  rsync -av -e "ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=6" --exclude '.DS_Store' \
    ~/Desktop/Megabug/flatbug_yolo_obb ~/Desktop/Megabug/yolofast_yolo_obb \
    ami-workspace-02-gpu:/mnt/ami-training/detector/sources/incoming-andy/
  ```

### Gotchas we hit

1. **No space left on device** during `pip install`: the root disk is full. Use the venv and caches on the mount (`env.sh`).
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks training** on this vGPU
   (`CUDA_ERROR_NOT_SUPPORTED … cuMemAddressReserve`). Don't set it.
3. **Ultralytics edits datasets in place.** It rewrites corrupt JPEGs and writes `labels/*.cache`, and
   `train_yolo_obb.py`'s pre-scan *moves* corrupt images to `quarantine/`. Train only from your
   own copies, never the shared `detector/sources/<other>` folders.
4. **CPU-bound with full-size Mothbox photos.** At full size an epoch took ~12 min, with the GPU waiting on JPEG decoding. With
   images shrunk to 2048 px (`scripts/cluster/shrink_images.py`) it takes **~1 min/epoch**. Ultralytics resizes to imgsz on load anyway,
   so the model input is the same.
5. **AutoBatch (`--batch -1`) is too cautious.** It picked 3, but real use at batch 3 was ~7 GB. For `s` at 1600, `--batch 6` peaks around 19–20 GB.
6. **`CUDA out of memory in TaskAlignedAssigner …` warnings** on photos with >1,000 insects are harmless:
   Ultralytics redoes the label matching one image at a time and carries on. Only a `RuntimeError` is fatal.
7. **Validation NMS out of memory on dense data** (Flat_v1, epoch 25): per-epoch validation keeps every prediction
   above conf 0.001. On sticky cards with 1,000+ insects that is about 30k candidates, and rotated NMS's IoU matrix
   needs ~3.5 GB, so training crashed in `batch_probiou`. Fix: `--val-conf 0.01` (new flag). To rescue a crashed run,
   set `train_args['conf'] = 0.01` inside `last.pt` and `--resume` (resume ignores command-line args).
8. **`lr0` is ignored.** `train_yolo_obb.py`'s `lr0=0.005` is ignored because `optimizer=auto` (it picks SGD lr 0.01 for long runs).
   This was also true for MBD-1-1.

## Commands

**TestFlat (workflow check):**
```
python3 train_yolo_obb.py --data /mnt/ami-training/detector/sources/incoming-andy/TestFlat_150MB/data.yaml \
  --model s --imgsz 1600 --epochs 5 --batch -1 --mosaic 1.0 --workers 6 \
  --project /mnt/ami-training/andy/runs --name testflat_smoke
```

**Mothbox:**
```
python3 shrink_images.py --src .../incoming-andy/mothbox_mbd11 --dst .../incoming-andy/mothbox_mbd11_2048   # once
python3 train_yolo_obb.py --data /mnt/ami-training/detector/sources/incoming-andy/mothbox_mbd11_2048/data.yaml \
  --model s --imgsz 1600 --epochs 200 --batch 6 --mosaic 1.0 --workers 6 \
  --project /mnt/ami-training/andy/runs --name mothbox_s1600_fast
# m: --model m --batch 3 --name mothbox_m1600
```

**Test-split comparison:**
```
yolo val task=obb split=test imgsz=1600 batch=4 data=<…>/mothbox_mbd11_2048/data.yaml model=<weights> \
  project=/mnt/ami-training/andy/runs name=test_<label>
```

## Choosing parameters (reasoning, 2026-10-14)

- **Model size:** train `s`, `m` and `l` and compare accuracy against laptop speed in Mothbot Process.
  Compute is roughly `s` 1×, `m` ~2.5×, `l` ~3×, `x` ~7× (`x` is too slow for laptops).
- **Mosaic:** on (1.0). Ultralytics turns it off for the last 10 epochs. The script's `scale=0.9` plus mosaic can shrink
  20 px insects to a few pixels; if small-insect recall suffers, try scale 0.5 (needs a new flag).
- **imgsz:** keep 1600, matching Process; laptop cost grows with the square of the size. Mothbox's smallest 10% of boxes are ≤11 px at 1600
  (~14 px at 2048). A 2048 experiment is worth doing once, but note that the ONNX export fixes the input size.

## Weights & Biases logging (moth-ai team on CoreWeave Forge)

Added 2026-10-17 at a collaborator's request; runs are also recorded in `records/results/` here.

- **One-time setup on the GPU machine:**
  - `source env.sh && pip install wandb`
  - `wandb login --host=https://api.forge.coreweave.com`, pasting your personal key (Forge → User settings → API keys) when prompted. That is the API server; runs show up at https://forge.coreweave.com/wandb/moth-ai/<project>. Verified 2026-10-17 with a TestFlat smoke run.
  - Never put the key in `env.sh` or any file on the shared drive. `wandb login` stores it in `~/.netrc` with private permissions.
- **Per run:** add `--wandb-project anybug-2 --wandb-entity moth-ai [--wandb-tags …] [--wandb-notes "…"]`.
  - `train_yolo_obb.py` starts the W&B run itself, so it gets a clean project and run name, plus config with the dataset,
    model, imgsz, epochs, batch, mosaic, val conf and the Anybug-2 git commit.
  - Ultralytics' callback then logs epoch metrics, plots and the best model into it.
  - After training, the energy report is added to the run summary as `energy/*`.
- **Gotcha:** without this, Ultralytics' own W&B logging names the project after the `--project` folder path
  (that's where the `-mnt-anybug-runs-runs` project on Forge came from).
- **Resumed runs** keep logging only if the original run used W&B.
- **Older runs** (trained before W&B was set up) can be uploaded without retraining with
  `scripts/train/wandb_backfill.py --run <run folder> --project anybug-2 --entity moth-ai [--weights] [--test NAME=P,R,mAP50,mAP50-95 …]`.
  It uploads `results.csv` as per-epoch history, `args.yaml` as config, the plots, the energy report and the test scores,
  tagged `backfilled`.

## Energy and carbon

`train_yolo_obb.py` prints an energy estimate after training and saves `energy_report.json` in the
run folder (added 2026-10-15).

- **GPU:** live power from NVML when the driver exposes it; otherwise estimated as board watts × the share
  of the card. Our MIG 1g slice can't report power (`nvidia-smi` shows N/A), so it's estimated as
  350 W × 26/132 SMs ≈ 69 W.
- **Host:** 10 W per visible CPU core (60 W on the cluster VM). Apple Silicon: GPU 20 W + host 15 W.
- **Data-centre overhead:** PUE 1.2 on servers, 1.0 on a laptop.
- **Grid carbon:** default `--grid-gco2-per-kwh 480` (≈ world average). The cluster is at BCNET (Vancouver, BC), and BC's
  grid is mostly hydro (on the order of 10–20 g/kWh), so pass a BC figure for real numbers.
- **Options:** `--gpu-watts`, `--host-watts`, `--pue`, `--grid-gco2-per-kwh`.

Rough figures for the runs so far, with these defaults:

| Run | Hours | ≈ kWh | ≈ kg CO2e at 480 g | ≈ g CO2e at 15 g (BC) |
|---|---|---|---|---|
| s1600 (cluster) | 3.3 | 0.5 | 0.25 | 8 |
| m1600 (cluster) | 5.5 | 0.85 | 0.4 | 13 |
| MBD-1-1 (Mac, MPS) | ~31 | 1.1 | 0.5 | (Mac's own grid) |

## Results

See `records/results/`.
