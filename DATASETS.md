# Datasets: where they came from and what we did to them

Every dataset considered for the Anybug-2 mashups, with its source, licence,
format, what was wrong with it, how we fixed it, and where the result lives.
Decisions were made by Andrew Quitmeyer while reviewing (2026-10-12 to 2026-10-15).

**Target format for everything:** Ultralytics YOLO OBB, one class `0: creature`,
label line = `0 x1 y1 x2 y2 x3 y3 x4 y4` (normalised 0–1). Boxes include legs and
antennae (the Mothbox convention).

**Groups:**
- **Flat** = Mothbox + datasets photographed on plain/flat backgrounds (sheets, trays,
  dishes, sticky cards, scanners).
- **Other** = naturalistic or otherwise non-flat backgrounds. Used only in the **Mixed** set
  (Mixed = Flat + Other).

Local working folder: `~/Desktop/Megabug/`. Cluster: `ami-workspace-02-gpu:/mnt/ami-training/`
(see [TRAINING.md](TRAINING.md)).

| Dataset | Status | Group | Images → output | Labels |
|---|---|---|---|---|
| [Mothbox MBD-1.1](#mothbox-mbd-11) | **In** | Flat (base set) | 1,695 | 13,102 OBB |
| [flat-bug](#flat-bug) | **In** | 19 subsets Flat, 4 Other | 6,131 → 7,341 (tiles) | 113,552 + 371 added |
| [YOLO-Fast](#yolo-fast) | **In** | Other only | 257 | 257 |
| [Fieldguide fg-arthropods-v1](#fieldguide-fg-arthropods-v1) | **In progress** (SAM boxes under review) | Other (expected) | 20,642 photos, 13,243 SAM boxes kept | 1 per photo |
| [insectDetector](#insectdetector) | Skipped | – | 777 | – |
| [broadTaxonClassifier](#broadtaxonclassifier) | Skipped | – | 125k crops | – |
| [Entomoscope 2.0](#entomoscope-20) | Skipped | – | 54 | – |
| [insects.zip (green roofs)](#insectszip-green-roof-camera-traps) | Skipped | – | 7,667 | – |
| [Charvalakis et al. 2026](#charvalakis-et-al-2026-sticky-sheets-under-streetlights) | Images not public; authors emailed | (Flat if obtained) | – | – |

---

## Mothbox MBD-1.1

- **Source:** our own Mothbox training set, the one MBD-1-1 was trained on. Local
  `~/Desktop/2026_09_17_YOLOOBB_TRAINING`. Published on Zenodo as "Mothbox Training
  Dataset 1-1" (record 23140320). The cluster copy at
  `/mnt/ami-training/detector/sources/mothbox-mbd-1.1-zenodo-23140320/extracted` is
  file-for-file identical (same 1,695 names, same split, same box counts; checked 2026-10-14).
- **Format:** YOLO OBB, `creature`, split train 1,356 / val 169 / test 170 images (10,817 / 792 / 1,550 boxes).
- **Processing:** none to the labels. For training on the cluster we use our own copies:
  `incoming-andy/mothbox_mbd11` (full size) and `incoming-andy/mothbox_mbd11_2048`, the
  latter shrunk to a 2048 px long side with `scripts/cluster/shrink_images.py` (pure speed-up, see TRAINING.md).
- **Note:** `~/Desktop/2026_MOTHBOTYOLO_FULL` (1,630 images, 9,029 boxes) is an *older* set; don't use it as the base.

## flat-bug

- **Source:** Svenning et al., "A General Method for Detection and Segmentation of
  Terrestrial Arthropods in Images" (bioRxiv 2025, doi:10.1101/2025.04.08.647223).
  Data: Zenodo doi:10.5281/zenodo.14761447 (Creative Commons; each of the 23 subsets also has
  its own Zenodo DOI — see `records/flatbug/flatbug_metadata.csv`). The raw download is also on
  the cluster at `detector/sources/flatbug-zenodo-14761447`.
- **Format:** COCO polygons (CVAT `instances_default.json` per subset), one class `insect`.
  6,131 annotated images (+55 empty), 113,552 instances. Annotated "manually or
  semi-automatically" in CVAT with a ~32 px minimum size (applied loosely: up to 49% of labels
  in some subsets are under 32 px).
- **Problems found:**
  1. Small creatures skipped by annotators (the 32 px rule plus the odd miss).
  2. 99 PeMaToEuroPep photos are EXIF-rotated (labels in upright coordinates).
  3. Some subsets are huge (pinoy2023 14039×10200, abram2023 7811×14027), so creatures would be ~10 px at imgsz 1600.
- **Refinement:**
  1. **Finding missed creatures.** `flatbug_find_missed.py` ran MBD-1-1 (imgsz 1600, conf 0.15,
     full frame + 3200 px tiles for big images) and kept detections not overlapping any flat-bug label
     (IoU ≤ 0.3 and centre outside every outline): 41,582 candidates. `score_ugliness.py` added
     Mothbot Process's blur/"ugly" score to each (it grouped look-alikes but didn't separate
     creature from junk). Andrew reviewed 5,120 of them in `review_missed.py`:
     **371 creatures, 4,749 not.** Record: `records/flatbug/missed_creature_decisions.json`.
     Policy: only those 371 become new labels; every other model detection is ignored. Andrew's
     notes: nearly all candidates were debris, text, blobs or printed patterns. AMI-traps
     creatures ≤15 px are all garbage, and blurry bugs at AMI image quality count as "bad".
     The model is blind on some subsets (it finds <10% of the existing labels on CollembolAI, DIRT,
     PeMaToEuroPep, abram2023 and biodiscover-arm), so no misses could be found there.
  2. **Conversion.** `flatbug_to_yolo_obb.py`: each outline becomes `cv2.minAreaRect`
     (outlines kept in `labels_seg/`). EXIF orientation applied. Subsets whose median creature would be
     smaller than 25 px at imgsz 1600 (Mothbox's median) are cut into the fewest evenly spaced
     overlapping tiles: pinoy2023 (4 tiles), abram2023 (6) and PeMaToEuroPep (4). A creature cut by a tile
     edge is kept if ≥25% of it is visible. The 55 unannotated images are skipped. Unchanged
     images are hard-linked. Files are named `FB-<code>_<name>[_x<X>_y<Y>]`.
- **Groups (Andrew):**
  - Other: NHM-beetles-crops, DIRT, PeMaToEuroPep, ArTaxOr.
  - Flat: the other 19 (ALUS, AMI-traps, AMT, BIOSCAN, CollembolAI, Diopsis, DiversityScanner,
    Mothitor, abram2023, amarathunga2022, anTraX, biodiscover-arm, cao2022, gernat2018, pinoy2023,
    sittinger2023, sticky-pi, ubc-pitfall-traps, ubc-scanned-sticky-cards).
- **Output:** `~/Desktop/Megabug/flatbug_yolo_obb/{flat,other}/{images,labels,labels_seg}`.
  - flat: 4,166 images, 107,457 boxes (98,059 flat-bug + 367 added).
  - other: 3,175 images, 20,493 boxes (15,493 + 4 added).
  - Partially uploaded to the cluster `incoming-andy/flatbug_yolo_obb` (an upload was interrupted;
    re-run the rsync in TRAINING.md).

## YOLO-Fast

- **Source:** `YOLO-Fast-dataset.zip` (origin and licence not recorded in the archive; the file
  names point to bark-beetle work: Ips, Pityogenes, Xyleborinus…).
- **Format:** 257 focus-stacked macro PNGs (mostly 4056×3040), axis-aligned YOLO, exactly one box per image.
- **Assessment:** single specimens filling a large part of the frame; the boxes sometimes cut off a wing or leg.
- **Refinement:** `yolofast_to_yolo_obb.py`. Their boxes are kept as unrotated OBBs and PNGs re-saved
  as JPEG q95. Re-fitting the boxes with MBD-1-1 was tried and rejected (it cut off legs, heads and half a
  beetle on macro portraits), so `--refit` is opt-in.
- **Group:** Other only (Andrew).
- **Output:** `~/Desktop/Megabug/yolofast_yolo_obb/other` (257 images, 420 MB).

## Fieldguide fg-arthropods-v1

- **Source:** Fieldguide photos, `fg-arthropods-v1` v1.0.0 (built 2026-10-12), 20,642 photos.
  Their README and spec are in `records/fieldguide/`. **Licences are mixed and mostly
  unclear:** ~75% "unspecified" or missing. Only cc0 / cc-by / cc-by-sa allow redistribution;
  check `license_class` before publishing anything derived from it.
- **Format:** one row per photo in `index.jsonl`, **one box per photo**: the square crop the
  uploader made (`bbox_xyxy`, loose). Groups: moth 15,796, butterfly 3,518, other_arthropod 1,017,
  mollusk_worm 311 (excluded).
- **Problems:**
  1. Every other creature in the frame is unlabelled.
  2. The one box is a loose square.
  3. Mostly close-ups on walls, bark and leaves.
- **Refinement so far:**
  1. `flatbug_find_missed.py --format fieldguide`. MBD-1-1 can't tighten the boxes: it finds the
     subject in only 51% of moth photos and gives exactly one box in 36%, often oversized. Its
     extra detections are mostly background texture: 2% of photos (mesh, stucco) hold 82% of them.
     Output `~/Desktop/Megabug/fieldguide_missed`.
  2. `fieldguide_sam_boxes.py`: SAM 2.1-b (`sam2.1_b.pt`) prompted with each user square. The
     largest connected piece of the mask goes through `cv2.minAreaRect` to give an OBB. Kept if the mask
     is in ≤12 pieces and the box covers <80% of the square (masks that grab textured
     walls/bark come back in dozens or hundreds of pieces). **Kept 13,243 / 20,331** (moth 69%,
     butterfly 52%, other 49%). In a 50-photo hand check this rule kept 28, of which 2 were bad.
  3. `fieldguide_sam_review_prep.py` + `review_missed.py`: Andrew reviews every kept box on the
     whole photo, with cyan circles marking confident detections outside the square (from step 1).
     **Rejecting** = bad box **or** unlabelled insects in the photo. Flagged photos (1,114) were
     reviewed first. Snapshot 2026-10-15: 5,147 reviewed (4,220 good, 927 rejected), in
     `records/fieldguide/sam_box_review_decisions.snapshot-2026-10-15.json`. Live file:
     `~/Desktop/Megabug/fieldguide_sam_review/decisions.json`.
- **Still to do:** finish the review, then convert approved photos (SAM OBB as the label; the outline can go into
  `labels_seg`) to `fieldguide_yolo_obb/other`. Unreviewed and rejected photos are dropped.
- **Earlier, sample only:** the 184-photo `fg-arthropods-v1-sample` (in `~/Downloads`) was used to first
  inspect the format. One photo there is EXIF-rotated, so always apply `exif_transpose`.

## insectDetector

- `insectDetector.zip`: 777 AMI light-trap snapshots (3840×2160; traps LV/OH/SS), axis-aligned YOLO, one class.
- **Skipped:** only insects ≳100 px are boxed (smallest 10% of boxes = 109 px). Many visible midges and flies are
  unlabelled, and the 86 "empty" images aren't empty. Not the same photos as flat-bug's AMI-traps.

## broadTaxonClassifier

- `broadTaxonClassifier.zip`: 125,135 crops in 16 order-level folders (LepidopteraMicro, DipteraNematocera, …,
  Vegetation), from the same AMI trap network. The moth crops are named by species.
- **Skipped:** classification crops only. There are no full images or boxes, so it's no use for the detector.

## Entomoscope 2.0

- `Entomoscope2.0.zip`: 54 single-specimen telecentric photos (4024×3036). The "bounding box drawn
  manually" in `Specimen_*_metadata.txt` is a rotated rect `(cx, cy, length, width, angle°)`.
- **Skipped:** these are body-only measurement boxes (they exclude wings, legs and antennae, against the Mothbox
  convention), and there are only 54 images.

## insects.zip (green-roof camera traps)

- 7,667 Wingscapes time-lapse photos (4224×2376) of sedum green roofs in Denmark (sites NAIM/BOR;
  likely Bjerge et al., Aarhus University). YOLO, classes `Insect`/`Animal` (only Insect used). 39% of the images are empty.
- **Skipped:** strange neon colour processing, noisy foliage, and no way to verify that all insects are labelled.

## Charvalakis et al. 2026 (sticky sheets under streetlights)

- "Testing the transferability of image-based automated insect detection under different light
  conditions" (*Approaches in Entomology*, doi:10.1002/3070-5436.70004). Time-lapse sticky sheets
  under streetlights / LED traps (NL), YOLOv11.
- Their DataverseNL deposit (doi:10.34894/N4EVBD, CC-BY-4.0) has only `best.pt`, counts and scripts:
  **no images or boxes**. The companion deposits doi:10.34894/SNKXWK and doi:10.34894/FLUWWY are spreadsheets only.
  Andrew emailed the authors (contact k.spoelstra@nioo.knaw.nl) on 2026-10-14.
- If the data arrives: check that every insect is boxed, and the share of infrared images. Likely Flat.
