# Refinement tools

Scripts in `scripts/refine/`. Run them with a Python that has ultralytics, opencv and PIL
(locally: `~/Documents/GitHub/Mothbot_Process/.venv/bin/python`). Every long-running script
saves its progress after each image to a `.jsonl` file and resumes if re-run.

## Finding creatures the annotators missed

**`flatbug_find_missed.py`** runs a Mothbox detector over a dataset and keeps the detections
that don't overlap any existing label.

- `--format flatbug` (COCO folders) or `--format fieldguide` (`index.jsonl`; subsets = groups, mollusks skipped).
- Images larger than 4000 px get a full-frame pass plus 3200 px tiles with 15% overlap. Detections cut
  by a tile edge are dropped, then NMS (overlap-merging) across passes.
- **Match rule:** a detection counts as already labelled if IoU > 0.3 with a label box, or its centre is inside a label's outline.
- **Writes:**
  - `detections.jsonl`: per image, every detection with a `matched` flag, plus how many labels the model found.
  - `crops/<id>_s.jpg` (tight, 200 px) and `_b.jpg` (wide, 720 px). Pink = detection, green = existing outlines.
- The summary table ("found-by-model") tells you which subsets the model can be trusted on.

**`score_ugliness.py`** adds Mothbot Process's blur score (`core/blur.py` `blur_fields`, same crop as
`pipeline/detect.py`) to every candidate, so the review page can sort sharpest-first or ugliest-first.

## The review page

**`review_missed.py`** is a tiny local web app (Python standard library only). Run
`python3 review_missed.py --dir <folder> [--port 8765]` and open http://127.0.0.1:8765.

**Marking:**

| Action | Meaning |
|---|---|
| left-click | green (creature / good). Click again to clear |
| right-click | red (not / bad / don't care). Right-click again to clear |
| unmarked | not reviewed yet. Saving never touches these |
| Enter / Space | save marks. Marked tiles leave the "unreviewed" view |
| A / N | mark every unmarked tile on the page green / red |
| U | undo the last save |
| shift-click or hover+Z | big view |
| ← / → | change page. Marks are kept until saved |

**Saving:** decisions go to `<dir>/decisions.json` (`{id: {label: "insect"|"not", t}}`) after every save.

**Options:**
- `--cols` / `--max-rows`: page size. Andrew prefers `--cols 5 --max-rows 2` (10 per page).
- `--max-size`: starting upper size filter.
- `--title`, `--legend`: help text.

**Filters:** subset, size range, min confidence, show (unreviewed / creature / not / all), sort
(confidence, size, ugly ↑/↓, by image).

It reads any folder shaped like `flatbug_find_missed.py` output: `detections.jsonl` with
`dets[].{id, matched, conf, long, short, caption?}` plus `crops/<id>_s.jpg` and `_b.jpg`. A
`caption` replaces the confidence/size text on a tile.

**UI lessons learned (Andrew):**
- Unmarked must mean "not reviewed". An early version saved unmarked tiles as "not", and that was confusing.
- Thumbnails must show the **whole photo**. Unlabelled insects outside a zoomed crop slipped through.
- Show possible extras in place (cyan circles), and review flagged items first.
- Thick (6 px) pink boxes.

## Fieldguide box tightening with SAM

**`fieldguide_sam_boxes.py`** prompts SAM 2.1 (`sam2.1_b.pt`, via Ultralytics `SAM`) with each user
square. It takes the largest connected mask piece, applies `cv2.minAreaRect` and simplifies the outline.
`keep` = mask exists, ≤12 pieces and box <80% of the square. Takes about 0.8–1 s per photo on Apple Silicon MPS.

**`fieldguide_sam_review_prep.py`** turns the kept boxes into a review folder:
- Each thumbnail is the whole photo, letterboxed to 480 px.
- Green = user square, yellow = SAM outline, pink = OBB that becomes the label, cyan circles = detector
  finds outside the square (from `--missed`, conf ≥0.5, ≥20 px; photos with >20 unmatched detections are treated as texture and not flagged).
- Captions: `⚠ N extra? · taxon · pieces · ratio`.
- "Confidence" is used only to order the page: flagged photos first, most extras first.
- `--force` redraws thumbnails.

Run the review with:
```
python3 review_missed.py --dir ~/Desktop/Megabug/fieldguide_sam_review \
  --cols 5 --max-rows 2 --max-size 100000 --title "Fieldguide SAM box review" \
  --legend "Judge the pink box …"
```

## Converters

- **`flatbug_to_yolo_obb.py`:** flat-bug COCO polygons → YOLO OBB in `flat/` and `other/` (see DATASETS.md).
- **`yolofast_to_yolo_obb.py`:** YOLO-Fast zip → YOLO OBB in `other/` (`--refit` is opt-in and not recommended).
- **`fieldguide_to_yolo_obb.py`:** approved Fieldguide SAM boxes → YOLO OBB in `other/` (+ `labels_seg`, `manifest.jsonl`).

## Mothbox empty-label check

**`mothbox_empty_check.py`** finds every image in a YOLO dataset whose label file is empty, runs a detector
(MBD-1-2) on it, and makes one review tile per detection (tight crop + wide view, as in the flat-bug review;
each keeps its rotated box). Green = real creature (will be written into that photo's label file), red = not.
`--dets-from` reuses saved detections. (A first version showed whole photos; Andrew pointed out that a
9248 px photo holds many candidates, so per-detection tiles are the right unit.)

## Sticky-sheet (Charvalakis) tools

- **`charvalakis_focus_rank.py`** reads the multi-part zips directly and drops infrared frames (near-grayscale).
  It scores each visible frame's focus inside the authors' insect boxes and writes `frames.jsonl` plus a review folder,
  sorted sharpest first. Tiles are native-resolution crops of the insects.
- **`charvalakis_to_yolo_obb.py`** takes the sharpest frame per trap-hour, then the top `--top-frac` scenes, capped at
  `--per-day` frames per trap-day. It writes YOLO OBB (class 0 only), shrunk to 2048 px.

## collect_yolo_training_data.py additions (2026-10-16)

- **Grouping:** images named `<scene>__t<time>` (time-lapse frames) are grouped like `<photo>_x<X>_y<Y>` tiles, so they
  always land in the same split.
- **`--keep-split DIR`:** a source that already has `images/{train,val,test}` keeps its split instead of being
  re-split. Used for Mothbox, so every model is scored on the same Mothbox test photos.