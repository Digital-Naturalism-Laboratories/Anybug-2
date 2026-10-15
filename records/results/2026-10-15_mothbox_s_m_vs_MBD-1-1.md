# Mothbox s and m @1600 (new recipe) vs MBD-1-1

Same data (Mothbox MBD-1.1, same split), same architecture (YOLO26s-obb, 9.75M params), imgsz 1600.
The new run: 200 epochs, mosaic 1.0, batch 6, trained on the 2048 px shrunk copy on the cluster
(H100 MIG 1g.24gb), **3 h 20 min**. Weights and ONNX: `/mnt/ami-training/andy/runs/mothbox_s1600_fast/weights/`.
MBD-1-1: 100 epochs, mosaic off, trained on Andrew's Mac (~31 h).

**Test split** (170 photos, 1,694 boxes, evaluated with `yolo val split=test` on the cluster):

| Model | P | R | mAP50 | mAP50-95 | inference |
|---|---|---|---|---|---|
| MBD-1-1 | 0.867 | 0.849 | 0.909 | 0.606 | 14.4 ms |
| s1600 new | 0.873 | **0.898** | **0.925** | **0.656** | 13.2 ms |
| m1600 new | **0.878** | 0.868 | 0.921 | 0.637 | (GPU val 19.5 ms) |

Val split (best.pt): mAP50 0.946, mAP50-95 0.682 (MBD-1-1's final-epoch val: 0.923 / 0.606).

The gain comes from the training recipe alone (longer training plus mosaic): recall is up 5 points with no loss of precision,
at the same laptop cost.

## m @1600 (2026-10-15)

YOLO26m-obb (21.2M params, 457 GFLOPs at 1600, about 3.2x `s`), same data and recipe except batch 3,
5 h 27 min. Run: `/mnt/ami-training/andy/runs/mothbox_m1600`. Val (best.pt): mAP50 0.936, mAP50-95 0.660.
On test it is **worse than s** (recall -3 pts, mAP50-95 -1.9 pts) at about 3x the laptop cost.
Likely cause: 1,356 training photos is too few for the bigger model to pay off.

**Decision:** `s` 1600 is the Mothbox-only upgrade candidate (`mothbox_s1600_fast/weights/best.onnx`);
`l` was skipped for Mothbox-only. Revisit model size on the larger Flat / Mixed sets.
