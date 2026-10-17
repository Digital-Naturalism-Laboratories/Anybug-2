# MBD-1-2 (final): s @1600 retrained on the corrected Mothbox labels

**Data:** Mothbox MBD-1.1 with the empty-label fix (+144 boxes in 64 photos; see DATASETS.md). Cluster copy
`incoming-andy/mothbox_mbd12_2048`: 10,940 / 796 / 1,567 boxes (train / val / test).

**Recipe:** same as `mothbox_s1600_fast`: YOLO26s-obb, imgsz 1600, 200 epochs, mosaic 1.0, batch 6, H100 MIG 1g.24gb.
3 h 20 min. Energy (estimated; the MIG slice can't report power): 0.52 kWh incl. PUE 1.2, ~8 g CO2e at
15 g/kWh (BC grid). Run: `/mnt/ami-training/andy/runs/mothbox_s1600_mbd12`.

**Corrected test split** (170 photos, 1,711 boxes; all three models evaluated on the same fixed labels):

| Model | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| MBD-1-1 | 0.867 | 0.840 | 0.906 | 0.603 |
| s1600, old labels (`mothbox_s1600_fast`) | 0.873 | 0.890 | 0.922 | 0.651 |
| **s1600, fixed labels (`mothbox_s1600_mbd12`) = MBD-1-2** | **0.878** | **0.902** | **0.935** | **0.668** |

The label fix alone is worth about +1.3 mAP50 / +1.7 mAP50-95 over the same recipe. Against MBD-1-1: +6.2 recall, +6.5 mAP50-95,
same model size and laptop cost.

**Shipped:** `Mothbot_Process/trained_models/MBD-1-2.{pt,onnx,yaml}` (yaml name `yolo26s_Oct162026_mbd12data`),
replacing the earlier MBD-1-2 (the old-label model, never released). sha256 of `best.pt`: d71b8c3d21d6257e…
