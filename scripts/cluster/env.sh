# Source this on ami-workspace-02-gpu before training:  source /mnt/ami-training/andy/env.sh
# The VM's root disk (/, which holds ~) is only 20 GB and was full, so the Python venv,
# pip/PyTorch/Ultralytics caches and temp files all live on the big shared mount.
export PIP_CACHE_DIR=/mnt/ami-training/andy/.cache/pip
export XDG_CACHE_HOME=/mnt/ami-training/andy/.cache
export TORCH_HOME=/mnt/ami-training/andy/.cache/torch
export YOLO_CONFIG_DIR=/mnt/ami-training/andy/.cache/Ultralytics
export TMPDIR=/mnt/ami-training/andy/tmp
[ -d /mnt/ami-training/andy/venv ] && source /mnt/ami-training/andy/venv/bin/activate
cd /mnt/ami-training/andy
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on this vGPU/MIG slice —
# training then dies with "CUDA_ERROR_NOT_SUPPORTED ... cuMemAddressReserve".
# W&B stages and caches artifact files (model uploads) under ~ by default; the 20 GB root disk filled
# up and crashed the end of a training run (mixed_n1600, 2026-10-23). Keep them on the mount.
export WANDB_DIR=/mnt/ami-training/andy
export WANDB_CACHE_DIR=/mnt/ami-training/andy/.cache/wandb
export WANDB_DATA_DIR=/mnt/ami-training/andy/.cache/wandb-data
export WANDB_ARTIFACT_DIR=/mnt/ami-training/andy/.cache/wandb-artifacts
