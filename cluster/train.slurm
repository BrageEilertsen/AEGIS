#!/bin/bash
#SBATCH --job-name=aegis_train
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/aegis_train-%j.out
#SBATCH --error=logs/aegis_train-%j.err
#SBATCH --export=ALL
set -euo pipefail

module purge
module load cuda12.1/toolkit/12.1.1     # confirm with `module avail cuda`; PyG wheels = cu121

PROJECT_DIR="/home/brageei/AEGIS"
cd "$PROJECT_DIR"

# Per-job THROWAWAY scratch on fast NODE-LOCAL /work (NOT the shared repo on /home).
# /work is local to each compute node, so it is ideal for transient per-job temp.
CACHE_ROOT="${SCRATCH:-/work/$USER}/aegis_cache_${SLURM_JOB_ID}"
mkdir -p "$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"

# Persistent spectral-feature cache (reused across runs), keyed by graph hash in code.
# MUST live on SHARED /home so compute nodes can read what was computed elsewhere.
export AEGIS_FEATURE_CACHE="/home/brageei/AEGIS/cache/features"
mkdir -p "$AEGIS_FEATURE_CACHE"

source "/home/brageei/AEGIS/env/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

echo "Job started at: $(date)"; echo "Node: $(hostname)"; nvidia-smi -L || true; echo

CONFIG="${1:-experiments/gat_lismall.yaml}"
RUN_NAME="$(basename "${CONFIG%.yaml}")"
OUT_DIR="outputs/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -u ml/train.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --feature-cache "$AEGIS_FEATURE_CACHE" \
    --seed 42 \
    2>&1 | tee "${OUT_DIR}/train.log"

echo "Run complete at $(date). Results: $OUT_DIR"
