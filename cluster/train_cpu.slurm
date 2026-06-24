#!/bin/bash
# CPU training/feature-build on milanq (conserves GPU hours for early/cheap phases: data prep,
# GCN baseline, spectral caching — CLAUDE.md). Usage: sbatch cluster/train_cpu.slurm <config>
# `logs/` must exist before submitting (mkdir -p logs).
#SBATCH --job-name=aegis_train_cpu
#SBATCH --partition=milanq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# NOTE: this cluster does not track memory (Slurm RealMemory=1); jobs get the full node RAM, and
# requesting --mem fails ("Memory specification can not be satisfied"). So no --mem here.
#SBATCH --time=06:00:00
#SBATCH --output=logs/aegis_train_cpu-%j.out
#SBATCH --error=logs/aegis_train_cpu-%j.err
#SBATCH --export=ALL
set -euo pipefail

PROJECT_DIR="/home/brageei/AEGIS"
cd "$PROJECT_DIR"

# Per-job throwaway scratch on node-local /work; persistent spectral cache on shared /home.
CACHE_ROOT="${SCRATCH:-/work/$USER}/aegis_cache_${SLURM_JOB_ID}"
mkdir -p "$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
export AEGIS_FEATURE_CACHE="/home/brageei/AEGIS/cache/features"
mkdir -p "$AEGIS_FEATURE_CACHE"

source "/home/brageei/AEGIS/env/bin/activate"
export PYTHONUNBUFFERED=1
# Match thread count to the allocation so numpy/scipy/torch don't oversubscribe.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job started at: $(date)"; echo "Node: $(hostname)"; echo "CPUs: ${SLURM_CPUS_PER_TASK}"; echo

CONFIG="${1:-experiments/gcn_lismall.yaml}"
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
