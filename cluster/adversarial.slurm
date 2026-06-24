#!/bin/bash
# Adversarial robustness: train naive + hardened GAT on LI-Small, then build the before/after
# artifact (spec §7.8). Usage: sbatch cluster/adversarial.slurm
# `logs/` must exist before submitting (mkdir -p logs).
#SBATCH --job-name=aegis_adv
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/aegis_adv-%j.out
#SBATCH --error=logs/aegis_adv-%j.err
#SBATCH --export=ALL
set -euo pipefail

module purge
module load cuda12.1/toolkit/12.1.1

PROJECT_DIR="/home/brageei/AEGIS"
cd "$PROJECT_DIR"

CACHE_ROOT="${SCRATCH:-/work/$USER}/aegis_cache_${SLURM_JOB_ID}"
mkdir -p "$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" TORCH_HOME="$CACHE_ROOT/torch" TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"
export AEGIS_FEATURE_CACHE="/home/brageei/AEGIS/cache/features"
mkdir -p "$AEGIS_FEATURE_CACHE"

source "/home/brageei/AEGIS/env/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
CA="/etc/ssl/certs/ca-certificates.crt"; [ -f "$CA" ] && export SSL_CERT_FILE="$CA" REQUESTS_CA_BUNDLE="$CA"

echo "Job started at: $(date)"; echo "Node: $(hostname)"; nvidia-smi -L || true; echo

NAIVE_DIR="outputs/gat_lismall_${SLURM_JOB_ID}"
HARD_DIR="outputs/gat_lismall_adv_${SLURM_JOB_ID}"
ARTIFACT_DIR="outputs/adversarial_${SLURM_JOB_ID}"

echo "=== 1/3 train naive GAT ==="
python -u ml/train.py --config experiments/gat_lismall.yaml \
    --out-dir "$NAIVE_DIR" --feature-cache "$AEGIS_FEATURE_CACHE" --seed 42

echo "=== 2/3 train hardened GAT (adversarial training) ==="
python -u ml/train.py --config experiments/gat_lismall_adversarial.yaml \
    --out-dir "$HARD_DIR" --feature-cache "$AEGIS_FEATURE_CACHE" --seed 42

echo "=== 3/3 before/after artifact ==="
python -u -m ml.adversarial \
    --naive-ckpt "$NAIVE_DIR/best.pt" --hardened-ckpt "$HARD_DIR/best.pt" \
    --feature-cache "$AEGIS_FEATURE_CACHE" --out-dir "$ARTIFACT_DIR" --seed 42 --split test

echo "Complete at $(date). Artifact: $ARTIFACT_DIR/artifact.json"
