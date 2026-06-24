#!/bin/bash
# Full-graph evaluation / large-scale testing on GPU (separate from training).
# Usage:  sbatch cluster/eval.slurm <run_dir> [split]
#   <run_dir>  a training --out-dir containing best.pt + run_context.json (e.g. outputs/gcn_baseline_12345)
#   [split]    val | test  (default: test)
# `logs/` must exist before submitting (mkdir -p logs).
#SBATCH --job-name=aegis_eval
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/aegis_eval-%j.out
#SBATCH --error=logs/aegis_eval-%j.err
#SBATCH --export=ALL
set -euo pipefail

module purge
module load cuda12.1/toolkit/12.1.1     # confirm with `module avail cuda`; PyG wheels = cu121

PROJECT_DIR="/home/brageei/AEGIS"
cd "$PROJECT_DIR"

# Per-job throwaway scratch on node-local /work (not the shared repo on /home).
CACHE_ROOT="${SCRATCH:-/work/$USER}/aegis_cache_${SLURM_JOB_ID}"
mkdir -p "$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"

# Persistent spectral-feature cache on shared /home (reused from training).
export AEGIS_FEATURE_CACHE="/home/brageei/AEGIS/cache/features"
mkdir -p "$AEGIS_FEATURE_CACHE"

source "/home/brageei/AEGIS/env/bin/activate"
export PYTHONUNBUFFERED=1

# Cluster TLS-intercepts outbound HTTPS; trust the system CA bundle (HF Hub, wandb).
CA="/etc/ssl/certs/ca-certificates.crt"
[ -f "$CA" ] && export SSL_CERT_FILE="$CA" REQUESTS_CA_BUNDLE="$CA"

echo "Job started at: $(date)"; echo "Node: $(hostname)"; nvidia-smi -L || true; echo

RUN_DIR="${1:?usage: sbatch cluster/eval.slurm <run_dir> [split]}"
SPLIT="${2:-test}"

python -u ml/eval.py \
    --run-dir "$RUN_DIR" \
    --split "$SPLIT" \
    --feature-cache "$AEGIS_FEATURE_CACHE" \
    2>&1 | tee "${RUN_DIR}/eval_${SPLIT}.log"

echo "Eval complete at $(date). Metrics written under: $RUN_DIR"
