#!/usr/bin/env bash
# ── SLURM resource request ────────────────────────────────────────────────────
# To change GPU count:  update BOTH --gres AND NUM_GPUS below to match.
# To change GPU type:   update --gres and --constraint (a100 / h100 / etc.).
#SBATCH --job-name=iter_sft
#SBATCH --account=YOUR_ACCOUNT       # ← set your cluster account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4           # ← GPU type:count
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/iter_sft_%j.out
#SBATCH --error=logs/iter_sft_%j.err

# ── Tune these to match #SBATCH settings above ───────────────────────────────
export NUM_GPUS=4                        # ← must match --gres count

# Batch config for A100 (80 GB VRAM).
# For H200, run run_sft.sh directly — its defaults are already tuned for 143 GB.
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=32   # effective batch = 1 × 4 GPUs × 32 = 128

# ZeRO stage: zero3.yaml is safe for all setups.
# Switch to zero2.yaml on A100/H200 with 4+ GPUs for ~15% throughput gain.
# export ACCELERATE_CONFIG=./configs/zero2.yaml

# ── Pipeline settings ─────────────────────────────────────────────────────────
export BASE_PATH="${BASE_PATH:-outputs/iter_sft_a100}"
export NUM_ITERS=8
export BEST_OF_K=8
export MIN_REWARD=1.0
export KEEP_STRATEGY=best
# export START_ITER=3                   # uncomment to resume from a specific iter
export WANDB_PROJECT=beacon-sft
export RESERVE_GPUS_DURING_CPU=0   # SLURM owns the GPU allocation — no need to hold it

# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/../logs"
echo "Job ${SLURM_JOB_ID} started on $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
exec bash "${SCRIPT_DIR}/run_sft.sh"
