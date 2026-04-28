#!/usr/bin/env bash
set -euo pipefail
#
# Iterative Online DPO pipeline.
# Works on local servers (H200) and under SLURM — same script, different defaults.
#
# ── Local usage ───────────────────────────────────────────────────────────────
#   bash scripts/run_dpo.sh 0,1,2,3,4,5,6,7     # explicit GPUs
#   GPU_IDS=6,7 bash scripts/run_dpo.sh          # same via env var
#   NUM_GPUS=4 bash scripts/run_dpo.sh           # auto-pick 4 free GPUs
#   START_ITER=3 bash scripts/run_dpo.sh 0,1     # resume from iter 3
#
# ── SLURM usage ───────────────────────────────────────────────────────────────
#   sbatch scripts/sbatch_dpo.sh                 # edit that file to change GPU count/type
#
# ── All settings below can be overridden via environment variable ─────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── GPU selection ─────────────────────────────────────────────────────────────
# Under SLURM, GPUs come from CUDA_VISIBLE_DEVICES (set automatically by SLURM).
# Locally, pass as first arg, set GPU_IDS, or let NUM_GPUS auto-pick free ones.
GPU_IDS="${1:-${GPU_IDS:-}}"
NUM_GPUS="${NUM_GPUS:-8}"                    # used only for local auto-selection
MAX_GPUS="${MAX_GPUS:-${NUM_GPUS}}"          # cap on auto-selection
FREE_GPU_MAX_USED_MB="${FREE_GPU_MAX_USED_MB:-1024}"
RESERVE_GPUS_DURING_CPU="${RESERVE_GPUS_DURING_CPU:-1}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-2048}"

# ── Paths ─────────────────────────────────────────────────────────────────────
INITIAL_MODEL="${INITIAL_MODEL:-/sfs/gpfs/tardis/project/sds-rise/guangya/huggingface/hub/Qwen2.5-Math-7B}"
BASE_PATH="${BASE_PATH:-${PROJECT_ROOT}/outputs/iter_dpo_h200}"
ITERATION_PREFIX="${ITERATION_PREFIX:-Train}"

ENVS_DIR="${ENVS_DIR:-/sfs/gpfs/tardis/project/sds-rise/guangya/conda_envs}"
GEN_PY="${GEN_PY:-${ENVS_DIR}/odpo-gen/bin/python}"
ACCELERATE="${ACCELERATE:-${ENVS_DIR}/odpo-train/bin/accelerate}"

# ── Pipeline ──────────────────────────────────────────────────────────────────
BEST_OF_K="${BEST_OF_K:-8}"
NUM_ITERS="${NUM_ITERS:-8}"
START_ITER="${START_ITER:-1}"

# ── Training config ───────────────────────────────────────────────────────────
# H200 (143 GB VRAM) — defaults below: effective batch = 2 × gpu_count × 32 = 128 on 2 GPUs
# Qwen2.5-7B has 151k vocab; logit tensor (fp32) = batch×2×seq×151k×4B ≈ 10 GB at batch=2 (vs 20 GB at batch=4)
# A100  (80 GB VRAM) — set batch=2, eval_batch=1, grad_accum=16 in sbatch_dpo.sh → effective 128 on 4 GPUs
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"

# ── DeepSpeed / ZeRO config ───────────────────────────────────────────────────
# zero3.yaml  — required for DPO (policy + reference model both loaded per GPU)
# zero2.yaml  — fine for SFT (single model); not safe for DPO on 2 GPUs
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-./configs/zero3.yaml}"

# Reduces memory fragmentation so large contiguous ZeRO-3 all-gather buffers
# can be satisfied even when total free memory is barely sufficient.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─────────────────────────────────────────────────────────────────────────────

resolve_gpus() {
  # SLURM: trust its allocation
  if [[ -n "${SLURM_JOB_ID:-}" && -z "${GPU_IDS}" ]]; then
    GPU_IDS="${CUDA_VISIBLE_DEVICES:?SLURM set no CUDA_VISIBLE_DEVICES}"
    return 0
  fi
  # Explicit: validate count
  if [[ -n "${GPU_IDS}" ]]; then
    local cnt; cnt="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
    if [[ "${cnt}" -gt "${MAX_GPUS}" ]]; then
      echo "ERROR: GPU_IDS has ${cnt} GPUs but MAX_GPUS=${MAX_GPUS}." >&2; return 1
    fi
    return 0
  fi
  # Auto-select free GPUs
  command -v nvidia-smi >/dev/null 2>&1 \
    || { echo "ERROR: GPU_IDS not set and nvidia-smi unavailable." >&2; return 1; }
  local requested="${NUM_GPUS}"
  [[ "${MAX_GPUS}" -lt "${requested}" ]] && requested="${MAX_GPUS}"
  GPU_IDS="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v thr="${FREE_GPU_MAX_USED_MB}" '{
          gsub(/ /,"",$1); gsub(/ /,"",$2)
          if (($2+0)<thr) print $1","$2
        }' \
      | sort -t, -k2,2n \
      | head -n "${requested}" \
      | awk -F',' '{print $1}' \
      | paste -sd ',' -
  )"
  [[ -n "${GPU_IDS}" ]] \
    || { echo "ERROR: no free GPUs under ${FREE_GPU_MAX_USED_MB} MB. Set GPU_IDS or raise FREE_GPU_MAX_USED_MB." >&2; return 1; }
}

resolve_gpus
gpu_count="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
mkdir -p "${BASE_PATH}"

echo "════════════════════════════════════════"
echo "GPU IDs   : ${GPU_IDS} (${gpu_count} GPUs)"
echo "Base path : ${BASE_PATH}"
echo "Model     : ${INITIAL_MODEL}"
echo "Iters     : ${START_ITER} → ${NUM_ITERS}"
echo "Batch     : ${PER_DEVICE_TRAIN_BATCH_SIZE} per-device × ${gpu_count} GPUs × ${GRADIENT_ACCUMULATION_STEPS} accum = $(( PER_DEVICE_TRAIN_BATCH_SIZE * gpu_count * GRADIENT_ACCUMULATION_STEPS )) effective"
echo "════════════════════════════════════════"

run_iteration() {
  local iteration_name="$1"
  local model_path="$2"
  local dataset_name="$3"
  local json_output="$4"   # shard file prefix (e.g. .../Train1_iter1)
  local model_output="$5"  # final merged reward JSON path

  IFS=',' read -r -a gpu_arr <<< "${GPU_IDS}"

  start_gpu_reservation() {
    local gpu="$1"
    local log_file="$2"
    [[ "${RESERVE_GPUS_DURING_CPU}" == "1" ]] || return 0
    CUDA_VISIBLE_DEVICES="${gpu}" GPU_RESERVE_MB="${GPU_RESERVE_MB}" "${GEN_PY}" -c '
import os, time, torch
reserve_mb = int(os.environ.get("GPU_RESERVE_MB", "2048"))
buf = torch.empty(reserve_mb * 1024 * 1024, dtype=torch.uint8, device="cuda")
buf.zero_()
print(f"reserved {reserve_mb} MB on logical cuda:0", flush=True)
while True:
    time.sleep(60)
' > "${log_file}" 2>&1 &
    echo "$!"
  }

  stop_gpu_reservation() {
    local pid="$1"
    [[ -n "${pid}" ]] || return 0
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  }

  # ── Generation + Reward: pipelined per GPU ───────────────────────────────
  # Skip if every shard's reward file already exists (safe to re-run partially
  # failed iterations — only missing shards are regenerated).
  local skip_gen=true
  for i in "${!gpu_arr[@]}"; do
    [[ -f "${json_output}_shard${i}_reward.json" ]] || { skip_gen=false; break; }
  done

  if [[ "${skip_gen}" == "true" ]]; then
    echo "  [skip] all shard reward files exist, skipping generation+reward"
  else
    local reward_workers=$(( 32 / gpu_count ))
    [[ "${reward_workers}" -lt 1 ]] && reward_workers=1
    local -a pids=()
    for i in "${!gpu_arr[@]}"; do
      if [[ -f "${json_output}_shard${i}_reward.json" ]]; then
        echo "  [skip] shard ${i} reward already exists"
        continue
      fi
      local shard_log="${json_output}_shard${i}.log"
      echo "  [shard ${i}] starting on GPU ${gpu_arr[$i]} — log: ${shard_log}"
      (
        reserve_pid=""
        cleanup_reservation() { stop_gpu_reservation "${reserve_pid}"; }
        trap cleanup_reservation EXIT
        CUDA_VISIBLE_DEVICES="${gpu_arr[$i]}" "${GEN_PY}" ./generation/gen_hf.py \
          --model_name_or_path "${model_path}" \
          --dataset_name_or_path "${dataset_name}" \
          --output_dir "${json_output}" \
          --K "${BEST_OF_K}" \
          --temperature 1.0 \
          --local_index "$i" \
          --my_world_size "${gpu_count}" \
        && reserve_pid="$(start_gpu_reservation "${gpu_arr[$i]}" "${json_output}_shard${i}_gpu_reserve.log")" \
        && REWARD_NUM_PROC="${reward_workers}" "${GEN_PY}" ./reward_labeling.py \
          --dataset_name_or_path "${json_output}${i}.json" \
          --output_dir "${json_output}_shard${i}_reward.json"
      ) > "${shard_log}" 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || { echo "ERROR: shard pipeline failed (PID ${pid}) — check ${json_output}_shard*.log" >&2; exit 1; }
    done
    echo "  [done] all shards complete"
  fi

  # ── Merge reward-labeled shards ──────────────────────────────────────────
  if [[ -f "${model_output}" ]]; then
    echo "  [skip] merged reward file already exists: ${model_output}"
  else
    local -a merge_reserve_pids=()
    if [[ "${RESERVE_GPUS_DURING_CPU}" == "1" ]]; then
      for gpu in "${gpu_arr[@]}"; do
        merge_reserve_pids+=("$(start_gpu_reservation "${gpu}" "${model_output}_merge_gpu${gpu}_reserve.log")")
      done
    fi
    if ! "${GEN_PY}" -c "
import json, random
data = []
for i in range(${gpu_count}):
    with open('${json_output}_shard' + str(i) + '_reward.json') as f:
        data.extend(json.load(f))
random.seed(42)
random.shuffle(data)
with open('${model_output}', 'w') as f:
    json.dump(data, f)
print('Merged', len(data), 'samples ->', '${model_output}')
"
    then
      for pid in "${merge_reserve_pids[@]}"; do
        stop_gpu_reservation "${pid}"
      done
      return 1
    fi
    for pid in "${merge_reserve_pids[@]}"; do
      stop_gpu_reservation "${pid}"
    done
  fi

  # ── DPO training ─────────────────────────────────────────────────────────
  local final_ckpt="${BASE_PATH}/${iteration_name}/final_checkpoint"
  if [[ -d "${final_ckpt}" ]]; then
    echo "  [skip] final checkpoint already exists: ${final_ckpt}"
  else
    cat > "${BASE_PATH}/dpo_config_${iteration_name}.yaml" <<EOT
run_name: ${iteration_name}
output_dir: ${BASE_PATH}/${iteration_name}
model_name_or_path: ${model_path}
ref_model: ${model_path}
learning_rate: 5.0e-7
num_train_epochs: 2
logging_steps: 2
gradient_checkpointing: true
do_train: true
do_eval: true
eval_steps: 10000
choose_type: max_min
train_dir: ${model_output}
eval_dir: ${model_output}
loss_type: sigmoid
lr_scheduler_type: cosine
max_length: 4096
max_prompt_length: 1000
eval_strategy: steps
bf16: true
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
per_device_eval_batch_size: ${PER_DEVICE_EVAL_BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
report_to: wandb
label_smoothing: 0.1
eot_token: <|im_end|>
save_strategy: epoch
EOT

    local free_port; free_port="$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE}" launch \
      --config_file "${ACCELERATE_CONFIG}" \
      --num_processes "${gpu_count}" \
      --main_process_port "${free_port}" \
      dpo_iteration/run_dpo.py "${BASE_PATH}/dpo_config_${iteration_name}.yaml"
  fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in $(seq "${START_ITER}" "${NUM_ITERS}"); do
  iteration_name="Qwen_numina_iter${i}"
  dataset_name="RLHFlow/numia_prompt_dpo${i}"
  json_output="${BASE_PATH}/${ITERATION_PREFIX}${i}_${iteration_name}"
  model_output="${BASE_PATH}/${ITERATION_PREFIX}${i}_${iteration_name}_reward.json"

  if [[ "${i}" -eq 1 ]]; then
    model_path="${INITIAL_MODEL}"
  else
    prev=$((i - 1))
    model_path="${BASE_PATH}/Qwen_numina_iter${prev}/final_checkpoint"
  fi

  echo ""
  echo "════════════════════════════════════════"
  echo "[iter ${i}/${NUM_ITERS}]  model:   ${model_path}"
  echo "[iter ${i}/${NUM_ITERS}]  dataset: ${dataset_name}"
  echo "════════════════════════════════════════"

  run_iteration "${iteration_name}" "${model_path}" "${dataset_name}" "${json_output}" "${model_output}"
done

echo ""
echo "════════════════════════════════════════"
echo "ALL ${NUM_ITERS} DPO ITERATIONS DONE"
echo "Final model: ${BASE_PATH}/Qwen_numina_iter${NUM_ITERS}/final_checkpoint"
echo "════════════════════════════════════════"
