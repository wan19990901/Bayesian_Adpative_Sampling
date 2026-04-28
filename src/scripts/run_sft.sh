#!/usr/bin/env bash
set -euo pipefail
#
# Iterative rejection-sampling SFT pipeline.
# Works on local servers (H200) and under SLURM — same script, different defaults.
#
# ── Local usage ───────────────────────────────────────────────────────────────
#   bash scripts/run_sft.sh 0,1,2,3,4,5,6,7     # explicit GPUs
#   GPU_IDS=6,7 bash scripts/run_sft.sh          # same via env var
#   NUM_GPUS=4 bash scripts/run_sft.sh           # auto-pick 4 free GPUs
#   START_ITER=3 bash scripts/run_sft.sh 0,1     # resume from iter 3
#
# ── SLURM usage ───────────────────────────────────────────────────────────────
#   sbatch scripts/sbatch_sft.sh                 # edit that file to change GPU count/type
#
# ── All settings below can be overridden via environment variable ─────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── GPU selection ─────────────────────────────────────────────────────────────
GPU_IDS="${1:-${GPU_IDS:-}}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_GPUS="${MAX_GPUS:-${NUM_GPUS}}"
FREE_GPU_MAX_USED_MB="${FREE_GPU_MAX_USED_MB:-1024}"
RESERVE_GPUS_DURING_CPU="${RESERVE_GPUS_DURING_CPU:-1}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-2048}"

# ── Paths ─────────────────────────────────────────────────────────────────────
INITIAL_MODEL="${INITIAL_MODEL:-/sfs/gpfs/tardis/project/sds-rise/guangya/huggingface/hub/Qwen2.5-Math-7B}"
BASE_PATH="${BASE_PATH:-${PROJECT_ROOT}/outputs/iter_sft_h200}"
ITERATION_PREFIX="${ITERATION_PREFIX:-Train}"

ENVS_DIR="${ENVS_DIR:-/sfs/gpfs/tardis/project/sds-rise/guangya/conda_envs}"
GEN_PY="${GEN_PY:-${ENVS_DIR}/odpo-gen/bin/python}"
ACCELERATE="${ACCELERATE:-${ENVS_DIR}/odpo-train/bin/accelerate}"

# ── Pipeline ──────────────────────────────────────────────────────────────────
BEST_OF_K="${BEST_OF_K:-8}"
NUM_ITERS="${NUM_ITERS:-8}"
START_ITER="${START_ITER:-1}"
MIN_REWARD="${MIN_REWARD:-1.0}"              # only responses with reward >= this are used for SFT
KEEP_STRATEGY="${KEEP_STRATEGY:-best}"       # best: one response per prompt; all: all positives
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-1500}" # discard correct responses longer than this

# ── Training config ───────────────────────────────────────────────────────────
# H200 (143 GB VRAM) — defaults below: effective batch = 4 × gpu_count × 8 = 128 on 4 GPUs
# A100  (80 GB VRAM) — set batch=1, grad_accum=32 in sbatch_sft.sh → effective 128 on 4 GPUs
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# ── DeepSpeed / ZeRO config ───────────────────────────────────────────────────
# zero3.yaml  — default; safe for all GPU counts and types
# zero2.yaml  — 10-20% faster on H200/A100 where VRAM is ample
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-./configs/zero3.yaml}"

# ─────────────────────────────────────────────────────────────────────────────

resolve_gpus() {
  if [[ -n "${SLURM_JOB_ID:-}" && -z "${GPU_IDS}" ]]; then
    GPU_IDS="${CUDA_VISIBLE_DEVICES:?SLURM set no CUDA_VISIBLE_DEVICES}"
    return 0
  fi
  if [[ -n "${GPU_IDS}" ]]; then
    local cnt; cnt="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
    if [[ "${cnt}" -gt "${MAX_GPUS}" ]]; then
      echo "ERROR: GPU_IDS has ${cnt} GPUs but MAX_GPUS=${MAX_GPUS}." >&2; return 1
    fi
    return 0
  fi
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
  local json_output="$4"   # shard file prefix
  local reward_json="$5"   # merged reward JSON path
  local sft_jsonl="$6"     # filtered SFT training data
  local model_output="$7"  # trained model directory

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
  if [[ -f "${reward_json}" ]]; then
    echo "  [skip] merged reward file already exists: ${reward_json}"
  else
    local -a merge_reserve_pids=()
    if [[ "${RESERVE_GPUS_DURING_CPU}" == "1" ]]; then
      for gpu in "${gpu_arr[@]}"; do
        merge_reserve_pids+=("$(start_gpu_reservation "${gpu}" "${reward_json}_merge_gpu${gpu}_reserve.log")")
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
with open('${reward_json}', 'w') as f:
    json.dump(data, f)
print('Merged', len(data), 'samples ->', '${reward_json}')
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

  # ── Filter to positive examples for SFT ─────────────────────────────────
  if [[ -f "${sft_jsonl}" ]]; then
    echo "  [skip] SFT data already exists: ${sft_jsonl}"
  else
    local -a filter_reserve_pids=()
    if [[ "${RESERVE_GPUS_DURING_CPU}" == "1" ]]; then
      for gpu in "${gpu_arr[@]}"; do
        filter_reserve_pids+=("$(start_gpu_reservation "${gpu}" "${sft_jsonl}_filter_gpu${gpu}_reserve.log")")
      done
    fi
    if ! "${GEN_PY}" ./sft_iteration/build_sft_data.py \
      --reward_json_path "${reward_json}" \
      --output_jsonl_path "${sft_jsonl}" \
      --min_reward "${MIN_REWARD}" \
      --keep_strategy "${KEEP_STRATEGY}" \
      --max_response_tokens "${MAX_RESPONSE_TOKENS}"
    then
      for pid in "${filter_reserve_pids[@]}"; do
        stop_gpu_reservation "${pid}"
      done
      return 1
    fi
    for pid in "${filter_reserve_pids[@]}"; do
      stop_gpu_reservation "${pid}"
    done
  fi

  # ── SFT training ─────────────────────────────────────────────────────────
  if [[ -f "${model_output}/config.json" ]]; then
    echo "  [skip] trained model already exists: ${model_output}"
  else
    local free_port; free_port="$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE}" launch \
      --config_file "${ACCELERATE_CONFIG}" \
      --num_processes "${gpu_count}" \
      --main_process_port "${free_port}" \
      ./sft_iteration/run_sft.py \
      --model_name_or_path "${model_path}" \
      --train_file "${sft_jsonl}" \
      --output_dir "${model_output}" \
      --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
      --learning_rate "${LEARNING_RATE}" \
      --max_length 4096 \
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --logging_steps 5 \
      --bf16 true
  fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in $(seq "${START_ITER}" "${NUM_ITERS}"); do
  iteration_name="Qwen_numina_sft_iter${i}"
  dataset_name="RLHFlow/numia_prompt_dpo${i}"
  json_output="${BASE_PATH}/${ITERATION_PREFIX}${i}_${iteration_name}"
  reward_json="${json_output}_reward.json"
  sft_jsonl="${json_output}_sft.jsonl"
  model_output="${BASE_PATH}/${iteration_name}"

  if [[ "${i}" -eq 1 ]]; then
    model_path="${INITIAL_MODEL}"
  else
    prev=$((i - 1))
    model_path="${BASE_PATH}/Qwen_numina_sft_iter${prev}"
  fi

  echo ""
  echo "════════════════════════════════════════"
  echo "[iter ${i}/${NUM_ITERS}]  model:   ${model_path}"
  echo "[iter ${i}/${NUM_ITERS}]  dataset: ${dataset_name}"
  echo "════════════════════════════════════════"

  run_iteration "${iteration_name}" "${model_path}" "${dataset_name}" \
    "${json_output}" "${reward_json}" "${sft_jsonl}" "${model_output}"
done

echo ""
echo "════════════════════════════════════════"
echo "ALL ${NUM_ITERS} SFT ITERATIONS DONE"
echo "Final model: ${BASE_PATH}/Qwen_numina_sft_iter${NUM_ITERS}"
echo "════════════════════════════════════════"
