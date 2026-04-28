#!/usr/bin/env bash
#SBATCH --job-name=beacon_test
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/sfs/gpfs/tardis/project/sds-rise/guangya/Beacon_Training_Code/logs/test_envs_%j.out
#SBATCH --error=/sfs/gpfs/tardis/project/sds-rise/guangya/Beacon_Training_Code/logs/test_envs_%j.err

set -euo pipefail

REPO=/sfs/gpfs/tardis/project/sds-rise/guangya/Beacon_Training_Code
GEN_PY=/sfs/gpfs/tardis/project/sds-rise/guangya/conda_envs/odpo-gen/bin/python
TRAIN_PY=/sfs/gpfs/tardis/project/sds-rise/guangya/conda_envs/odpo-train/bin/python
ACCELERATE=/sfs/gpfs/tardis/project/sds-rise/guangya/conda_envs/odpo-train/bin/accelerate
MODEL=/sfs/gpfs/tardis/project/sds-rise/guangya/huggingface/hub/Qwen2.5-Math-7B
export WORK=/sfs/gpfs/tardis/project/sds-rise/guangya/beacon_test_${SLURM_JOB_ID}

mkdir -p "$WORK" "${REPO}/logs"
cd "$REPO"

echo "========================================"
echo "NODE: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "WORK: $WORK"
echo "========================================"

# ── Stage 1: create tiny local dataset ───────────────────────────────────────
echo ""
echo "=== STAGE 1: build tiny dataset ==="
$GEN_PY - <<'PYEOF'
import json, os
problems = [
    {"problem": "What is 2 + 2?",         "gt": "4"},
    {"problem": "What is 3 * 5?",         "gt": "15"},
    {"problem": "What is 10 - 3?",        "gt": "7"},
    {"problem": "What is 12 / 4?",        "gt": "3"},
    {"problem": "What is 2^3?",           "gt": "8"},
    {"problem": "Simplify: sqrt(16).",    "gt": "4"},
    {"problem": "What is 100 mod 7?",     "gt": "2"},
    {"problem": "What is 5! (factorial)?","gt": "120"},
]
out = os.environ["WORK"]
with open(f"{out}/mini_dataset.jsonl", "w") as f:
    for p in problems:
        f.write(json.dumps(p) + "\n")
print(f"Wrote {len(problems)} problems to {out}/mini_dataset.jsonl")
PYEOF
echo "Stage 1 DONE"

# ── Stage 2: vllm generation ──────────────────────────────────────────────────
echo ""
echo "=== STAGE 2: odpo-gen vllm generation (K=4) ==="
$GEN_PY "$REPO/generation/gen_hf.py" \
    --model_name_or_path "$MODEL" \
    --dataset_name_or_path "$WORK/mini_dataset.jsonl" \
    --output_dir "$WORK/gen_out" \
    --K 4 \
    --temperature 0.8 \
    --local_index 0 \
    --my_world_size 1 \
    --dataset_key problem \
    --gt_key gt \
    --max_new_tokens 256
echo "Stage 2 DONE — output: ${WORK}/gen_out0.json"

# ── Stage 3: merge ────────────────────────────────────────────────────────────
echo ""
echo "=== STAGE 3: merge (1 worker) ==="
$GEN_PY "$REPO/generation/merge_data.py" \
    --base_path "$WORK/gen_out" \
    --output_dir "$WORK/gen_merged.jsonl" \
    --num_datasets 1
echo "Stage 3 DONE — output: ${WORK}/gen_merged.jsonl"

# ── Stage 4: reward labeling ──────────────────────────────────────────────────
echo ""
echo "=== STAGE 4: reward labeling ==="
$GEN_PY "$REPO/reward_labeling.py" \
    --dataset_name_or_path "$WORK/gen_merged.jsonl" \
    --output_dir "$WORK/reward.json"
echo "Stage 4 DONE — output: ${WORK}/reward.json"

# peek at results
$GEN_PY - <<PYEOF
import json
with open("${WORK}/reward.json") as f:
    data = json.load(f)
print(f"  samples: {len(data)}")
for i, s in enumerate(data[:3]):
    print(f"  [{i}] rewards={s['rewards']}  gt={s['gt']!r}")
    print(f"       resp[0][:80]={s['responses'][0][:80]!r}")
PYEOF

# ── Stage 5: DPO training (5 steps, 1 GPU, ZeRO-3) ───────────────────────────
echo ""
echo "=== STAGE 5: DPO training (5 optimizer steps, ZeRO-3, 1 GPU) ==="

cat > "$WORK/dpo_test_config.yaml" <<EOT
run_name: beacon_test_dpo
output_dir: ${WORK}/dpo_out
model_name_or_path: ${MODEL}
ref_model: ${MODEL}
train_dir: ${WORK}/reward.json
eval_dir: ${WORK}/reward.json
learning_rate: 5.0e-7
max_steps: 5
logging_steps: 1
gradient_checkpointing: true
do_train: true
do_eval: false
choose_type: max_min
loss_type: sigmoid
lr_scheduler_type: cosine
max_length: 512
max_prompt_length: 256
bf16: true
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 1
report_to: none
label_smoothing: 0.1
eot_token: "<|im_end|>"
save_strategy: "no"
EOT

$ACCELERATE launch \
    --config_file "$REPO/configs/zero3_1gpu.yaml" \
    --num_processes 2 \
    "$REPO/dpo_iteration/run_dpo.py" \
    "$WORK/dpo_test_config.yaml"

echo "Stage 5 DONE"

# ── Stage 6: build SFT data from reward JSON ─────────────────────────────────
echo ""
echo "=== STAGE 6: build SFT data ==="
$GEN_PY "$REPO/sft_iteration/build_sft_data.py" \
    --reward_json_path "$WORK/reward.json" \
    --output_jsonl_path "$WORK/sft_data.jsonl" \
    --min_reward 1.0 \
    --keep_strategy best
echo "Stage 6 DONE — output: ${WORK}/sft_data.jsonl"

# ── Stage 7: SFT training (5 steps, 2 GPU, ZeRO-3) ───────────────────────────
echo ""
echo "=== STAGE 7: SFT training (5 optimizer steps, ZeRO-3, 2 GPU) ==="
$ACCELERATE launch \
    --config_file "$REPO/configs/zero3.yaml" \
    --num_processes 2 \
    "$REPO/sft_iteration/run_sft.py" \
    --model_name_or_path "$MODEL" \
    --train_file "$WORK/sft_data.jsonl" \
    --output_dir "$WORK/sft_out" \
    --max_steps 5 \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --max_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 true

echo "Stage 7 DONE"

echo ""
echo "========================================"
echo "ALL STAGES PASSED — end-to-end test OK"
echo "Artifacts in: $WORK"
echo "========================================"
