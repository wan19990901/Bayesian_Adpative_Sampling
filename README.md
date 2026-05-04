# BEACON: Bayesian Efficient Adaptive Criterion for Optimal N-stopping

BEACON is a framework for efficient LLM inference-time sampling based on Bayesian Optimal Stopping (BOS). Instead of sampling a fixed Best-of-N responses, BEACON maintains a Normal-Inverse-Gamma (NIG) posterior over the reward distribution and stops generating new samples as soon as the expected marginal gain no longer justifies the cost, reducing average sample counts by 40–80% while preserving answer quality.

---

## How It Works

For each prompt, BEACON runs the following online loop:

1. **Explore:** generate `k0` initial responses in parallel and score them with a reward model.
2. **Seed posterior:** fit a NIG posterior (μ, σ) to the `k0` reward observations.
3. **Stop or continue:**
   - Normalise the current best reward: `z = (z_k − μ_k) / σ_k`
   - Look up the expected gain `h(k, z)` from a pre-computed DP table.
   - If `h(k, z) ≤ cost / σ_k` → stop and return the best response so far.
   - Otherwise generate one more response, score it, update the posterior, and repeat.

The stopping threshold adapts automatically: hard problems (high posterior variance) get more samples; easy problems (reward converged quickly) stop early.

---

## Project Structure

```
src/
├── adapt_sample/
│   ├── sampling_comparison.py   # BEACON core + AdaBoN baseline comparison
│   └── stopping_analysis.py     # Post-hoc stopping-point analysis
├── inference/
│   ├── data/                    # Data loading, prompt construction, parsing
│   ├── math_utils/              # Math grading (math_equal, latex2sympy)
│   └── evaluation/              # Reward-model majority-vote eval
├── scripts/
│   ├── eval_gemini_math.py      # Fixed-N Gemini evaluation (baseline)
│   └── eval_gemini_beacon.py    # Adaptive BEACON evaluation (main)
├── utils/
│   └── reward_evaluator.py      # Skywork reward model (local GPU or REST API)
├── sft_iteration/               # Supervised fine-tuning data building + training
└── dpo_iteration/               # Iterative DPO training and generation
configs/
├── training.yaml                # Main training config
├── dpo_config.yaml              # DPO hyperparameters
└── deepspeed_stage{1,2,3}.json  # DeepSpeed ZeRO configs
data/
├── aime24/test.jsonl
├── aime25/test.jsonl
└── math500/test.jsonl
examples/
└── beacon_example.py
```

---

## Installation

Two separate conda environments are required: one for inference/evaluation and one for training.

### Inference environment

```bash
conda create -n vllm python=3.10.9
conda activate vllm

pip install vllm==0.5.4
pip install datasets accelerate==0.33.0 deepspeed==0.14.5
pip install transformers==4.48.1
pip install numpy==1.26.4          # numpy < 2.0 required
pip install antlr4-python3-runtime==4.7.2
pip install sympy==1.12 latex2sympy2==1.9.1 word2number==1.1
pip install numba
pip install google-genai            # for Gemini API
pip install python-dotenv tqdm requests
```

### Training environment

```bash
conda create -n rlhflow python=3.10.9
conda activate rlhflow

git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip install torch==2.1.2 torchvision torchaudio
pip install -e .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0 huggingface-hub==0.24.7
pip install transformers==4.42.2 peft==0.7.1
pip install deepspeed==0.15.4 trl==0.9.6 wandb
```

### Reward model (optional, for adaptive inference)

The Skywork reward model can be used either locally (requires GPU) or via the Skywork REST API.

```bash
pip install torch transformers  # for local GPU backend
# API key: set SKY_API_KEY or SKYWORK_API_KEY in your .env
```

---

## Adaptive Inference with BEACON

`src/scripts/eval_gemini_beacon.py` runs the full online BEACON loop: it generates samples one at a time (after the initial `k0` batch) and stops as soon as the reward model signal indicates diminishing returns.

```bash
python -m src.scripts.eval_gemini_beacon \
    --model gemini-2.5-flash \
    --data_names aime24,aime25 \
    --n_max 16 \          # hard cap on samples per problem
    --k0 3 \              # initial parallel samples
    --cost 0.05 \         # BOS cost threshold (higher = stop sooner)
    --reward_backend api \
    --output_dir results/gemini_beacon
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--n_max` | 16 | Maximum samples per problem (BOS horizon) |
| `--k0` | 3 | Initial samples before BOS activates |
| `--cost` | 0.05 | BOS cost threshold `c` |
| `--reward_backend` | api | `local` (GPU) or `api` (Skywork REST) |
| `--beacon_alpha0` | -0.5 | NIG prior α₀ (−0.5 = non-informative) |
| `--beacon_mu0` | 0.0 | NIG prior mean |
| `--beacon_adaptive_ignore` | on | Down-weight outlier observations |
| `--thinking_budget` | 8192 | Thinking tokens for Gemini 2.5+ |

BEACON **stops early** for problems where the reward signal converges quickly, and **uses more samples** for hard problems where the posterior remains uncertain.

---

## Fixed-N Gemini Evaluation

`src/scripts/eval_gemini_math.py` evaluates Gemini models with a fixed number of samples per problem. Useful as a baseline or when a reward model is not available.

```bash
python -m src.scripts.eval_gemini_math \
    --model gemini-2.5-flash \
    --data_names aime24,aime25 \
    --n_sampling 8 \
    --parallel_samples 8 \
    --thinking_budget 8192 \
    --output_dir results/gemini_fixed_n8
```

Important defaults (corrected from earlier broken runs):

| Argument | Default | Notes |
|---|---|---|
| `--prompt_type` | `cot` | Use `cot`, not `mathstral` (Mathstral-specific format) |
| `--temperature` | `1.0` | Required for sampling diversity; 0.0 gives identical outputs |
| `--max_output_tokens` | `32768` | Must be large enough for chain-of-thought |
| `--thinking_budget` | `8192` | Enables Gemini 2.5 extended thinking (~70% AIME reported) |

---

## Post-hoc Analysis with `sampling_comparison.py`

If you already have a results file (with pre-generated responses and nemotron reward scores), `src/adapt_sample/sampling_comparison.py` lets you compare BEACON against baselines by replaying the reward sequences without re-generating anything.

```bash
python -m src.adapt_sample.sampling_comparison \
    --results_file results/my_run/responses.json \
    --n_total 32 \
    --cost_thresholds 0.001 0.01 0.05 0.1 0.5 1.0 \
    --output_dir results/beacon_analysis
```

This produces accuracy/reward/value plots and a JSON summary comparing:
- **BEACON** (dynamic BOS, main method)
- **AdaBoN** — two-phase adaptive Best-of-N baseline
- Random, Self-Consistency, Best-of-N baselines

BEACON hyperparameters exposed via CLI:

```bash
    --beacon_alpha0 -0.5       # NIG prior (non-informative)
    --beacon_k0 3              # warm-up samples
    --beacon_adaptive_ignore   # robust outlier handling
    --beacon_grid_size 100     # DP table resolution
    --adabon_k0 3              # AdaBoN exploration budget
    --adabon_difficulty variance  # difficulty metric: variance | max_reward | combined
```

---

## Training Pipeline

BEACON uses an iterative self-improvement loop alternating between SFT and DPO, using BEACON-selected responses as preference data.

### Step 1 — Generate responses

```bash
bash src/scripts/run_evaluation_qwen.sh
# or for LLaMA:
bash src/scripts/run_evaluation_llama.sh
```

Generates 32 responses per problem using vLLM + a DeepInfra API backend.

### Step 2 — Score with reward model

```bash
# Local GPU
python src/utils/reward_evaluator.py

# or score AlpacaEval responses
python src/scripts/score_alpaca_with_skywork.py
```

### Step 3 — Build SFT data

```bash
python src/sft_iteration/build_sft_data.py
```

Selects the highest-reward response per problem as the SFT target.

### Step 4 — Supervised fine-tuning

```bash
bash src/scripts/sbatch_sft.sh
# or run directly:
bash src/scripts/run_sft.sh
```

### Step 5 — Iterative DPO

Pairs the highest-reward response (chosen) against a lower-reward response (rejected) per problem.

```bash
bash src/scripts/run_iter_dpo.sh
# submits to SLURM:
bash src/scripts/sbatch_dpo.sh
```

DeepSpeed configs: `configs/zero{0,2,3}.yaml` and `configs/deepspeed_stage{1,2,3}.json`.

---

## Baselines

| Baseline | Description | Reference |
|---|---|---|
| **Random** | Pick 1 response at random | — |
| **Best-of-N** | Pick highest-reward from all N | — |
| **Self-Consistency** | Majority vote over all N | Wang et al., 2023 |
| **AdaBoN** | Two-phase adaptive BoN: explore k₀ per prompt, reallocate remaining budget proportional to difficulty | — |
| **BEACON** | Sequential Bayesian stopping — our main method | — |

---

## Configuration Reference

### BEACON NIG Prior

| Parameter | Default | Meaning |
|---|---|---|
| `alpha0` | −0.5 | Shape parameter (−0.5 + k/2 → improper Jeffreys prior) |
| `nu0` | 0.0 | Prior sample count (0 = non-informative) |
| `beta0` | 0.0 | Scale parameter |
| `mu0` | 0.0 | Prior mean of reward |

Setting `alpha0=-0.5, nu0=0` gives a non-informative (improper) Jeffreys prior, which makes the posterior converge quickly from data with minimal prior bias.

### Cost Threshold

The cost `c` controls how aggressively BEACON stops:
- **Low `c` (e.g. 0.001):** sample-hungry, close to Best-of-N.
- **High `c` (e.g. 1.0):** stop very early, close to pass@1.
- **Recommended range:** 0.01–0.1 for most settings.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
