#!/usr/bin/env python3
"""
Adaptive BEACON evaluation: combines Gemini generation with the BEACON
Bayesian Optimal Stopping rule so the number of API calls per problem is
decided on-the-fly rather than fixed in advance.

Pipeline per problem
--------------------
Phase 1  Generate k0 responses in parallel (initial exploration).
Phase 2  Score all k0 responses with the reward model → seed the NIG posterior.
Phase 3  BEACON BOS loop:
           a. Compute normalised best z_val and h(k, z_val) from the DP table.
           b. If h <= cost/sigma_k  →  expected marginal gain is too small, stop.
           c. Else generate one more response, score it, update posterior, repeat.
Final    Evaluate the highest-reward response for mathematical correctness.

Usage
-----
python -m src.scripts.eval_gemini_beacon \\
    --model gemini-2.5-flash \\
    --data_names aime24,aime25 \\
    --n_max 16 --k0 3 --cost 0.05 \\
    --reward_backend api \\
    --output_dir results/gemini_beacon
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.inference.data.data_loader import load_data
from src.inference.data.parser import (
    choice_answer_clean,
    parse_ground_truth,
    parse_question,
    run_execute,
)
from src.inference.data.utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from src.inference.math_utils.grader import math_equal
from src.adapt_sample.sampling_comparison import (
    SamplingComparison,
    compute_initial_parameters,
    update_parameters,
)
from src.utils.reward_evaluator import RewardEvaluator


for _env in [".env", os.path.expanduser("~/.env")]:
    if os.path.exists(_env):
        load_dotenv(_env)
        break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="BEACON adaptive evaluation with Gemini + reward model."
    )
    # ── Generation ──────────────────────────────────────────────────────────
    p.add_argument("--model", required=True,
                   help="Gemini model name, e.g. gemini-2.5-flash")
    p.add_argument("--api_key", default=None,
                   help="Gemini API key (falls back to GEMINI_API_KEY / GOOGLE_API_KEY).")
    p.add_argument("--data_names", default="aime24,aime25",
                   help="Comma-separated benchmark names.")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="results/gemini_beacon")
    p.add_argument("--prompt_type", default="cot",
                   help="Prompt template (cot | direct | qwen25-math-cot).")
    p.add_argument("--split", default="test")
    p.add_argument("--num_test_sample", default=-1, type=int)
    p.add_argument("--start", default=0, type=int)
    p.add_argument("--end", default=-1, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--temperature", default=1.0, type=float)
    p.add_argument("--top_p", default=None, type=float)
    p.add_argument("--max_output_tokens", default=32768, type=int)
    p.add_argument("--thinking_budget", default=8192, type=int,
                   help="Thinking token budget for Gemini 2.5+ (0 to disable).")
    p.add_argument("--no_thinking", action="store_true",
                   help="Disable thinking mode even on Gemini 2.5+ models.")
    p.add_argument("--max_retries", default=3, type=int)
    p.add_argument("--sleep", default=0.0, type=float,
                   help="Seconds to sleep between sequential API calls.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_every", default=1, type=int)
    # ── BEACON stopping ─────────────────────────────────────────────────────
    p.add_argument("--n_max", default=16, type=int,
                   help="Hard cap on samples per problem (BOS horizon).")
    p.add_argument("--k0", default=3, type=int,
                   help="Initial samples generated in parallel before BOS activates.")
    p.add_argument("--cost", default=0.05, type=float,
                   help="BOS cost threshold c. Higher = stop sooner.")
    p.add_argument("--beacon_alpha0", default=-0.5, type=float,
                   help="NIG prior alpha0 (-0.5 = non-informative Jeffreys prior).")
    p.add_argument("--beacon_nu0", default=0.0, type=float)
    p.add_argument("--beacon_beta0", default=0.0, type=float)
    p.add_argument("--beacon_mu0", default=0.0, type=float,
                   help="NIG prior mean of reward distribution.")
    p.add_argument("--beacon_adaptive_ignore", action="store_true", default=True,
                   help="Down-weight extreme low-reward observations (default: on).")
    p.add_argument("--beacon_no_adaptive_ignore",
                   dest="beacon_adaptive_ignore", action="store_false")
    p.add_argument("--beacon_adaptive_alpha", default=0.01, type=float,
                   help="Tail probability threshold for adaptive-ignore rule.")
    p.add_argument("--beacon_grid_size", default=100, type=int,
                   help="Resolution of the pre-computed h-index DP look-up table.")
    # ── Reward model ─────────────────────────────────────────────────────────
    p.add_argument("--reward_backend", default="api", choices=["local", "api"],
                   help="'local' loads Skywork on GPU; 'api' calls Skywork REST API.")
    p.add_argument("--reward_model",
                   default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
                   help="HuggingFace model name (only used for --reward_backend local).")
    p.add_argument("--reward_device", default=None,
                   help="CUDA device for local reward model, e.g. cuda:0.")
    p.add_argument("--reward_api_key", default=None,
                   help="Skywork API key (falls back to SKY_API_KEY env var).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Gemini generation helpers
# ---------------------------------------------------------------------------

def _supports_thinking(model_name: str) -> bool:
    return "gemini-2.5" in model_name or "gemini-exp" in model_name


def build_gemini_client(api_key):
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --api_key.")
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        pass
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        return genai_legacy
    except ImportError as exc:
        raise SystemExit("pip install google-genai  # or google-generativeai") from exc


def generate_one(client, model_name: str, prompt: str, args) -> str:
    use_thinking = _supports_thinking(model_name) and not args.no_thinking
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            if hasattr(client, "models"):
                from google.genai import types as gtypes
                cfg = {
                    "temperature": args.temperature,
                    "max_output_tokens": args.max_output_tokens,
                }
                if args.top_p is not None:
                    cfg["top_p"] = args.top_p
                if use_thinking and args.thinking_budget > 0:
                    cfg["thinking_config"] = gtypes.ThinkingConfig(
                        thinking_budget=args.thinking_budget
                    )
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(**cfg),
                )
                return resp.text or ""
            else:
                model = client.GenerativeModel(model_name)
                cfg = {
                    "temperature": args.temperature,
                    "max_output_tokens": args.max_output_tokens,
                }
                if args.top_p is not None:
                    cfg["top_p"] = args.top_p
                resp = model.generate_content(
                    prompt,
                    generation_config=client.types.GenerationConfig(**cfg),
                )
                return getattr(resp, "text", "") or ""
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(min(2 ** attempt, 30))
    return f"ERROR: {last_error}"


def generate_batch(client, model_name: str, prompt: str, n: int, args):
    """Generate n samples concurrently (used for the k0 initial phase)."""
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(generate_one, client, model_name, prompt, args)
            for _ in range(n)
        ]
        return [f.result().strip() for f in futures]


# ---------------------------------------------------------------------------
# BEACON adaptive sampling loop
# ---------------------------------------------------------------------------

def beacon_sample(client, model_name: str, prompt: str,
                  reward_eval: RewardEvaluator,
                  bos: SamplingComparison,
                  args) -> dict:
    """
    Run BEACON adaptive sampling for a single prompt.

    Returns a dict with:
        responses    : list of all generated response strings
        rewards      : corresponding reward scores
        selected_idx : index of the best (highest-reward) response
        samples_used : total LLM calls made
    """
    k0 = max(3, args.k0)

    # ── Phase 1: parallel initial exploration ────────────────────────────────
    responses = generate_batch(client, model_name, prompt, k0, args)
    rewards = [reward_eval.compute_reward(prompt, r) for r in responses]

    if len(rewards) < 3:
        best_idx = int(np.argmax(rewards))
        return {
            "responses": responses,
            "rewards": rewards,
            "selected_idx": best_idx,
            "samples_used": len(responses),
        }

    # ── Phase 2: seed the NIG posterior from k0 observations ────────────────
    z_k, mu_k, sigma_k = compute_initial_parameters(
        np.array(rewards, dtype=np.float64),
        args.beacon_alpha0, args.beacon_nu0,
        args.beacon_beta0, args.beacon_mu0,
    )
    best_idx = int(np.argmax(rewards))

    # ── Phase 3: BOS sequential stopping loop ───────────────────────────────
    for k in range(k0, args.n_max):
        if sigma_k <= 1e-9:
            break  # degenerate posterior — variance collapsed, stop

        z_val = (z_k - mu_k) / sigma_k
        c_val = args.cost / sigma_k
        h_val = bos._get_h_value(k=k, z_val=z_val)

        if h_val <= c_val:
            break  # expected marginal gain no longer justifies cost

        # Generate and score one additional sample
        new_response = generate_one(client, model_name, prompt, args).strip()
        new_reward = reward_eval.compute_reward(prompt, new_response)
        responses.append(new_response)
        rewards.append(new_reward)

        # Update NIG posterior
        z_k, mu_k, sigma_k = update_parameters(
            z_k, mu_k, sigma_k, new_reward, k,
            args.beacon_alpha0, args.beacon_nu0,
            args.beacon_beta0, args.beacon_mu0,
            use_adaptive_ignore=args.beacon_adaptive_ignore,
            alpha=args.beacon_adaptive_alpha,
        )

        # z_k = max(old_z_k, new_reward) after update
        if new_reward == z_k:
            best_idx = len(rewards) - 1

        if args.sleep:
            time.sleep(args.sleep)

    return {
        "responses": responses,
        "rewards": rewards,
        "selected_idx": best_idx,
        "samples_used": len(responses),
    }


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def is_multi_choice(answer):
    return all(c in "ABCDE" for c in str(answer))


def evaluate_dataset(client, data_name: str,
                     reward_eval: RewardEvaluator,
                     bos: SamplingComparison,
                     args) -> dict:
    examples = load_data(data_name, args.split, args.data_dir)
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    examples = examples[args.start: len(examples) if args.end == -1 else args.end]

    prefix = (
        f"{args.split}_{args.prompt_type}_beacon"
        f"_k{args.k0}_c{args.cost}_seed{args.seed}_t{args.temperature}"
    )
    out_file = os.path.join(
        args.output_dir, data_name,
        f"{prefix}_s{args.start}_e{args.end}.jsonl",
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    processed_samples = []
    if not args.overwrite and os.path.exists(out_file):
        processed_samples = list(load_jsonl(out_file))
        processed_idxs = {s["idx"] for s in processed_samples}
        examples = [e for e in examples if e["idx"] not in processed_idxs]

    prompt_args = SimpleNamespace(
        prompt_type=args.prompt_type, num_shots=0, adapt_few_shot=False
    )
    samples = []
    start_time = time.time()

    for example in tqdm(examples, desc=data_name):
        example["question"] = parse_question(example, data_name)
        if not example["question"]:
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        prompt = construct_prompt(example, data_name, prompt_args)

        result = beacon_sample(client, args.model, prompt, reward_eval, bos, args)
        selected = result["responses"][result["selected_idx"]]

        pred, report = run_execute(None, selected, args.prompt_type, data_name)
        if gt_ans in "ABCDE" and pred not in "ABCDE":
            pred = choice_answer_clean(selected)
        elif is_multi_choice(gt_ans) and not is_multi_choice(pred):
            pred = "".join(c for c in str(pred) if c in "ABCDE")

        sample = {
            "idx": example["idx"],
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "code": result["responses"],        # all generated responses
            "rewards": result["rewards"],       # reward score per response
            "selected_idx": result["selected_idx"],
            "samples_used": result["samples_used"],
            "pred": [pred],
            "report": [report],
        }
        for key in ["level", "type", "solution", "answer", "dataset"]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

        if args.save_every and len(samples) % args.save_every == 0:
            save_jsonl(processed_samples + samples, out_file)

    all_samples = processed_samples + samples
    timeout_count = 0
    for sample in all_samples:
        _, sample["gt"] = parse_ground_truth(sample, data_name)
        try:
            correct = bool(math_equal(sample["pred"][0], sample["gt"], timeout=True))
        except Exception:
            timeout_count += 1
            correct = False
        sample["score"] = [correct]

    scores = [s["score"] for s in all_samples]
    avg_samples = (
        float(np.mean([s["samples_used"] for s in all_samples]))
        if all_samples else 0.0
    )
    metrics = {
        "num_samples": len(all_samples),
        "timeout_samples": timeout_count,
        "acc": round(100 * sum(r[0] for r in scores) / len(scores), 1) if scores else 0.0,
        "avg_samples_used": round(avg_samples, 2),
        "model": args.model,
        "k0": args.k0,
        "n_max": args.n_max,
        "cost_threshold": args.cost,
        "reward_backend": args.reward_backend,
        "beacon_alpha0": args.beacon_alpha0,
        "beacon_nu0": args.beacon_nu0,
        "time_use_in_second": round(time.time() - start_time, 1),
    }

    save_jsonl(all_samples, out_file)
    with open(out_file.replace(".jsonl", "_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    print("Building Gemini client...")
    client = build_gemini_client(args.api_key)

    print(f"Initialising reward model (backend={args.reward_backend})...")
    reward_eval = RewardEvaluator(
        model_name=args.reward_model,
        backend=args.reward_backend,
        device=args.reward_device,
        skywork_api_key=args.reward_api_key,
    )

    print(
        f"Pre-computing BEACON h-matrix "
        f"(n_max={args.n_max}, grid_size={args.beacon_grid_size})..."
    )
    bos = SamplingComparison(
        n_total=args.n_max,
        results_file=None,
        alpha0=args.beacon_alpha0,
        nu0=args.beacon_nu0,
        beta0=args.beacon_beta0,
        mu0=args.beacon_mu0,
        k0_bos=args.k0,
        adaptive_ignore=args.beacon_adaptive_ignore,
        adaptive_alpha=args.beacon_adaptive_alpha,
        grid_size=args.beacon_grid_size,
    )

    results = {}
    for data_name in [n.strip() for n in args.data_names.split(",") if n.strip()]:
        results[data_name] = evaluate_dataset(client, data_name, reward_eval, bos, args)

    if results:
        avg_acc = sum(v["acc"] for v in results.values()) / len(results)
        avg_samp = sum(v["avg_samples_used"] for v in results.values()) / len(results)
        print(f"\nAverage acc: {avg_acc:.1f}%  |  Average samples used: {avg_samp:.2f}")


if __name__ == "__main__":
    main()
