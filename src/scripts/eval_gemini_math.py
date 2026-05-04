#!/usr/bin/env python3
"""Evaluate Gemini models on math benchmarks using the inference parser stack."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

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


for _env in [".env", os.path.expanduser("~/.env")]:
    if os.path.exists(_env):
        load_dotenv(_env)
        break


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Gemini model on math benchmarks.")
    parser.add_argument("--model", required=True, help="Gemini model name, e.g. gemini-1.5-pro or gemini-2.0-flash")
    parser.add_argument("--api_key", default=None, help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.")
    parser.add_argument("--data_names", default="math500,aime24,aime25", help="Comma-separated dataset names.")
    parser.add_argument("--data_dir", default="./data", help="Directory containing benchmark JSONL files.")
    parser.add_argument("--output_dir", default="results/gemini_math", help="Directory for JSONL outputs and metrics.")
    parser.add_argument("--prompt_type", default="cot", help="Prompt template from src.inference.data.utils.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=None, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--parallel_samples", default=1, type=int, help="Concurrent Gemini calls per problem.")
    parser.add_argument("--max_output_tokens", default=32768, type=int)
    parser.add_argument("--thinking_budget", default=8192, type=int, help="Thinking token budget (0 to disable). Only used by models that support thinking.")
    parser.add_argument("--no_thinking", action="store_true", help="Disable thinking mode even for models that support it.")
    parser.add_argument("--num_shots", default=0, type=int)
    parser.add_argument("--adapt_few_shot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--sleep", default=0.0, type=float, help="Seconds to sleep between API calls.")
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--save_every", default=1, type=int, help="Save JSONL progress after this many new problems.")
    return parser.parse_args()


def build_client(api_key):
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --api_key.")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        pass
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        return genai_legacy
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install google-genai  # or google-generativeai") from exc


def _supports_thinking(model_name: str) -> bool:
    """Return True for Gemini 2.5+ models that support thinking mode."""
    return "gemini-2.5" in model_name or "gemini-exp" in model_name


def generate_one(client, model_name, prompt, args):
    use_thinking = _supports_thinking(model_name) and not args.no_thinking

    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            # New google-genai SDK path
            if hasattr(client, "models"):
                from google.genai import types as gtypes
                config_kwargs = {
                    "temperature": args.temperature,
                    "max_output_tokens": args.max_output_tokens,
                }
                if args.top_p is not None:
                    config_kwargs["top_p"] = args.top_p
                if use_thinking and args.thinking_budget > 0:
                    config_kwargs["thinking_config"] = gtypes.ThinkingConfig(
                        thinking_budget=args.thinking_budget
                    )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(**config_kwargs),
                )
                return response.text or ""
            # Legacy google-generativeai SDK fallback
            else:
                model = client.GenerativeModel(model_name)
                config_args = {
                    "temperature": args.temperature,
                    "max_output_tokens": args.max_output_tokens,
                }
                if args.top_p is not None:
                    config_args["top_p"] = args.top_p
                generation_config = client.types.GenerationConfig(**config_args)
                response = model.generate_content(prompt, generation_config=generation_config)
                return getattr(response, "text", "") or ""
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(min(2**attempt, 30))
    return f"ERROR: {last_error}"


def generate_samples(genai, model_name, prompt, args):
    if args.n_sampling <= 1 or args.parallel_samples <= 1:
        responses = []
        for _ in range(args.n_sampling):
            responses.append(generate_one(genai, model_name, prompt, args).strip())
            if args.sleep:
                time.sleep(args.sleep)
        return responses

    responses = [""] * args.n_sampling
    max_workers = min(args.parallel_samples, args.n_sampling)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_one, genai, model_name, prompt, args): idx
            for idx in range(args.n_sampling)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                responses[idx] = future.result().strip()
            except Exception as exc:
                responses[idx] = f"ERROR: {exc}"
            if args.sleep:
                time.sleep(args.sleep)
    return responses


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_n{args.n_sampling}"
    out_file = os.path.join(args.output_dir, data_name, f"{prefix}_s{args.start}_e{args.end}.jsonl")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    processed_samples = []
    if not args.overwrite and os.path.exists(out_file):
        processed_samples = list(load_jsonl(out_file))
        processed_idxs = {sample["idx"] for sample in processed_samples}
        examples = [example for example in examples if example["idx"] not in processed_idxs]

    return examples, processed_samples, out_file


def is_multi_choice(answer):
    return all(c in ["A", "B", "C", "D", "E"] for c in str(answer))


def evaluate_dataset(genai, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, ", remaining samples:", len(examples))

    prompt_args = SimpleNamespace(
        prompt_type=args.prompt_type,
        num_shots=args.num_shots,
        adapt_few_shot=args.adapt_few_shot,
    )
    if args.prompt_type in ["pal", "pot", "tool-integrated", "jiuzhang_tora"]:
        raise SystemExit("Gemini API math evaluator currently supports non-tool prompts such as cot, direct, or qwen25-math-cot.")
    samples = []
    start_time = time.time()

    for example in tqdm(examples, total=len(examples), desc=data_name):
        example["question"] = parse_question(example, data_name)
        if not example["question"]:
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        prompt = construct_prompt(example, data_name, prompt_args)

        responses = generate_samples(genai, args.model, prompt, args)
        preds = []
        reports = []
        for response in responses:
            pred, report = run_execute(None, response, args.prompt_type, data_name)
            preds.append(pred)
            reports.append(report)

        for i, pred in enumerate(preds):
            if gt_ans in ["A", "B", "C", "D", "E"] and pred not in ["A", "B", "C", "D", "E"]:
                preds[i] = choice_answer_clean(responses[i])
            elif is_multi_choice(gt_ans) and not is_multi_choice(pred):
                preds[i] = "".join([c for c in str(pred) if c in ["A", "B", "C", "D", "E"]])

        sample = {
            "idx": example["idx"],
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "code": responses,
            "pred": preds,
            "report": reports,
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
        sample["score"] = []
        for pred in sample["pred"]:
            try:
                sample["score"].append(bool(math_equal(pred, sample["gt"], timeout=True)))
            except Exception:
                timeout_count += 1
                sample["score"].append(False)

    scores = [sample["score"] for sample in all_samples]
    metrics = {
        "num_samples": len(all_samples),
        "num_scores": sum(len(row) for row in scores),
        "timeout_samples": timeout_count,
        "empty_samples": len([sample for sample in all_samples if not sample["pred"][-1]]),
        "acc": round(100 * sum(row[0] for row in scores) / len(scores), 1) if scores else 0.0,
        "pass_at_n": round(100 * sum(any(row) for row in scores) / len(scores), 1) if scores else 0.0,
        "acc_by_sample": [
            round(100 * sum(row[i] for row in scores if len(row) > i) / len(scores), 1)
            for i in range(args.n_sampling)
        ] if scores else [],
        "model": args.model,
        "n_sampling": args.n_sampling,
        "parallel_samples": args.parallel_samples,
        "time_use_in_second": time.time() - start_time,
    }

    if args.save_outputs or samples:
        save_jsonl(all_samples, out_file)
    with open(out_file.replace(".jsonl", "_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    genai = build_client(args.api_key)
    results = {}
    for data_name in [name.strip() for name in args.data_names.split(",") if name.strip()]:
        results[data_name] = evaluate_dataset(genai, data_name, args)
    if results:
        avg = sum(item["acc"] for item in results.values()) / len(results)
        print("average first-sample acc:", round(avg, 1))


if __name__ == "__main__":
    main()
