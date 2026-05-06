import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.reward_evaluator import SKYWORK_REWARD_MODEL, RewardEvaluator


for _env in [".env", os.path.expanduser("~/.env")]:
    if os.path.exists(_env):
        load_dotenv(_env)
        break


def load_outputs(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of outputs in {path}")
    return data


def normalize_responses(item: Dict) -> List[str]:
    responses = item.get("responses") or item.get("code")
    if responses is None:
        output = item.get("output", "")
        return [output] if output else []
    if responses and isinstance(responses[0], list):
        return responses[0]
    return responses


def score_outputs(args: argparse.Namespace) -> Dict[str, List[float]]:
    scorer = RewardEvaluator(
        model_name=args.reward_model,
        backend=args.reward_backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=True,
        skywork_api_key=args.skywork_api_key,
        skywork_api_model=args.skywork_api_model,
        skywork_api_url=args.skywork_api_url,
        request_timeout=args.request_timeout,
    )

    data = load_outputs(args.input_file)
    if args.limit is not None:
        data = data[: args.limit]

    scores: Dict[str, List[float]] = {}
    for idx, item in enumerate(tqdm(data, desc="Scoring Alpaca responses"), start=1):
        instruction = item.get("instruction") or item.get("question")
        if not instruction:
            raise ValueError(f"Missing instruction/question for item {idx}")

        responses = normalize_responses(item)
        if args.max_responses is not None:
            responses = responses[: args.max_responses]

        item_scores = []
        for response in responses:
            if response and not str(response).startswith("ERROR:"):
                item_scores.append(float(scorer.compute_reward(instruction, str(response))))
            else:
                item_scores.append(float("-inf"))
        scores[str(item.get("idx", idx))] = item_scores

    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score AlpacaEval or math JSONL outputs with Skywork Reward Llama 3.1 8B."
    )
    parser.add_argument("--input_file", required=True, help="Generation JSON/JSONL with responses/code lists")
    parser.add_argument("--output_file", required=True, help="Reward-score sidecar JSON to write")
    parser.add_argument(
        "--reward_backend",
        default="local",
        choices=["local", "api"],
        help="Use local HF reward model or Skywork hosted API.",
    )
    parser.add_argument("--reward_model", default=SKYWORK_REWARD_MODEL)
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu. Defaults to CUDA when available.")
    parser.add_argument("--torch_dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", default=None, help="Example: flash_attention_2 or eager")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N prompts")
    parser.add_argument("--max_responses", type=int, default=None, help="Only score the first N responses per prompt")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skywork_api_key", default=None, help="Optional override for Skywork API key.")
    parser.add_argument(
        "--skywork_api_model",
        default="Skywork-Reward-V2-Llama-3.1-8B",
        help="Skywork hosted reward model name when --reward_backend api.",
    )
    parser.add_argument(
        "--skywork_api_url",
        default="https://api.skywork.ai/v1/score",
        help="Skywork scoring endpoint when --reward_backend api.",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=60,
        help="HTTP timeout seconds for hosted API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = score_outputs(args)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Saved Skywork reward scores to {output_path}")


if __name__ == "__main__":
    main()
