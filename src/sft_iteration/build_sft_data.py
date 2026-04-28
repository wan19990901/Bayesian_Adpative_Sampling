import json
import random
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    reward_json_path: Optional[str] = field(
        default="iter_sft/reward.json",
        metadata={"help": "Path to reward-labeled data JSON produced by reward_labeling.py"},
    )
    output_jsonl_path: Optional[str] = field(
        default="iter_sft/sft_data.jsonl",
        metadata={"help": "Path to output SFT JSONL"},
    )
    min_reward: Optional[float] = field(
        default=1.0,
        metadata={"help": "Minimum reward threshold for acceptance"},
    )
    keep_strategy: Optional[str] = field(
        default="best",
        metadata={"help": "best or threshold"},
    )
    max_response_tokens: Optional[int] = field(
        default=1500,
        metadata={"help": "Discard correct responses longer than this many whitespace-delimited tokens. "
                          "Prevents verbosity spiral across iterations. Set -1 to disable."},
    )
    require_non_empty: Optional[bool] = field(
        default=True,
        metadata={"help": "Drop empty selected responses"},
    )
    eot_token: Optional[str] = field(
        default="<|im_end|>",
        metadata={"help": "End-of-turn token appended to each response in the text field"},
    )


def _select_responses(sample, keep_strategy: str, min_reward: float):
    responses = sample.get("responses", [])
    rewards = sample.get("rewards", [])
    if len(responses) == 0 or len(rewards) == 0:
        return []
    if len(responses) != len(rewards):
        n = min(len(responses), len(rewards))
        responses = responses[:n]
        rewards = rewards[:n]

    if keep_strategy == "threshold":
        return [responses[i] for i, r in enumerate(rewards) if r >= min_reward]

    # best: one response per prompt, randomly sampled from all correct responses
    correct_indices = [i for i, r in enumerate(rewards) if r >= min_reward]
    if not correct_indices:
        return []
    return [responses[random.choice(correct_indices)]]


def _length_ok(response: str, max_tokens: int) -> bool:
    if max_tokens < 0:
        return True
    return len(response.split()) <= max_tokens


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    random.seed(42)
    ds = load_dataset("json", data_files=args.reward_json_path, split="train")
    accepted = 0
    skipped_empty = 0
    skipped_length = 0
    total = 0

    with open(args.output_jsonl_path, "w", encoding="utf-8") as fout:
        for sample in ds:
            total += 1
            prompt = sample.get("prompt", "")
            chosen_responses = _select_responses(sample, args.keep_strategy, args.min_reward)
            for resp in chosen_responses:
                text = (resp or "").strip()
                if args.require_non_empty and not text:
                    skipped_empty += 1
                    continue
                if not _length_ok(text, args.max_response_tokens):
                    skipped_length += 1
                    continue
                row = {
                    "prompt": prompt,
                    "response": text,
                    "text": f"{prompt}{text}{args.eot_token}",
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1

    print(
        f"Built SFT dataset at {args.output_jsonl_path}. "
        f"Accepted {accepted} / {total} prompts "
        f"(skipped {skipped_length} over length limit, {skipped_empty} empty)."
    )


if __name__ == "__main__":
    main()
