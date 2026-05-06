from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-Math-7B",
        metadata={"help": "Base or previous-iteration model path (HF hub ID or local path)"},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Pre-built SFT JSONL (prompt/response/text). Used when train_dir is not set."},
    )
    train_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Raw reward-labeled JSON (same format as DPO pipeline). When set, build_sft_data step is skipped."},
    )
    output_dir: Optional[str] = field(
        default="Qwen_numina_sft_iter1",
        metadata={"help": "Output checkpoint directory"},
    )
    max_length: Optional[int] = field(
        default=4096,
        metadata={"help": "Max sequence length"},
    )
    learning_rate: Optional[float] = field(
        default=2.0e-6,
        metadata={"help": "Learning rate"},
    )
    num_train_epochs: Optional[float] = field(
        default=1.0,
        metadata={"help": "Number of training epochs"},
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "Max training steps (-1 = use num_train_epochs)"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Train batch size per device"},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16,
        metadata={"help": "Gradient accumulation steps"},
    )
    logging_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Logging frequency"},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Use bf16"},
    )
    min_reward: Optional[float] = field(
        default=1.0,
        metadata={"help": "When reading raw DPO format, minimum reward to accept a response"},
    )
    eot_token: Optional[str] = field(
        default="<|im_end|>",
        metadata={"help": "End-of-turn token appended to each training example"},
    )


def prepare_data_from_dpo_format(data_dir: str, min_reward: float = 1.0, eot_token: str = "<|im_end|>") -> Dataset:
    """Read raw reward-labeled JSON (same format as DPO pipeline) and select best correct responses."""
    ds = load_dataset("json", data_files=data_dir, split="train")
    print(ds)

    texts = []
    skipped = 0
    for sample in ds:
        responses = sample.get("responses", [])
        rewards = sample.get("rewards", [])
        if not responses or not rewards:
            skipped += 1
            continue
        n = min(len(responses), len(rewards))
        best_idx = int(np.argmax(rewards[:n]))
        if rewards[best_idx] < min_reward:
            skipped += 1
            continue
        prompt = sample.get("prompt", "")
        text = f"{prompt}{responses[best_idx]}{eot_token}"
        texts.append({"prompt": prompt, "response": responses[best_idx], "text": text})

    print(f"[prepare_data_from_dpo_format] accepted {len(texts)}, skipped {skipped}")
    return Dataset.from_list(texts)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    assert args.train_file or args.train_dir, "Provide either --train_file (pre-built JSONL) or --train_dir (raw DPO reward JSON)"

    if args.train_dir:
        ds = prepare_data_from_dpo_format(args.train_dir, min_reward=args.min_reward, eot_token=args.eot_token)
    else:
        ds = load_dataset("json", data_files=args.train_file, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    train_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_only_model=True,
        report_to="wandb",
        bf16=args.bf16,
        max_seq_length=args.max_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
