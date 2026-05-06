# Evaluation Module

This folder is for instruction-following evaluation, primarily AlpacaEval.

## AlpacaEval

Use `alpaca_eval.py` to generate model outputs for AlpacaEval/AlpacaEval 2.0:

```bash
python -m src.evaluation.alpaca_eval \
  --model gemini/gemini-1.5-pro \
  --output_file results/alpaca_eval/gemini_responses_alpaca.json \
  --temperature 0.7 \
  --num_runs 1
```

Then run the AlpacaEval judge:

```bash
alpaca_eval \
  --model_outputs results/alpaca_eval/gemini_responses_alpaca.json \
  --annotators_config weighted_alpaca_eval_gpt4_turbo
```

Supported generation providers in `alpaca_eval.py`:

- `gemini/<model-name>` using `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `deepinfra/<model-name>` using `DEEPINFRA_API_KEY`
- `openai/<model-name>` routed to xAI using `XAI_API_KEY`

## Reward Model Reranking

To score multi-response AlpacaEval generations with Skywork Reward Llama 3.1 8B:

```bash
python -m src.scripts.score_alpaca_with_skywork \
  --input_file results/alpaca_eval/grok3_responses_alpaca.json \
  --output_file results/alpaca_eval/skywork_reward_grok3_alpaca.json \
  --reward_model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
  --device cuda:0 \
  --torch_dtype bfloat16 \
  --attn_implementation eager
```

Then select one response per prompt using the score sidecar:

```bash
python src/inference/fix_alpaca_format.py
```

`fix_alpaca_format.py` currently uses `use_dynamic=False`, so it picks the highest
Skywork score for each prompt. Set `use_dynamic=True` in that call to use the
BOS-style adaptive stopping selector.

## Math Evaluation

Math benchmark evaluation lives under `src/inference`, not this folder.

For Gemini on MATH-500/AIME, use:

```bash
python -m src.scripts.eval_gemini_math \
  --model gemini-1.5-pro \
  --data_names math500,aime24,aime25 \
  --data_dir data \
  --output_dir results/gemini_math \
  --prompt_type mathstral \
  --n_sampling 1 \
  --save_outputs
```

The math command reuses the inference parser/evaluator stack and reports
first-sample accuracy plus `pass_at_n` when `--n_sampling` is greater than 1.
