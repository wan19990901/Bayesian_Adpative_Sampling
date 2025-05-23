# BEACON: Bayesian Optimal Stopping for Efficient LLM Sampling

This repository implements BEACON (Bayesian Efficient Adaptive Criterion for Optimal N-stopping), a framework for efficient LLM sampling based on Sequential Search with Gaussian Learning (SSGL).

## Key Features

- **Optimal Stopping**: Implements UIP for optimal stopping decisions
- **Adaptive Sampling**: Dynamically adjusts sample counts based on reward model feedback
- **Efficiency**: Reduces average sample counts by up to 80% compared to fixed BoN
- **Quality Preservation**: Maintains comparable response quality while reducing computation

## Project Structure

```
src/
├── BEACON/               # Best Early-stopping And COmparison of N-samples
│   ├── stopping_analysis.py    # Dynamic stopping analysis
│   └── sampling_comparison.py  # Sample comparison utilities
├── inference/            # Model inference and evaluation
│   ├── data/            # Data loading and processing
│   ├── math_utils/      # Mathematical problem utilities
│   └── evaluation/      # Model evaluation utilities
├── results/             # Analysis and result processing
│   ├── reward_analysis.py      # Reward analysis
│   └── llm_evaluator.py        # LLM evaluation
├── evaluation/          # Evaluation utilities
├── dpo_iteration/       # DPO training implementation
├── scripts/            # Utility scripts
└── utils/              # Utility functions
    └── llm/            # Language Model utilities
        ├── llm_inference.py    # LLM inference
        └── model_utils.py      # Model utilities
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Model Inference

```python
from src.utils.llm import LLMGenerator

# Initialize LLM generator
generator = LLMGenerator(
    provider="openai",
    api_key="your-api-key",
    model_name="gpt-4"
)

# Generate responses
results = generator.generate_responses(
    data_file="path/to/data.jsonl",
    output_file="path/to/output.json"
)
```

### Dynamic Stopping

```python
from src.BEACON import analyze_stopping, compare_samples

# Analyze stopping behavior
stopping_results = analyze_stopping(
    responses_file="path/to/responses.json",
    output_file="path/to/analysis.json"
)

# Compare samples
comparison_results = compare_samples(
    samples_file="path/to/samples.json",
    output_file="path/to/comparison.json"
)
```

### Result Analysis

```python
from src.results import analyze_rewards, process_leaderboard

# Analyze rewards
reward_analysis = analyze_rewards(
    results_file="path/to/results.json",
    output_file="path/to/analysis.json"
)

# Process leaderboard
leaderboard = process_leaderboard(
    results_file="path/to/results.json",
    output_file="path/to/leaderboard.json"
)
```

## Features

- **Dynamic Stopping**: Implemented in the BEACON module for optimal response selection
- **Multiple LLM Support**: Integration with OpenAI, Claude, Gemini, and DeepInfra
- **Comprehensive Evaluation**: Tools for evaluating model performance on mathematical problems
- **Result Analysis**: Utilities for analyzing and processing model outputs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

We have two separate environments for running the Iterative DPO.

### Generation
```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets

# The following code is tested for CUDA12.0-12.2, and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.48.1
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!

pip install antlr4-python3-runtime==4.7.2
pip install sympy==1.12
pip install latex2sympy2==1.9.1
pip install word2number==1.1
```


### Training


```sh
conda create -n rlhflow python=3.10.9
conda activate rlhflow

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0
pip install huggingface-hub==0.24.7

pip install transformers==4.42.2
pip install peft==0.7.1  #We do not use peft, but some versions would cause errors.
pip install deepspeed==0.15.4
pip install trl==0.9.6
pip install wandb
```

## Running Online-DPO with numina prompt set:

```
bash run_iter_dpo.sh
```

## Using the Reward Evaluation System

To use the reward evaluation system:

```python
from utils.reward_evaluator import RewardEvaluator

# Initialize the evaluator with a specific model
evaluator = RewardEvaluator(model_name="OpenAssistant/reward-model-deberta-v3-large-v2")

# Evaluate a single response
reward = evaluator.compute_reward(prompt="What is 2+2?", response="The answer is 4.")

# Evaluate multiple responses
rewards = evaluator.evaluate_responses(
    prompt="What is 2+2?",
    responses=["The answer is 4.", "It's 4.", "I think it's 4."]
)
```

## Using Bayesian Optimal Stopping

To use the Bayesian Optimal Stopping mechanism:

```python
from utils.reward_evaluator import BayesianOptimalStopping

# Initialize the BOS model
bos = BayesianOptimalStopping(
    cost_per_sample=0.1,
    max_iterations=100
)

# Run sampling with initial samples
samples_used, max_reward, correctness = bos.run_sampling(
    initial_sample_pairs=[(0.8, 1), (0.6, 1), (0.4, 0)],
    true_underlying_mean=0.7,
    true_underlying_std=0.2
)
```

## Evaluation

The evaluation system supports various mathematical benchmarks and tasks. The evaluation utilities are located in the `src/evaluation` directory.

### Supported Datasets
- GSM8K: Grade School Math 8K dataset
- MATH: Mathematical problem-solving dataset
- Custom mathematical problems

### Evaluation Metrics
- Accuracy
- Reward scores
- Execution success rate
- Response quality metrics

To use the evaluation system:

```python
from src.evaluation import evaluate

# Evaluate model outputs
results = evaluate(
    samples=model_outputs,
    data_name="gsm8k",
    prompt_type="tool-integrated",
    execute=True
)
```

For detailed evaluation metrics and analysis, refer to the documentation in `src/evaluation/README.md`.

