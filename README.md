This is the repository for running the Iterative DPO with rule-based rewards. In every iteration, we sample responses from the model and label the rewards using the rule-based method. We then construct the preference pair based on the reward scores for DPO training. In our code, we perform iterative DPO starting with Qwen2.5-MATH-7B with prompts from MATH-7500. 


<div align="center">
  <img src="figures/dpo_overview.png" alt="Figure caption" width="100%">
  <p><em>Illustration of the iterative DPO pipeline. Here the exploration is implemented via best-of-n v.s. worst of n sampling. In other words, we sample n responses and use the response with the highest reward and lowest reward as a preference pair.</em></p>
</div>

## New Features

### Reward Evaluation
The repository now includes a flexible reward evaluation system that can:
- Evaluate responses using pre-trained reward models
- Support both regression and classification-based reward models
- Process multiple responses efficiently
- Work with custom reward models from Hugging Face

### Bayesian Optimal Stopping
We've implemented a Bayesian Optimal Stopping (BOS) mechanism that:
- Dynamically determines when to stop sampling based on reward distributions
- Adapts to the underlying reward distribution
- Balances exploration and exploitation
- Minimizes sampling costs while maximizing reward quality

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

We provide the evaluation scripts for all the benchmarks we use, including **AIME24**, **AMC23**, **MATH500**. Please go to ```eval_math``` folder for the detailed instructions.

