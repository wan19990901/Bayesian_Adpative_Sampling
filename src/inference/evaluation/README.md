# Model Evaluation Module

This module contains utilities for evaluating model outputs, including correctness checking and reward computation.

## Files

- `evaluate.py`: Core evaluation functions for model outputs
- `evaluate_Inference.py`: Inference-specific evaluation utilities
- `rm_maj_eval.py`: Reward model and majority voting evaluation

## Usage

```python
from src.inference.evaluation import evaluate_model_output, compute_reward

# Evaluate a model output
result = evaluate_model_output(
    prompt="What is 2+2?",
    response="The answer is 4.",
    ground_truth="4"
)

# Compute reward for a response
reward = compute_reward(
    prompt="What is 2+2?",
    response="The answer is 4."
)
```

## Functions

### evaluate_model_output
Evaluates a model's output against ground truth.

### compute_reward
Computes reward score for a model response.

### evaluate_with_majority_voting
Evaluates responses using majority voting from multiple models. 