# Evaluation Module

This module provides utilities for evaluating model performance on mathematical problems and other tasks.

## Core Components

### 1. Mathematical Evaluation (`math_eval.py`)
- Evaluates model performance on mathematical problems
- Supports multiple datasets (GSM8K, MATH, etc.)
- Handles different prompt types and evaluation metrics

### 2. Model Evaluation (`model_eval.py`)
- General model evaluation utilities
- Supports multiple model providers
- Handles different evaluation metrics

### 3. Reward Analysis (`reward_analysis.py`)
- Analyzes reward scores
- Computes performance metrics
- Generates evaluation reports

## Usage

### Basic Usage

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

### Advanced Usage

```python
from src.evaluation.math_eval import MathEvaluator

# Initialize evaluator
evaluator = MathEvaluator(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2048
)

# Evaluate specific problems
results = evaluator.evaluate_problems(
    problems=math_problems,
    num_samples=5
)
```

## Supported Datasets

- GSM8K
- MATH
- Custom mathematical problems

## Evaluation Metrics

- Accuracy
- Reward scores
- Execution success rate
- Response quality metrics

## Configuration

The evaluation module can be configured through:
1. Command line arguments
2. Configuration files
3. Environment variables

See `configs/evaluation/` for example configurations. 