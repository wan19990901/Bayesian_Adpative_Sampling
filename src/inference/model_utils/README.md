# Model Utilities Module

This module contains utilities for loading models and performing inference, including text generation and response sampling.

## Files

- `model_utils.py`: Core model utility functions
- `llm_inference.py`: Language model inference utilities

## Usage

```python
from src.inference.model_utils import load_model, generate_response

# Load a model
model = load_model(
    model_name="Qwen/Qwen1.5-7B",
    device="cuda"
)

# Generate a response
response = generate_response(
    model=model,
    prompt="What is 2+2?",
    max_length=100
)
```

## Functions

### load_model
Loads a language model with specified configuration.

### generate_response
Generates a response from a model given a prompt.

### sample_responses
Samples multiple responses from a model.

### compute_logits
Computes model logits for a given input. 