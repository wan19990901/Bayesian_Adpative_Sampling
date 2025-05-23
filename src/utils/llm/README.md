# LLM Utilities Module

This module provides utilities for working with Language Models, including model loading, text generation, and API integration.

## Files

- `llm_inference.py`: Core LLM inference implementation
- `model_utils.py`: Model utility functions

## Usage

```python
from src.utils.llm import (
    LLMGenerator,
    OpenAIProvider,
    ClaudeProvider,
    GeminiProvider,
    DeepInfraProvider
)

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

## Features

### Model Providers
- OpenAI API integration
- Claude API integration
- Gemini API integration
- DeepInfra API integration

### Text Generation
- Configurable generation parameters
- Multiple response sampling
- Token usage tracking
- Error handling and retries

### Model Utilities
- Model loading and initialization
- Tokenizer management
- Generation utilities
- Performance optimization

## Implementation Details

### LLMGenerator
The main class for generating responses:
- Supports multiple providers
- Handles API authentication
- Manages generation parameters
- Tracks token usage

### Provider Classes
Each provider implements:
- API-specific authentication
- Request formatting
- Response parsing
- Error handling

### Model Utilities
Utility functions for:
- Model loading
- Tokenization
- Generation
- Performance optimization 