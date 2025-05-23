# BEACON Examples

This directory contains example scripts demonstrating how to use BEACON (Bayesian Efficient Adaptive Criterion for Optimal N-stopping) for efficient LLM sampling.

## Available Examples

### 1. BEACON Example (`beacon_example.py`)

Demonstrates BEACON's core functionality for optimal stopping and sample comparison using Sequential Search with Gaussian Learning (SSGL).

```bash
python beacon_example.py
```

This example:
- Creates math problems of varying difficulty
- Demonstrates UIP-based optimal stopping
- Shows adaptive sampling based on problem difficulty
- Analyzes efficiency gains compared to fixed sampling
- Illustrates sample comparison and selection

Key features demonstrated:
- Universal Index Policy (UIP) implementation
- Posterior belief updates
- Cost-aware stopping decisions
- Efficiency analysis

### 2. LLM Inference Example (`llm_inference_example.py`)

Shows how to use BEACON with different LLM providers for generating responses.

```bash
python llm_inference_example.py
```

This example:
- Initializes LLM generators with different providers
- Generates responses for math problems
- Integrates with BEACON for optimal stopping
- Demonstrates reward model integration

### 3. Results Analysis Example (`results_analysis_example.py`)

Demonstrates how to analyze and evaluate BEACON's performance.

```bash
python results_analysis_example.py
```

This example:
- Analyzes reward distributions
- Processes leaderboard results
- Evaluates LLM outputs
- Integrates with AlpacaEval

## Theoretical Background

The examples demonstrate BEACON's key theoretical components:

1. **Sequential Search with Gaussian Learning (SSGL)**
   - Modeling reward distributions
   - Posterior belief updates
   - Optimal stopping decisions

2. **Universal Index Policy (UIP)**
   - Cost-aware stopping criteria
   - Dynamic sample size adjustment
   - Quality preservation guarantees

3. **Adaptive Sampling**
   - Problem difficulty adaptation
   - Reward model integration
   - Efficiency optimization

## Requirements

Before running the examples, ensure you have:

1. Installed all required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Configured any necessary API keys or credentials.

## Notes

- The examples use temporary files that are automatically cleaned up
- Some examples require specific API keys or configurations
- The examples are designed to demonstrate BEACON's core functionality
- Modify the examples as needed for your specific use case

## Performance Metrics

The examples demonstrate BEACON's key performance metrics:

1. **Efficiency**
   - Sample count reduction
   - Computational savings
   - Resource utilization

2. **Quality**
   - Response quality preservation
   - Reward distribution analysis
   - Consistency metrics

3. **Adaptability**
   - Problem difficulty handling
   - Dynamic parameter adjustment
   - Robust performance 