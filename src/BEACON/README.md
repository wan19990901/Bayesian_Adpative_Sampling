# BEACON Module

The BEACON module implements the core functionality of our optimal stopping framework for LLM sampling.

## Core Components

### 1. Optimal Stopping (`stopping/`)

Implements the Universal Index Policy and related algorithms:
- `stopping_analysis.py`: Core stopping analysis implementation
- `uip.py`: Universal Index Policy implementation
- `posterior.py`: Posterior belief updates

### 2. Sample Comparison (`comparison/`)

Utilities for comparing and selecting samples:
- `sample_comparison.py`: Sample comparison and selection
- `reward_analysis.py`: Reward model integration

### 3. Utilities (`utils/`)

Helper functions and common utilities:
- `math_utils.py`: Mathematical utilities
- `config.py`: Configuration management
- `logging.py`: Logging utilities

## Usage

### Basic Usage

```python
from src.BEACON import analyze_stopping, compare_samples

# Analyze stopping behavior
results = analyze_stopping(
    responses_file="model_outputs.json",
    output_file="stopping_analysis.json"
)

# Compare and select samples
comparison = compare_samples(
    samples_file="model_outputs.json",
    output_file="sample_comparison.json"
)
```

### Advanced Usage

```python
from src.BEACON.stopping import UIP
from src.BEACON.comparison import RewardAnalyzer

# Initialize UIP with custom parameters
uip = UIP(
    prior_mean=0.0,
    prior_variance=1.0,
    sampling_cost=0.1
)

# Analyze rewards
analyzer = RewardAnalyzer(
    reward_model="gpt-4",
    temperature=0.7
)

# Get stopping decision
should_stop = uip.should_stop(
    current_rewards=[0.8, 0.9, 0.85],
    posterior_variance=0.1
)
```

## Implementation Details

### Stopping Analysis

The stopping analysis is based on the following principles:
1. Model rewards as Gaussian random variables
2. Update posterior beliefs after each sample
3. Use UIP to determine optimal stopping points
4. Consider sampling costs in decisions

### Sample Comparison

Sample comparison uses:
1. Reward model evaluation
2. Quality metrics computation
3. Efficient selection algorithms
4. Consistency checks

## Configuration

BEACON can be configured through:
1. Environment variables
2. Configuration files
3. Runtime parameters

See `configs/` directory for example configurations. 