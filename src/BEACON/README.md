# BEACON: Bayesian Efficient Adaptive Criterion for Optimal N-stopping

BEACON is a framework for efficient LLM sampling based on Sequential Search with Gaussian Learning (SSGL). It implements an optimal stopping strategy that dynamically determines when to stop generating samples based on reward model feedback.

## Theoretical Foundation

BEACON is built on the following key principles:

1. **Bayesian Learning**: Models rewards as Gaussian random variables and updates posterior beliefs after each sample
2. **Universal Index Policy (UIP)**: Implements an optimal stopping strategy that balances exploration and exploitation
3. **Adaptive Sampling**: Dynamically adjusts sample counts based on reward model feedback
4. **Cost-Aware Decision Making**: Considers sampling costs in stopping decisions

## Core Components

### 1. Stopping Analysis (`stopping_analysis.py`)
- Implements the Universal Index Policy (UIP)
- Handles posterior belief updates
- Manages stopping decisions
- Tracks sampling efficiency

### 2. Sample Comparison (`sampling_comparison.py`)
- Compares and selects optimal samples
- Integrates with reward models
- Computes quality metrics
- Ensures consistency in selection

## Usage

### Basic Usage

```python
from src.BEACON import analyze_stopping, compare_samples

# Analyze stopping behavior with BEACON
results = analyze_stopping(
    responses_file="model_outputs.json",
    output_file="stopping_analysis.json",
    sampling_cost=0.1,  # Cost per sample
    prior_mean=0.0,     # Initial belief about mean reward
    prior_variance=1.0  # Initial uncertainty
)

# Compare and select samples using BEACON's criteria
comparison = compare_samples(
    samples_file="model_outputs.json",
    output_file="sample_comparison.json",
    reward_model="gpt-4",
    temperature=0.7
)
```

### Advanced Usage

```python
from src.BEACON.stopping_analysis import UIP
from src.BEACON.sampling_comparison import RewardAnalyzer

# Initialize UIP with custom parameters
uip = UIP(
    prior_mean=0.0,
    prior_variance=1.0,
    sampling_cost=0.1,
    max_samples=10
)

# Analyze rewards with BEACON's reward model
analyzer = RewardAnalyzer(
    reward_model="gpt-4",
    temperature=0.7,
    quality_threshold=0.8
)

# Get optimal stopping decision
should_stop = uip.should_stop(
    current_rewards=[0.8, 0.9, 0.85],
    posterior_variance=0.1,
    remaining_budget=5
)
```

## Key Features

1. **Optimal Stopping**
   - Dynamic sample count determination
   - Cost-aware decision making
   - Posterior belief updates
   - Confidence-based stopping

2. **Sample Comparison**
   - Reward model integration
   - Quality metrics computation
   - Efficient selection algorithms
   - Consistency checks

3. **Efficiency Metrics**
   - Sample count reduction
   - Quality preservation
   - Cost savings
   - Performance tracking

## Configuration

BEACON can be configured through:

1. **Environment Variables**
   ```bash
   BEACON_SAMPLING_COST=0.1
   BEACON_MAX_SAMPLES=10
   BEACON_QUALITY_THRESHOLD=0.8
   ```

2. **Configuration Files**
   ```json
   {
     "sampling_cost": 0.1,
     "max_samples": 10,
     "quality_threshold": 0.8,
     "reward_model": "gpt-4"
   }
   ```

3. **Runtime Parameters**
   - Prior beliefs
   - Sampling costs
   - Quality thresholds
   - Model configurations

## Performance

BEACON typically achieves:
- 40-80% reduction in average sample counts
- Comparable or better response quality
- Significant cost savings
- Improved sampling efficiency

See `configs/` directory for example configurations and performance benchmarks. 