# BEACON: Bayesian Efficient Adaptive Criterion for Optimal N-stopping

BEACON is a framework for efficient LLM sampling based on Sequential Search with Gaussian Learning (SSGL). It implements an optimal stopping strategy that dynamically determines when to stop generating samples based on reward model feedback.

## Key Features

- **Optimal Stopping**: Implements UIP for optimal stopping decisions
- **Adaptive Sampling**: Dynamically adjusts sample counts based on reward model feedback
- **Efficiency**: Reduces average sample counts by up to 80% compared to fixed BoN
- **Quality Preservation**: Maintains comparable response quality while reducing computation

## Project Structure

```
src/
├── BEACON/               # Core BEACON implementation
│   ├── stopping_analysis.py    # Dynamic stopping analysis
│   └── sampling_comparison.py  # Sample comparison utilities
├── inference/            # Model inference and evaluation
│   ├── data/            # Data loading and processing
│   ├── math_utils/      # Mathematical problem utilities
│   └── evaluation/      # Model evaluation utilities
├── results/             # Analysis and result processing
│   ├── reward_analysis.py      # Reward analysis
│   └── llm_evaluator.py        # LLM evaluation
└── utils/               # Utility functions
    └── llm/            # Language Model utilities
        ├── llm_inference.py    # LLM inference
        └── model_utils.py      # Model utilities
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic BEACON Usage

```python
from src.BEACON import analyze_stopping, compare_samples

# Initialize BEACON with optimal stopping
results = analyze_stopping(
    responses_file="model_outputs.json",
    output_file="stopping_analysis.json",
    sampling_cost=0.1,
    prior_mean=0.0,
    prior_variance=1.0
)

# Compare samples using BEACON's criteria
comparison = compare_samples(
    samples_file="model_outputs.json",
    output_file="sample_comparison.json",
    reward_model="gpt-4",
    temperature=0.7
)
```

### Advanced BEACON Usage

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

## BEACON Features

### 1. Optimal Stopping
- Dynamic sample count determination
- Cost-aware decision making
- Posterior belief updates
- Confidence-based stopping

### 2. Sample Comparison
- Reward model integration
- Quality metrics computation
- Efficient selection algorithms
- Consistency checks

### 3. Efficiency Metrics
- Sample count reduction (40-80%)
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Features

- **Dynamic Stopping**: Implemented in the BEACON module for optimal response selection
- **Multiple LLM Support**: Integration with OpenAI, Claude, Gemini, and DeepInfra
- **Comprehensive Evaluation**: Tools for evaluating model performance on mathematical problems
- **Result Analysis**: Utilities for analyzing and processing model outputs

## Running Online-DPO with numina prompt set:

```
bash run_iter_dpo.sh
```