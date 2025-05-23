# Data Utilities Module

This module contains utilities for loading and processing data, including trajectory tracking and example management.

## Files

- `data_loader.py`: Data loading utilities
- `examples.py`: Example management and processing
- `trajectory.py`: Trajectory tracking utilities
- `utils.py`: General data utility functions
- `process.py`: Data processing utilities
- `parser.py`: Data parsing utilities

## Directories

- `aime24/`: AIME 2024 dataset
- `math500/`: MATH-500 dataset
- `mmlu_stem/`: MMLU-STEM dataset
- `amc23/`: AMC 2023 dataset

## Usage

```python
from src.inference.data import load_dataset, process_examples

# Load a dataset
dataset = load_dataset(
    dataset_name="math-7500",
    split="train"
)

# Process examples
processed_data = process_examples(
    examples=dataset,
    max_length=2048
)
```

## Functions

### load_dataset
Loads a dataset with specified configuration.

### process_examples
Processes examples for model training/inference.

### track_trajectory
Tracks model training/inference trajectories.

### parse_data
Parses raw data into structured format. 