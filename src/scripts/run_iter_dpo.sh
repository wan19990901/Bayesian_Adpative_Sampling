#!/bin/bash

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run the iterative DPO training
python src/dpo_iteration/train.py \
    --model_name "Qwen/Qwen1.5-7B" \
    --dataset "math-7500" \
    --output_dir "outputs/iterative_dpo" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --max_length 2048 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --num_samples 4 \
    --use_wandb true

