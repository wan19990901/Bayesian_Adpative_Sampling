#!/bin/bash

# Load environment variables from parent directory
if [ -f "../.env" ]; then
    while IFS= read -r line; do
        if [[ $line != \#* ]] && [[ -n $line ]]; then
            export "$line"
        fi
    done < "../.env"
else
    echo "Error: .env file not found in parent directory"
    exit 1
fi

# Set variables
MODEL_NAME="openai/grok-3-mini-fast-beta"  # Using XAI provider with Grok model
OUTPUT_DIR="results/alpaca_eval"
OUTPUT_FILE="${OUTPUT_DIR}/grok3_responses_alpaca.json"
SUBSET_SIZE=${1:-50}  # Default to 50 questions
START_IDX=${2:-0}  # Default to 0 if no start index provided
NUM_RUNS=${3:-32}  # Default to 32 samples per question

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start timing
START_TIME=$(date +%s)

# Generate responses
echo -e "\nStarting response generation with $MODEL_NAME..."
echo "Target: $NUM_RUNS responses per question for $SUBSET_SIZE questions"
python3 alpaca_eval.py \
    --model "$MODEL_NAME" \
    --output_file "$OUTPUT_FILE" \
    --subset_size "$SUBSET_SIZE" \
    --temperature 0.7 \
    --num_runs "$NUM_RUNS" \
    --start_idx "$START_IDX"

# Calculate and display time taken
END_TIME=$(date +%s)
TIME_TAKEN=$((END_TIME - START_TIME))
HOURS=$((TIME_TAKEN / 3600))
MINUTES=$(( (TIME_TAKEN % 3600) / 60 ))
SECONDS=$((TIME_TAKEN % 60))

echo -e "\nResults saved to: $OUTPUT_FILE"
echo "Total time taken: ${HOURS}h ${MINUTES}m ${SECONDS}s" 