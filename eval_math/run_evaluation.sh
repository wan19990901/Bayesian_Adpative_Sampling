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
PROVIDER="openai"
MODEL_NAME="grok-3-mini-fast-beta"
DATA_FILE="data/amc23/test.jsonl"
OUTPUT_DIR="results"
RESPONSES_FILE="${OUTPUT_DIR}/xai_responses_amc23.json"
START_ID=${1:-0}  # Default to 0 if no start ID provided

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start timing
START_TIME=$(date +%s)

# Generate responses
echo "Generating responses with XAI starting from ID $START_ID..."
python3 llm_inference.py \
    --provider "$PROVIDER" \
    --model_name "$MODEL_NAME" \
    --data_file "$DATA_FILE" \
    --output_file "$RESPONSES_FILE" \
    --api_key "$XAI_API_KEY" \
    --base_url "$XAI_BASE_URL" \
    --num_runs 32 \
    --temperature 0.7 \
    --start_id "$START_ID"

# Calculate and display time taken
END_TIME=$(date +%s)
TIME_TAKEN=$((END_TIME - START_TIME))
HOURS=$((TIME_TAKEN / 3600))
MINUTES=$(( (TIME_TAKEN % 3600) / 60 ))
SECONDS=$((TIME_TAKEN % 60))

echo "Process complete. Results saved to: $RESPONSES_FILE"
echo "Total time taken: ${HOURS}h ${MINUTES}m ${SECONDS}s" 