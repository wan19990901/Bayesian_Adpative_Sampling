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
MODEL_NAME=${1:-"openai/gpt-4"}  # Default to GPT-4 if no model specified
OUTPUT_DIR="results/alpaca_eval"
RESPONSES_FILE="${OUTPUT_DIR}/${MODEL_NAME//\//_}_responses.json"
ANNOTATORS_CONFIG=${2:-"weighted_alpaca_eval_gpt4_turbo"}  # Default annotator config

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start timing
START_TIME=$(date +%s)

# Run AlpacaEval analysis
echo -e "\nRunning AlpacaEval analysis for $MODEL_NAME..."
alpaca_eval \
    --model_outputs "$RESPONSES_FILE" \
    --annotators_config "$ANNOTATORS_CONFIG" \
    --output_dir "$OUTPUT_DIR/analysis_${MODEL_NAME//\//_}"

# Calculate and display time taken
END_TIME=$(date +%s)
TIME_TAKEN=$((END_TIME - START_TIME))
HOURS=$((TIME_TAKEN / 3600))
MINUTES=$(( (TIME_TAKEN % 3600) / 60 ))
SECONDS=$((TIME_TAKEN % 60))

echo -e "\nAnalysis complete. Results saved to: $OUTPUT_DIR/analysis_${MODEL_NAME//\//_}"
echo "Total time taken: ${HOURS}h ${MINUTES}m ${SECONDS}s" 