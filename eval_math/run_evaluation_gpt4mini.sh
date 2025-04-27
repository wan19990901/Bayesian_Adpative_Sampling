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
MODEL_NAME="gpt-4-mini"  # Adjust this to the correct model name
DATA_FILE="data/aime24/test.jsonl"
OUTPUT_DIR="results"
RESPONSES_FILE="${OUTPUT_DIR}/gpt4mini_responses.json"
EVAL_FILE="${OUTPUT_DIR}/gpt4mini_evaluation.json"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Step 1: Generate responses
echo "Generating responses with GPT-4-mini..."
python3 llm_inference.py \
    --provider "$PROVIDER" \
    --model_name "$MODEL_NAME" \
    --data_file "$DATA_FILE" \
    --output_file "$RESPONSES_FILE" \
    --api_key "$OPENAI_API_KEY" \
    --base_url "$OPENAI_BASE_URL" \
    --num_runs 4 \
    --num_problems 2 \
    --temperature 0.7

# Step 2: Evaluate responses
echo "Evaluating responses..."
python3 llm_evaluator.py \
    --responses_file "$RESPONSES_FILE" \
    --output_file "$EVAL_FILE"

echo "Process complete. Results saved to:"
echo "Responses: $RESPONSES_FILE"
echo "Evaluation: $EVAL_FILE" 