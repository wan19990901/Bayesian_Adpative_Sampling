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
PROVIDER="deepinfra"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
DATA_FILE="data/math500/test.jsonl"
OUTPUT_DIR="results/Raw_response"
RESPONSES_FILE="${OUTPUT_DIR}/llama_responses_math500.json"
START_ID=${1:-0}  # Default to 0 if no start ID provided
NUM_RUNS=32  # Number of responses per question

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Check if output file exists and count valid responses
if [ -f "$RESPONSES_FILE" ]; then
    echo "Checking existing responses file..."
    ANALYSIS=$(python3 -c "
import json
with open('$RESPONSES_FILE', 'r') as f:
    data = json.load(f)
    total_problems = len(data)
    complete_problems = sum(1 for item in data 
        if 'responses' in item and 
        len([r for r in item['responses'] if r and not r.startswith('ERROR:')]) >= $NUM_RUNS)
    incomplete_problems = sum(1 for item in data 
        if 'responses' in item and 
        0 < len([r for r in item['responses'] if r and not r.startswith('ERROR:')]) < $NUM_RUNS)
    empty_problems = sum(1 for item in data 
        if 'responses' not in item or 
        len([r for r in item['responses'] if r and not r.startswith('ERROR:')]) == 0)
    print(f'{total_problems},{complete_problems},{incomplete_problems},{empty_problems}')
")
    IFS=',' read -r TOTAL_PROBLEMS COMPLETE_PROBLEMS INCOMPLETE_PROBLEMS EMPTY_PROBLEMS <<< "$ANALYSIS"
    
    echo "Analysis of existing responses file:"
    echo "  Total problems: $TOTAL_PROBLEMS"
    echo "  Complete problems (all $NUM_RUNS responses): $COMPLETE_PROBLEMS"
    echo "  Incomplete problems (some responses): $INCOMPLETE_PROBLEMS"
    echo "  Empty problems (no responses): $EMPTY_PROBLEMS"
else
    echo "No existing responses file found, starting fresh"
    TOTAL_PROBLEMS=0
    COMPLETE_PROBLEMS=0
    INCOMPLETE_PROBLEMS=0
    EMPTY_PROBLEMS=0
fi

# Start timing
START_TIME=$(date +%s)

# Generate responses
echo -e "\nStarting response generation with Llama-3.2-3B-Instruct..."
echo "Target: $NUM_RUNS responses per question"
python3 llm_inference.py \
    --provider "$PROVIDER" \
    --model_name "$MODEL_NAME" \
    --data_file "$DATA_FILE" \
    --output_file "$RESPONSES_FILE" \
    --api_key "$DEEPINFRA_API_KEY" \
    --num_runs "$NUM_RUNS" \
    --temperature 0.7 \
    --start_id "$START_ID"

# Calculate and display time taken
END_TIME=$(date +%s)
TIME_TAKEN=$((END_TIME - START_TIME))
HOURS=$((TIME_TAKEN / 3600))
MINUTES=$(( (TIME_TAKEN % 3600) / 60 ))
SECONDS=$((TIME_TAKEN % 60))

# Final status check
if [ -f "$RESPONSES_FILE" ]; then
    NEW_ANALYSIS=$(python3 -c "
import json
with open('$RESPONSES_FILE', 'r') as f:
    data = json.load(f)
    total_problems = len(data)
    complete_problems = sum(1 for item in data 
        if 'responses' in item and 
        len([r for r in item['responses'] if r and not r.startswith('ERROR:')]) >= $NUM_RUNS)
    incomplete_problems = sum(1 for item in data 
        if 'responses' in item and 
        0 < len([r for r in item['responses'] if r and not r.startswith('ERROR:')]) < $NUM_RUNS)
    empty_problems = sum(1 for item in data 
        if 'responses' not in item or 
        len([r for r in item['responses'] if r and not r.startswith('ERROR:')]) == 0)
    print(f'{total_problems},{complete_problems},{incomplete_problems},{empty_problems}')
")
    IFS=',' read -r NEW_TOTAL NEW_COMPLETE NEW_INCOMPLETE NEW_EMPTY <<< "$NEW_ANALYSIS"
    
    echo -e "\nFinal Analysis:"
    echo "  Total problems: $NEW_TOTAL"
    echo "  Complete problems (all $NUM_RUNS responses): $NEW_COMPLETE"
    echo "  Incomplete problems (some responses): $NEW_INCOMPLETE"
    echo "  Empty problems (no responses): $NEW_EMPTY"
    echo -e "\nProgress Summary:"
    echo "  New complete problems: $((NEW_COMPLETE - COMPLETE_PROBLEMS))"
    echo "  New incomplete problems: $((NEW_INCOMPLETE - INCOMPLETE_PROBLEMS))"
    echo "  New empty problems: $((NEW_EMPTY - EMPTY_PROBLEMS))"
    echo -e "\nResults saved to: $RESPONSES_FILE"
    echo "Total time taken: ${HOURS}h ${MINUTES}m ${SECONDS}s"
else
    echo "Error: No output file was created"
    exit 1
fi 