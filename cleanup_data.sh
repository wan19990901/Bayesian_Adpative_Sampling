#!/bin/bash

# Base directory
BASE_DIR="eval_math/data"

# Directories to keep
KEEP_DIRS=("mmlu_stem" "math500" "aime24" "amc23")

# Remove all directories except the ones we want to keep
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        if [[ ! " ${KEEP_DIRS[@]} " =~ " ${dir_name} " ]]; then
            echo "Removing directory: $dir"
            rm -rf "$dir"
        fi
    fi
done

echo "Cleanup complete. Only the following directories remain:"
echo "${KEEP_DIRS[@]}" 