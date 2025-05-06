import json
import os
from tqdm import tqdm

def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def fix_ground_truth(responses_file, original_data_file, output_file):
    """
    Match ground truth from original data with generated responses using question IDs.
    
    Args:
        responses_file: Path to file containing LLM responses
        original_data_file: Path to original data file containing ground truth
        output_file: Path to save the fixed responses
    """
    # Load the data
    print(f"Loading responses from {responses_file}...")
    responses_data = load_json(responses_file)
    
    print(f"Loading original data from {original_data_file}...")
    original_data = load_jsonl(original_data_file)
    
    # Create a mapping of question IDs to ground truth answers
    id_to_answer = {item['id']: item['answer'] for item in original_data}
    
    # Fix the ground truth in responses
    print("Fixing ground truth in responses...")
    for item in tqdm(responses_data):
        question_id = item['id']
        if question_id in id_to_answer:
            item['ground_truth'] = id_to_answer[question_id]
        else:
            print(f"Warning: Question ID {question_id} not found in original data")
    
    # Save the fixed responses
    print(f"Saving fixed responses to {output_file}...")
    save_json(responses_data, output_file)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix ground truth in LLM responses")
    parser.add_argument("--responses_file", type=str, required=True,
                      help="File containing LLM responses")
    parser.add_argument("--original_data_file", type=str, required=True,
                      help="Original data file containing ground truth")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Output file for fixed responses")
    
    args = parser.parse_args()
    fix_ground_truth(args.responses_file, args.original_data_file, args.output_file) 