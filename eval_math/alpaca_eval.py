import os
import json
import argparse
from typing import List, Dict, Optional
from datasets import load_dataset
from litellm import completion
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv('../.env')

def load_eval_dataset(subset_size: Optional[int] = None) -> List[Dict]:
    """
    Load the AlpacaEval dataset, optionally taking a subset.
    
    Args:
        subset_size: If provided, only load this many examples
        
    Returns:
        List of examples from the dataset
    """
    print("Loading AlpacaEval dataset...")
    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    
    if subset_size:
        print(f"Taking subset of {subset_size} examples")
        eval_set = eval_set.select(range(min(subset_size, len(eval_set))))
    
    return eval_set

def generate_responses(
    model_name: str,
    eval_set: List[Dict],
    output_file: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    batch_size: int = 1,
    start_idx: int = 0
) -> None:
    """
    Generate responses for the evaluation set using the specified model.
    
    Args:
        model_name: Name of the model to use (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet-20240229")
        eval_set: List of examples from the dataset
        output_file: Path to save the outputs
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate (None for no limit)
        batch_size: Number of examples to process in parallel
        start_idx: Index to start from (for resuming interrupted runs)
    """
    # Load existing outputs if file exists
    existing_outputs = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_outputs = json.load(f)
            print(f"Loaded {len(existing_outputs)} existing outputs")
        except json.JSONDecodeError:
            print("Warning: Could not load existing outputs file, starting fresh")
    
    # Convert existing outputs to dict for easier lookup
    existing_dict = {item["instruction"]: item for item in existing_outputs}
    
    # Process examples
    outputs = existing_outputs.copy()
    for i in tqdm(range(start_idx, len(eval_set)), desc="Generating responses"):
        example = eval_set[i]
        
        # Skip if we already have this example
        if example["instruction"] in existing_dict:
            print(f"Skipping example {i} (already processed)")
            continue
        
        # Prepare messages
        messages = [{"role": "user", "content": example["instruction"]}]
        
        # Generate response with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Prepare completion parameters
                completion_params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature
                }
                
                # Only add max_tokens if specified
                if max_tokens is not None:
                    completion_params["max_tokens"] = max_tokens
                
                response = completion(**completion_params)
                output_text = response.choices[0].message.content
                
                # Add to outputs
                outputs.append({
                    "instruction": example["instruction"],
                    "output": output_text,
                    "generator": model_name
                })
                
                # Save progress after each example
                with open(output_file, 'w') as f:
                    json.dump(outputs, f, indent=2)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"Failed to generate response for example {i} after {max_retries} attempts")
                time.sleep(5)  # Wait before retrying
    
    print(f"Generation complete. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate responses for AlpacaEval dataset")
    parser.add_argument("--model", type=str, required=True,
                      help="Model name (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet-20240229')")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Output file path")
    parser.add_argument("--subset_size", type=int,
                      help="Number of examples to use (optional)")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int,
                      help="Maximum tokens to generate (optional)")
    parser.add_argument("--start_idx", type=int, default=0,
                      help="Index to start from (for resuming interrupted runs)")
    
    args = parser.parse_args()
    
    # Load dataset
    eval_set = load_eval_dataset(args.subset_size)
    
    # Generate responses
    generate_responses(
        model_name=args.model,
        eval_set=eval_set,
        output_file=args.output_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        start_idx=args.start_idx
    )

if __name__ == "__main__":
    main() 