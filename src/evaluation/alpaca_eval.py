import os
import json
import argparse
from typing import List, Dict, Optional
from datasets import load_dataset
from litellm import completion
from tqdm import tqdm
import time
from dotenv import load_dotenv
import openai

# Load environment variables from parent directory
load_dotenv('../.env')

def get_client(model_name: str):
    """
    Get the appropriate client based on the model name.
    
    Args:
        model_name: Name of the model (e.g., "deepinfra/Qwen/Qwen2.5-7B-Instruct" or "openai/grok-3-mini-fast-beta")
        
    Returns:
        Tuple of (client, model_name, provider)
    """
    if model_name.startswith("deepinfra/"):
        return None, model_name, "deepinfra"
    elif model_name.startswith("openai/"):
        client = openai.OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        return client, model_name.replace("openai/", ""), "xai"
    else:
        raise ValueError(f"Unsupported model provider in {model_name}")

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
    num_runs: int = 32,  # Default to 32 samples per question
    start_idx: int = 0
) -> None:
    """
    Generate responses for the evaluation set using the specified model.
    
    Args:
        model_name: Name of the model to use
        eval_set: List of examples from the dataset
        output_file: Path to save the outputs
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate (None for no limit)
        num_runs: Number of samples to generate per question
        start_idx: Index to start from (for resuming interrupted runs)
    """
    # Get appropriate client and model name
    client, actual_model_name, provider = get_client(model_name)
    
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
        
        # Check if we already have this example in results
        existing_problem = existing_dict.get(example["instruction"])
        if existing_problem:
            # Check if we need to generate more responses
            valid_responses = [r for r in existing_problem["responses"] if r and not r.startswith("ERROR:")]
            if len(valid_responses) >= num_runs:
                print(f"\nProblem {i}: All {num_runs} responses already present, skipping...")
                continue
            else:
                print(f"\nProblem {i}: Found {len(valid_responses)} valid responses, need {num_runs - len(valid_responses)} more")
                responses = valid_responses
                remaining_runs = num_runs - len(valid_responses)
        else:
            print(f"\nProblem {i}: No existing responses, generating {num_runs} new responses")
            responses = []
            remaining_runs = num_runs
        
        empty_responses = 0
        error_responses = 0
        
        # Generate remaining responses
        for run_num in range(remaining_runs):
            print(f"  Generating response {run_num + 1}/{remaining_runs} for problem {i}...")
            
            # Prepare messages
            messages = [{"role": "user", "content": example["instruction"]}]
            
            # Generate response with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if provider == "deepinfra":
                        # Use litellm for DeepInfra
                        completion_params = {
                            "model": actual_model_name,
                            "messages": messages,
                            "temperature": temperature
                        }
                        
                        if max_tokens is not None:
                            completion_params["max_tokens"] = max_tokens
                        
                        completion_params["api_base"] = "https://api.deepinfra.com/v1/openai"
                        completion_params["api_key"] = os.getenv("DEEPINFRA_API_KEY")
                        
                        response = completion(**completion_params)
                        output_text = response.choices[0].message.content
                    else:  # XAI provider
                        # Use OpenAI client for XAI
                        response = client.chat.completions.create(
                            model=actual_model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        output_text = response.choices[0].message.content
                    
                    if not output_text.strip():
                        empty_responses += 1
                        print(f"  Warning: Empty response for problem {i}")
                    elif output_text.startswith("ERROR:"):
                        error_responses += 1
                        print(f"  Error response for problem {i}: {output_text}")
                    
                    responses.append(output_text)
                    
                    # Update the file after each response
                    result = {
                        "instruction": example["instruction"],
                        "responses": responses,
                        "generator": model_name
                    }
                    
                    # Update results dict and list
                    if example["instruction"] in existing_dict:
                        # Update existing entry
                        for idx, item in enumerate(outputs):
                            if item["instruction"] == example["instruction"]:
                                outputs[idx] = result
                                break
                    else:
                        # Add new entry
                        outputs.append(result)
                        existing_dict[example["instruction"]] = result
                    
                    # Save after each response
                    with open(output_file, 'w') as f:
                        json.dump(outputs, f, indent=2)
                    
                    print(f"  Saved progress: {len(responses)}/{num_runs} responses for problem {i}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"Error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"Failed to generate response for problem {i} after {max_retries} attempts")
                        responses.append(f"ERROR: {str(e)}")
                    time.sleep(5)  # Wait before retrying
            
            if empty_responses > 0:
                print(f"  Warning: {empty_responses} empty responses out of {remaining_runs} for problem {i}")
            if error_responses > 0:
                print(f"  Warning: {error_responses} error responses out of {remaining_runs} for problem {i}")
    
    print(f"Generation complete. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate responses for AlpacaEval dataset")
    parser.add_argument("--model", type=str, required=True,
                      help="Model name (e.g., 'deepinfra/Qwen/Qwen2.5-7B-Instruct' or 'openai/grok-3-mini-fast-beta')")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Output file path")
    parser.add_argument("--subset_size", type=int,
                      help="Number of examples to use (optional)")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int,
                      help="Maximum tokens to generate (optional)")
    parser.add_argument("--num_runs", type=int, default=32,
                      help="Number of samples to generate per question")
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
        num_runs=args.num_runs,
        start_idx=args.start_idx
    )

if __name__ == "__main__":
    main() 