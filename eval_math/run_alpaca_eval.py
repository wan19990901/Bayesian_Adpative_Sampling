import os
from dotenv import load_dotenv
import subprocess

# Load environment variables from parent directory
load_dotenv('../.env')

def run_alpaca_eval_analysis(model_outputs_file):
    """
    Run AlpacaEval analysis with proper environment variables loaded.
    
    Args:
        model_outputs_file: Path to the model outputs JSON file
    """
    # Verify API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Build command
    cmd = [
        "alpaca_eval",
        "--model_outputs", model_outputs_file,
        "--annotators_config", "weighted_alpaca_eval_gpt4_turbo"
    ]
    
    # Run command with environment variables
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=os.environ)

if __name__ == "__main__":
    # Path to your fixed responses file
    model_outputs = "results/alpaca_eval/grok3_responses_alpaca_first.json"
    
    try:
        run_alpaca_eval_analysis(model_outputs)
    except Exception as e:
        print(f"Error running AlpacaEval analysis: {str(e)}") 