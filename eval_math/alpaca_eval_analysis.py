import os
import json
import argparse
from typing import List, Dict
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_alpaca_eval(
    model_outputs: str,
    annotators_config: str = "weighted_alpaca_eval_gpt4_turbo",
    output_dir: str = None,
    force: bool = False
) -> None:
    """
    Run AlpacaEval analysis on model outputs.
    
    Args:
        model_outputs: Path to the model outputs JSON file
        annotators_config: Name of the annotators configuration to use
        output_dir: Directory to save analysis results (defaults to same directory as model_outputs)
        force: Whether to force re-evaluation even if model is in precomputed leaderboard
    """
    if output_dir is None:
        output_dir = os.path.dirname(model_outputs)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "alpaca_eval",
        "--model_outputs", model_outputs,
        "--annotators_config", annotators_config
    ]
    
    if force:
        cmd.append("--force")
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def analyze_multiple_models(
    model_outputs_dir: str,
    annotators_config: str = "weighted_alpaca_eval_gpt4_turbo",
    force: bool = False
) -> None:
    """
    Run AlpacaEval analysis on multiple model outputs in a directory.
    
    Args:
        model_outputs_dir: Directory containing model output JSON files
        annotators_config: Name of the annotators configuration to use
        force: Whether to force re-evaluation even if model is in precomputed leaderboard
    """
    # Find all JSON files in the directory
    output_files = list(Path(model_outputs_dir).glob("*_responses.json"))
    
    if not output_files:
        print(f"No model output files found in {model_outputs_dir}")
        return
    
    print(f"Found {len(output_files)} model output files")
    
    # Run analysis for each model
    for output_file in output_files:
        model_name = output_file.stem.replace("_responses", "")
        print(f"\nAnalyzing model: {model_name}")
        
        output_dir = os.path.join(model_outputs_dir, f"analysis_{model_name}")
        run_alpaca_eval(
            model_outputs=str(output_file),
            annotators_config=annotators_config,
            output_dir=output_dir,
            force=force
        )

def main():
    parser = argparse.ArgumentParser(description="Run AlpacaEval analysis")
    parser.add_argument("--model_outputs", type=str,
                      help="Path to model outputs JSON file")
    parser.add_argument("--model_outputs_dir", type=str,
                      help="Directory containing model output files")
    parser.add_argument("--annotators_config", type=str,
                      default="weighted_alpaca_eval_gpt4_turbo",
                      help="Name of the annotators configuration to use")
    parser.add_argument("--force", action="store_true",
                      help="Force re-evaluation even if model is in precomputed leaderboard")
    
    args = parser.parse_args()
    
    if args.model_outputs:
        # Run analysis for a single model
        run_alpaca_eval(
            model_outputs=args.model_outputs,
            annotators_config=args.annotators_config,
            force=args.force
        )
    elif args.model_outputs_dir:
        # Run analysis for multiple models
        analyze_multiple_models(
            model_outputs_dir=args.model_outputs_dir,
            annotators_config=args.annotators_config,
            force=args.force
        )
    else:
        parser.error("Either --model_outputs or --model_outputs_dir must be provided")

if __name__ == "__main__":
    main() 