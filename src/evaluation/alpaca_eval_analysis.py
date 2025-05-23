import os
import json
import argparse
import logging
from typing import List, Dict, Optional
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def run_alpaca_eval(
    model_outputs: str,
    annotators_config: str = "weighted_alpaca_eval_gpt4_turbo",
    output_dir: Optional[str] = None
) -> None:
    """
    Run AlpacaEval analysis on model outputs.
    
    Args:
        model_outputs: Path to the model outputs JSON file
        annotators_config: Name of the annotators configuration to use
        output_dir: Directory to save analysis results (defaults to same directory as model_outputs)
    
    Raises:
        FileNotFoundError: If the model outputs file doesn't exist
        subprocess.CalledProcessError: If the alpaca_eval command fails
    """
    # Validate input file exists
    if not os.path.exists(model_outputs):
        raise FileNotFoundError(f"Model outputs file not found: {model_outputs}")
    
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
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully analyzed model outputs: {model_outputs}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run alpaca_eval: {str(e)}")
        raise

def analyze_multiple_models(
    model_outputs_dir: str,
    annotators_config: str = "weighted_alpaca_eval_gpt4_turbo"
) -> None:
    """
    Run AlpacaEval analysis on multiple model outputs in a directory.
    
    Args:
        model_outputs_dir: Directory containing model output JSON files
        annotators_config: Name of the annotators configuration to use
    
    Raises:
        FileNotFoundError: If the model outputs directory doesn't exist
    """
    # Validate directory exists
    if not os.path.exists(model_outputs_dir):
        raise FileNotFoundError(f"Model outputs directory not found: {model_outputs_dir}")
    
    # Find all JSON files in the directory
    output_files = list(Path(model_outputs_dir).glob("*_responses.json"))
    
    if not output_files:
        logger.warning(f"No model output files found in {model_outputs_dir}")
        return
    
    logger.info(f"Found {len(output_files)} model output files")
    
    # Run analysis for each model
    for output_file in output_files:
        model_name = output_file.stem.replace("_responses", "")
        logger.info(f"\nAnalyzing model: {model_name}")
        
        output_dir = os.path.join(model_outputs_dir, f"analysis_{model_name}")
        try:
            run_alpaca_eval(
                model_outputs=str(output_file),
                annotators_config=annotators_config,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Failed to analyze model {model_name}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(
        description="Run AlpacaEval analysis on model outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_outputs",
        type=str,
        help="Path to model outputs JSON file"
    )
    parser.add_argument(
        "--model_outputs_dir",
        type=str,
        help="Directory containing model output files"
    )
    parser.add_argument(
        "--annotators_config",
        type=str,
        default="weighted_alpaca_eval_gpt4_turbo",
        help="Name of the annotators configuration to use"
    )
    
    args = parser.parse_args()
    
    try:
        if args.model_outputs:
            # Run analysis for a single model
            run_alpaca_eval(
                model_outputs=args.model_outputs,
                annotators_config=args.annotators_config
            )
        elif args.model_outputs_dir:
            # Run analysis for multiple models
            analyze_multiple_models(
                model_outputs_dir=args.model_outputs_dir,
                annotators_config=args.annotators_config
            )
        else:
            parser.error("Either --model_outputs or --model_outputs_dir must be provided")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 