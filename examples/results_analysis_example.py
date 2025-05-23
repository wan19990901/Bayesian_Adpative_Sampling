"""
Example script demonstrating BEACON's performance analysis and evaluation.
This example shows how to analyze BEACON's efficiency gains and quality preservation
across different tasks and model configurations.
"""

import json
import numpy as np
from src.results import (
    analyze_rewards,
    process_leaderboard,
    evaluate_llm,
    run_alpaca_evaluation
)

def main():
    # Example model outputs with varying configurations
    model_outputs = [
        {
            "id": 1,
            "problem": "What is 2+2?",
            "difficulty": "easy",
            "model_response": "The answer is 4.",
            "ground_truth": "4",
            "reward": 0.95,
            "correctness": 1.0,
            "samples_used": 3,
            "config": "beacon_easy"
        },
        {
            "id": 2,
            "problem": "Solve for x: 2x + 5 = 13",
            "difficulty": "medium",
            "model_response": "x = 4",
            "ground_truth": "4",
            "reward": 0.92,
            "correctness": 1.0,
            "samples_used": 5,
            "config": "beacon_medium"
        },
        {
            "id": 3,
            "problem": "Find the sum of the infinite series: 1 + 1/2 + 1/4 + 1/8 + ...",
            "difficulty": "hard",
            "model_response": "The sum is 2.",
            "ground_truth": "2",
            "reward": 0.98,
            "correctness": 1.0,
            "samples_used": 7,
            "config": "beacon_hard"
        }
    ]

    # Add baseline (fixed BoN) results
    baseline_outputs = [
        {
            **output,
            "config": "fixed_bon",
            "samples_used": 10  # Fixed number of samples
        }
        for output in model_outputs
    ]
    model_outputs.extend(baseline_outputs)

    # Save outputs to a file
    with open("temp_outputs.json", "w") as f:
        json.dump(model_outputs, f, indent=2)

    # Analyze rewards and efficiency
    reward_analysis = analyze_rewards(
        results_file="temp_outputs.json",
        output_file="reward_analysis.json"
    )

    print("\nReward Analysis Results:")
    print(json.dumps(reward_analysis, indent=2))

    # Process leaderboard with efficiency metrics
    leaderboard = process_leaderboard(
        results_file="temp_outputs.json",
        output_file="leaderboard.json",
        include_efficiency=True
    )

    print("\nLeaderboard Results:")
    print(json.dumps(leaderboard, indent=2))

    # Evaluate LLM outputs with quality metrics
    evaluation_results = evaluate_llm(
        outputs_file="temp_outputs.json",
        metrics_file="evaluation_metrics.json"
    )

    print("\nEvaluation Results:")
    print(json.dumps(evaluation_results, indent=2))

    # Calculate efficiency gains
    beacon_results = [r for r in model_outputs if r["config"].startswith("beacon")]
    baseline_results = [r for r in model_outputs if r["config"] == "fixed_bon"]
    
    beacon_samples = sum(r["samples_used"] for r in beacon_results)
    baseline_samples = sum(r["samples_used"] for r in baseline_results)
    
    efficiency_gain = (baseline_samples - beacon_samples) / baseline_samples * 100
    
    print(f"\nEfficiency Analysis:")
    print(f"BEACON samples used: {beacon_samples}")
    print(f"Fixed BoN samples used: {baseline_samples}")
    print(f"Efficiency gain: {efficiency_gain:.1f}%")

    # Analyze quality preservation
    beacon_rewards = [r["reward"] for r in beacon_results]
    baseline_rewards = [r["reward"] for r in baseline_results]
    
    quality_preservation = np.mean(beacon_rewards) / np.mean(baseline_rewards) * 100
    
    print(f"\nQuality Analysis:")
    print(f"BEACON average reward: {np.mean(beacon_rewards):.3f}")
    print(f"Fixed BoN average reward: {np.mean(baseline_rewards):.3f}")
    print(f"Quality preservation: {quality_preservation:.1f}%")

    # Run AlpacaEval (if configured)
    try:
        alpaca_results = run_alpaca_evaluation(
            model_outputs="temp_outputs.json",
            config="weighted_alpaca_eval_gpt4_turbo"
        )
        print("\nAlpacaEval Results:")
        print(json.dumps(alpaca_results, indent=2))
    except Exception as e:
        print(f"\nAlpacaEval not configured: {e}")

    # Clean up
    import os
    os.remove("temp_outputs.json")

if __name__ == "__main__":
    main() 