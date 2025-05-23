"""
Example script demonstrating BEACON's optimal stopping and sample comparison capabilities.
This example shows how BEACON uses Sequential Search with Gaussian Learning (SSGL)
to efficiently sample from language models while maintaining response quality.
"""

import json
import numpy as np
from src.BEACON import analyze_stopping, compare_samples
from src.BEACON.stopping import UIP
from src.BEACON.comparison import RewardAnalyzer

def main():
    # Example math problems with varying difficulty
    problems = [
        {
            "id": 1,
            "problem": "What is 2+2?",
            "difficulty": "easy",
            "responses": [
                {"text": "The answer is 4.", "reward": 0.95},
                {"text": "It's 4.", "reward": 0.90},
                {"text": "I think it's 4.", "reward": 0.85}
            ]
        },
        {
            "id": 2,
            "problem": "Solve for x: 2x + 5 = 13",
            "difficulty": "medium",
            "responses": [
                {"text": "x = 4", "reward": 0.92},
                {"text": "The answer is x = 4", "reward": 0.88},
                {"text": "x equals 4", "reward": 0.85},
                {"text": "Let me solve this...", "reward": 0.75}
            ]
        },
        {
            "id": 3,
            "problem": "Find the sum of the infinite series: 1 + 1/2 + 1/4 + 1/8 + ...",
            "difficulty": "hard",
            "responses": [
                {"text": "The sum is 2.", "reward": 0.98},
                {"text": "Using the geometric series formula...", "reward": 0.85},
                {"text": "Let me think about this...", "reward": 0.70},
                {"text": "I'm not sure.", "reward": 0.50}
            ]
        }
    ]

    # Save problems to a file
    with open("temp_problems.json", "w") as f:
        json.dump(problems, f, indent=2)

    # Initialize UIP with different sampling costs
    sampling_costs = {
        "easy": 0.1,    # Lower cost for easy problems
        "medium": 0.2,  # Medium cost for medium problems
        "hard": 0.3     # Higher cost for hard problems
    }

    # Analyze stopping behavior for each problem
    stopping_results = {}
    for problem in problems:
        difficulty = problem["difficulty"]
        uip = UIP(
            prior_mean=0.0,
            prior_variance=1.0,
            sampling_cost=sampling_costs[difficulty]
        )
        
        # Get rewards in sequence
        rewards = [r["reward"] for r in problem["responses"]]
        
        # Analyze stopping behavior
        stopping_analysis = analyze_stopping(
            responses_file="temp_problems.json",
            output_file=f"stopping_analysis_{problem['id']}.json",
            problem_id=problem["id"]
        )
        
        stopping_results[problem["id"]] = {
            "difficulty": difficulty,
            "sampling_cost": sampling_costs[difficulty],
            "analysis": stopping_analysis,
            "optimal_stop": uip.should_stop(
                current_rewards=rewards,
                posterior_variance=np.var(rewards)
            )
        }

    print("\nStopping Analysis Results:")
    print(json.dumps(stopping_results, indent=2))

    # Compare samples and select best responses
    comparison_results = compare_samples(
        samples_file="temp_problems.json",
        output_file="sample_comparison.json"
    )

    print("\nSample Comparison Results:")
    print(json.dumps(comparison_results, indent=2))

    # Analyze efficiency gains
    total_samples = sum(len(p["responses"]) for p in problems)
    optimal_samples = sum(
        len(p["responses"][:r["optimal_stop"]]) 
        for p, r in zip(problems, stopping_results.values())
    )
    
    efficiency_gain = (total_samples - optimal_samples) / total_samples * 100
    
    print(f"\nEfficiency Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Optimal samples: {optimal_samples}")
    print(f"Efficiency gain: {efficiency_gain:.1f}%")

    # Clean up
    import os
    os.remove("temp_problems.json")
    for problem in problems:
        os.remove(f"stopping_analysis_{problem['id']}.json")

if __name__ == "__main__":
    main() 