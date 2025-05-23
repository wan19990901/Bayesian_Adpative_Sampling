"""
Example script demonstrating LLM inference with BEACON integration.
This example shows how to use BEACON's optimal stopping with different LLM providers
for efficient sampling while maintaining response quality.
"""

import os
import json
from dotenv import load_dotenv
from src.utils.llm import LLMGenerator
from src.BEACON import analyze_stopping, compare_samples
from src.BEACON.stopping import UIP

# Load environment variables
load_dotenv()

def main():
    # Example math problems with varying difficulty
    math_problems = [
        {
            "id": 1,
            "problem": "What is the sum of the first 10 natural numbers?",
            "difficulty": "easy",
            "solution": "55",
            "answer": "55"
        },
        {
            "id": 2,
            "problem": "Solve for x: 2x + 5 = 13",
            "difficulty": "medium",
            "solution": "x = 4",
            "answer": "4"
        },
        {
            "id": 3,
            "problem": "Find the sum of the infinite series: 1 + 1/2 + 1/4 + 1/8 + ...",
            "difficulty": "hard",
            "solution": "2",
            "answer": "2"
        }
    ]

    # Save problems to a temporary file
    with open("temp_problems.jsonl", "w") as f:
        for problem in math_problems:
            f.write(json.dumps(problem) + "\n")

    # Initialize LLM generators with different providers
    providers = {
        "openai": {
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "claude": {
            "model_name": "claude-3-opus-20240229",
            "temperature": 0.7,
            "max_tokens": 2048
        }
    }

    # Initialize UIP with different sampling costs based on difficulty
    sampling_costs = {
        "easy": 0.1,
        "medium": 0.2,
        "hard": 0.3
    }

    results = {}
    for provider, config in providers.items():
        print(f"\nGenerating responses using {provider}...")
        
        # Initialize LLM generator
        generator = LLMGenerator(
            provider=provider,
            api_key=os.getenv(f"{provider.upper()}_API_KEY"),
            **config
        )

        # Generate initial responses
        initial_results = generator.generate_responses(
            data_file="temp_problems.jsonl",
            output_file=f"{provider}_responses.json",
            num_runs=5  # Start with 5 samples
        )

        # Analyze stopping behavior for each problem
        stopping_results = {}
        for problem in math_problems:
            difficulty = problem["difficulty"]
            uip = UIP(
                prior_mean=0.0,
                prior_variance=1.0,
                sampling_cost=sampling_costs[difficulty]
            )
            
            # Get rewards for this problem
            problem_responses = [
                r for r in initial_results 
                if r["id"] == problem["id"]
            ]
            
            # Analyze stopping behavior
            stopping_analysis = analyze_stopping(
                responses_file=f"{provider}_responses.json",
                output_file=f"{provider}_stopping_{problem['id']}.json",
                problem_id=problem["id"]
            )
            
            # Determine if we should generate more samples
            should_stop = uip.should_stop(
                current_rewards=[r["reward"] for r in problem_responses],
                posterior_variance=stopping_analysis["posterior_variance"]
            )
            
            if not should_stop:
                print(f"Generating additional samples for problem {problem['id']}...")
                additional_results = generator.generate_responses(
                    data_file="temp_problems.jsonl",
                    output_file=f"{provider}_additional_{problem['id']}.json",
                    num_runs=3,  # Generate 3 more samples
                    problem_ids=[problem["id"]]
                )
                problem_responses.extend(additional_results)
            
            stopping_results[problem["id"]] = {
                "difficulty": difficulty,
                "samples_used": len(problem_responses),
                "analysis": stopping_analysis,
                "responses": problem_responses
            }

        # Compare and select best responses
        comparison = compare_samples(
            samples_file=f"{provider}_responses.json",
            output_file=f"{provider}_comparison.json"
        )

        results[provider] = {
            "stopping_results": stopping_results,
            "comparison": comparison
        }

        # Print results
        print(f"\nResults for {provider}:")
        for problem_id, data in stopping_results.items():
            print(f"\nProblem {problem_id}:")
            print(f"Difficulty: {data['difficulty']}")
            print(f"Samples used: {data['samples_used']}")
            print("Best response:", data["responses"][0]["text"])

    # Clean up
    import os
    os.remove("temp_problems.jsonl")
    for provider in providers:
        os.remove(f"{provider}_responses.json")
        os.remove(f"{provider}_comparison.json")
        for problem in math_problems:
            os.remove(f"{provider}_stopping_{problem['id']}.json")
            if os.path.exists(f"{provider}_additional_{problem['id']}.json"):
                os.remove(f"{provider}_additional_{problem['id']}.json")

if __name__ == "__main__":
    main() 