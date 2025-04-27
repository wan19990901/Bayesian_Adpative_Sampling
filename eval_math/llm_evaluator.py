import json
import os
from typing import Dict, List, Optional
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from grader import math_equal_process
import re

class LLMEvaluator:
    def __init__(self):
        self.answer_pattern = re.compile(r"Therefore, the answer is\s*(\d+)")

    def parse_response(self, response: str) -> Dict:
        """
        Parse the LLM response into structured components.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Dictionary containing:
            - reasoning: List of reasoning steps
            - final_answer: Extracted final answer
            - raw_response: Original response
        """
        # Split response into lines
        lines = response.split('\n')
        
        # Extract reasoning steps
        reasoning = []
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new step (starts with number and parenthesis)
            if re.match(r'^\d+\)', line):
                if current_step:
                    reasoning.append(current_step)
                current_step = line
            elif current_step:
                current_step += '\n' + line
                
        if current_step:
            reasoning.append(current_step)
            
        # Extract final answer
        final_answer = None
        answer_match = self.answer_pattern.search(response)
        if answer_match:
            final_answer = answer_match.group(1)
            
        return {
            "reasoning": reasoning,
            "final_answer": final_answer,
            "raw_response": response
        }

    def evaluate_answer(self, prediction: str, ground_truth: str) -> bool:
        """
        Evaluate if the predicted answer matches the ground truth.
        """
        try:
            with ProcessPool(max_workers=1) as pool:
                future = pool.schedule(math_equal_process, 
                                    args=((0, prediction, ground_truth),),
                                    timeout=3)
                return future.result()
        except TimeoutError:
            print("Evaluation timed out")
            return False
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return False

    def evaluate_responses(
        self,
        responses_file: str,
        output_file: str,
        ground_truth_key: str = "answer"
    ) -> Dict:
        """
        Evaluate all responses in a file and save results.
        
        Args:
            responses_file: Path to file containing LLM responses
            output_file: Path to save evaluation results
            ground_truth_key: Key in the data containing ground truth answer
            
        Returns:
            Dictionary containing evaluation results
        """
        with open(responses_file, 'r') as f:
            data = json.load(f)
            
        results = {
            "total_questions": 0,
            "total_responses": 0,
            "correct_answers": 0,
            "missing_answers": 0,
            "detailed_results": []
        }
        
        for item in data:
            question_id = item.get("id")
            ground_truth = item.get(ground_truth_key)
            responses = item.get("responses", [])
            
            if not ground_truth:
                continue
                
            results["total_questions"] += 1
            results["total_responses"] += len(responses)
            
            question_results = {
                "id": question_id,
                "question": item.get("question"),
                "ground_truth": ground_truth,
                "responses": []
            }
            
            for response in responses:
                parsed_response = self.parse_response(response)
                is_correct = False
                
                if parsed_response["final_answer"]:
                    is_correct = self.evaluate_answer(
                        parsed_response["final_answer"],
                        ground_truth
                    )
                    if is_correct:
                        results["correct_answers"] += 1
                else:
                    results["missing_answers"] += 1
                    
                question_results["responses"].append({
                    "parsed_response": parsed_response,
                    "is_correct": is_correct
                })
                
            results["detailed_results"].append(question_results)
            
        # Calculate accuracy
        if results["total_responses"] > 0:
            results["accuracy"] = results["correct_answers"] / results["total_responses"]
        else:
            results["accuracy"] = 0.0
            
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLM responses on math problems")
    parser.add_argument("--responses_file", type=str, required=True,
                      help="File containing LLM responses")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Output file for evaluation results")
    parser.add_argument("--ground_truth_key", type=str, default="answer",
                      help="Key containing ground truth answer")
    
    args = parser.parse_args()
    
    evaluator = LLMEvaluator()
    results = evaluator.evaluate_responses(
        args.responses_file,
        args.output_file,
        args.ground_truth_key
    )
    
    print(f"Evaluation complete. Results saved to {args.output_file}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Total questions: {results['total_questions']}")
    print(f"Total responses: {results['total_responses']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Missing answers: {results['missing_answers']}")

if __name__ == "__main__":
    main() 