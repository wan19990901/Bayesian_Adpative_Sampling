import json
import os
from typing import Dict, List, Optional, Tuple
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from grader import math_equal_process
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reward_labeling import get_nemotron_reward, get_rise_reward
import numpy as np

class LLMEvaluator:
    def __init__(self, use_nemotron: bool = True, use_rise: bool = True):
        self.answer_pattern = re.compile(r"Therefore, the answer is\s*(\d+)")
        self.use_nemotron = use_nemotron
        self.use_rise = use_rise
        
        # Initialize reward models if needed
        if use_nemotron:
            self.nemotron_tokenizer = AutoTokenizer.from_pretrained("NVIDIA/Llama-3.1-Nemotron-70B-Reward")
            self.nemotron_model = AutoModelForSequenceClassification.from_pretrained("NVIDIA/Llama-3.1-Nemotron-70B-Reward")
            
        if use_rise:
            self.rise_tokenizer = AutoTokenizer.from_pretrained("R-I-S-E/RISE-Judge-Qwen2.5-32B")
            self.rise_model = AutoModelForSequenceClassification.from_pretrained("R-I-S-E/RISE-Judge-Qwen2.5-32B")

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

    def compute_reward_scores(self, response: str) -> Dict[str, float]:
        """
        Compute reward scores using Nemotron and RISE models.
        """
        scores = {}
        
        if self.use_nemotron:
            scores["nemotron"] = get_nemotron_reward(response)
            
        if self.use_rise:
            scores["rise"] = get_rise_reward(response)
            
        return scores

    def self_consistency(self, responses: List[str], ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate self-consistency across multiple responses.
        
        Args:
            responses: List of responses for the same question
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary containing:
            - consistency_score: Average agreement between responses
            - majority_answer: Most common answer
            - is_majority_correct: Whether majority answer matches ground truth
        """
        answers = []
        correct_count = 0
        
        for response in responses:
            parsed = self.parse_response(response)
            if parsed["final_answer"]:
                answers.append(parsed["final_answer"])
                if self.evaluate_answer(parsed["final_answer"], ground_truth):
                    correct_count += 1
                    
        if not answers:
            return {
                "consistency_score": 0.0,
                "majority_answer": None,
                "is_majority_correct": False
            }
            
        # Find majority answer
        unique_answers, counts = np.unique(answers, return_counts=True)
        majority_answer = unique_answers[np.argmax(counts)]
        
        # Calculate consistency score
        consistency_score = np.max(counts) / len(answers)
        
        return {
            "consistency_score": float(consistency_score),
            "majority_answer": majority_answer,
            "is_majority_correct": self.evaluate_answer(majority_answer, ground_truth),
            "correct_count": correct_count,
            "total_responses": len(responses)
        }

    def best_of_n(self, responses: List[str], ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate using best-of-N sampling.
        
        Args:
            responses: List of responses for the same question
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary containing:
            - best_response: Response with highest combined score
            - best_score: Combined score of best response
            - is_best_correct: Whether best response is correct
        """
        best_score = -float('inf')
        best_response = None
        best_is_correct = False
        
        for response in responses:
            # Compute base correctness
            parsed = self.parse_response(response)
            is_correct = False
            if parsed["final_answer"]:
                is_correct = self.evaluate_answer(parsed["final_answer"], ground_truth)
            
            # Compute reward scores
            reward_scores = self.compute_reward_scores(response)
            
            # Combine scores (you can adjust weights as needed)
            combined_score = (1.0 if is_correct else -1.0)
            if self.use_nemotron:
                combined_score += reward_scores["nemotron"]
            if self.use_rise:
                combined_score += reward_scores["rise"]
                
            if combined_score > best_score:
                best_score = combined_score
                best_response = response
                best_is_correct = is_correct
                
        return {
            "best_response": best_response,
            "best_score": best_score,
            "is_best_correct": best_is_correct
        }

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
                "responses": [],
                "self_consistency": None,
                "best_of_n": None
            }
            
            # Evaluate self-consistency
            question_results["self_consistency"] = self.self_consistency(responses, ground_truth)
            
            # Evaluate best-of-N
            question_results["best_of_n"] = self.best_of_n(responses, ground_truth)
            
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
                    
                # Compute reward scores
                reward_scores = self.compute_reward_scores(response)
                    
                question_results["responses"].append({
                    "parsed_response": parsed_response,
                    "is_correct": is_correct,
                    "reward_scores": reward_scores
                })
                
            results["detailed_results"].append(question_results)
            
        # Calculate accuracy
        if results["total_responses"] > 0:
            results["accuracy"] = results["correct_answers"] / results["total_responses"]
        else:
            results["accuracy"] = 0.0
            
        # Calculate self-consistency metrics
        consistency_scores = [r["self_consistency"]["consistency_score"] for r in results["detailed_results"]]
        results["avg_consistency"] = float(np.mean(consistency_scores))
        
        # Calculate best-of-N metrics
        best_of_n_correct = sum(1 for r in results["detailed_results"] if r["best_of_n"]["is_best_correct"])
        results["best_of_n_accuracy"] = best_of_n_correct / results["total_questions"]
            
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
    parser.add_argument("--use_nemotron", action="store_true",
                      help="Use Nemotron reward model")
    parser.add_argument("--use_rise", action="store_true",
                      help="Use RISE reward model")
    
    args = parser.parse_args()
    
    evaluator = LLMEvaluator(use_nemotron=args.use_nemotron, use_rise=args.use_rise)
    results = evaluator.evaluate_responses(
        args.responses_file,
        args.output_file,
        args.ground_truth_key
    )
    
    print(f"Evaluation complete. Results saved to {args.output_file}")
    print(f"Base Accuracy: {results['accuracy']:.2%}")
    print(f"Self-Consistency Score: {results['avg_consistency']:.2%}")
    print(f"Best-of-N Accuracy: {results['best_of_n_accuracy']:.2%}")
    print(f"Total questions: {results['total_questions']}")
    print(f"Total responses: {results['total_responses']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Missing answers: {results['missing_answers']}")

if __name__ == "__main__":
    main() 