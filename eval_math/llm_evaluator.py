import json
import os
from typing import Dict, List, Optional, Tuple, Any
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from grader import math_equal_process
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward_labeling import get_rise_reward
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMEvaluator:
    def __init__(self, use_nemotron: bool = True, use_rise: bool = False, nemotron_api_base: str = "https://integrate.api.nvidia.com/v1"):
        self.answer_pattern = re.compile(r"Therefore, the answer is\s*(\d+)")
        self.use_nemotron = use_nemotron
        self.use_rise = use_rise
        
        # Initialize NVIDIA's reward model client
        if use_nemotron:
            # Check if NVIDIA API key is set
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not nvidia_api_key:
                raise ValueError("NVIDIA_API_KEY environment variable is not set. Please set it in your .env file.")
                
            self.nemotron_client = OpenAI(
                base_url=nemotron_api_base,
                api_key=nvidia_api_key
            )
            
        # Initialize RISE model if needed
        if use_rise:
            # Check if HuggingFace token is set
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise ValueError("HUGGINGFACE_TOKEN environment variable is not set. Please set it in your .env file.")
                
            # Initialize RISE model with token
            self.rise_tokenizer = AutoTokenizer.from_pretrained(
                "R-I-S-E/RISE-Judge-Qwen2.5-32B",
                token=hf_token
            )
            self.rise_model = AutoModelForSequenceClassification.from_pretrained(
                "R-I-S-E/RISE-Judge-Qwen2.5-32B",
                token=hf_token
            )

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

    def get_nemotron_reward(self, prompt: str, response: str) -> float:
        """
        Get reward score from NVIDIA's Nemotron model using their API.
        """
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            response = self.nemotron_client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-reward",
                messages=messages,
                stream=False
            )
            
            # Extract numeric value from the response
            reward_text = response.choices[0].message.content
            # The response format is "reward:value" or just "value"
            if "reward:" in reward_text:
                reward_value = reward_text.split("reward:")[1].strip()
            else:
                reward_value = reward_text.strip()
                
            return float(reward_value)
        except Exception as e:
            print(f"Error getting Nemotron reward: {e}")
            return 0.0

    def get_rise_reward(self, response: str) -> float:
        """
        Get reward score from RISE model using HuggingFace token.
        """
        try:
            inputs = self.rise_tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.rise_model(**inputs)
                scores = torch.sigmoid(outputs.logits)
            return scores.item()
        except Exception as e:
            print(f"Error getting RISE reward: {e}")
            return 0.0

    def compute_reward_scores(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Compute reward scores using Nemotron and RISE models.
        """
        scores = {}
        
        if self.use_nemotron:
            scores["nemotron"] = self.get_nemotron_reward(prompt, response)
            
        if self.use_rise:
            scores["rise"] = self.get_rise_reward(response)
            
        return scores

    def self_consistency(self, prompt: str, responses: List[str], ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate self-consistency across multiple responses.
        
        Args:
            prompt: The original question/prompt
            responses: List of responses for the same question
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary containing:
            - consistency_score: Average agreement between responses
            - majority_answer: Most common answer
            - is_majority_correct: Whether majority answer matches ground truth
            - reward_scores: Reward scores for the majority response
        """
        answers = []
        correct_count = 0
        reward_scores = {}
        
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
                "is_majority_correct": False,
                "reward_scores": {}
            }
            
        # Find majority answer
        unique_answers, counts = np.unique(answers, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_answer = unique_answers[majority_idx]
        
        # Get the response that gave the majority answer
        majority_response = next(
            (r for r in responses if self.parse_response(r)["final_answer"] == majority_answer),
            None
        )
        
        # Calculate consistency score
        consistency_score = np.max(counts) / len(answers)
        
        # Get reward scores for majority response
        if majority_response:
            reward_scores = self.compute_reward_scores(prompt, majority_response)
        
        return {
            "consistency_score": float(consistency_score),
            "majority_answer": majority_answer,
            "is_majority_correct": self.evaluate_answer(majority_answer, ground_truth),
            "correct_count": correct_count,
            "total_responses": len(responses),
            "reward_scores": reward_scores
        }

    def best_of_n(self, prompt: str, responses: List[str], ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate using best-of-N sampling.
        
        Args:
            prompt: The original question/prompt
            responses: List of responses for the same question
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary containing:
            - best_response: Response with highest combined score
            - best_score: Combined score of best response
            - is_best_correct: Whether best response is correct
            - reward_scores: Reward scores for the best response
        """
        best_score = -float('inf')
        best_response = None
        best_is_correct = False
        best_reward_scores = {}
        
        for response in responses:
            # Compute base correctness
            parsed = self.parse_response(response)
            is_correct = False
            if parsed["final_answer"]:
                is_correct = self.evaluate_answer(parsed["final_answer"], ground_truth)
            
            # Compute reward scores
            reward_scores = self.compute_reward_scores(prompt, response)
            
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
                best_reward_scores = reward_scores
                
        return {
            "best_response": best_response,
            "best_score": best_score,
            "is_best_correct": best_is_correct,
            "reward_scores": best_reward_scores
        }

    def evaluate_responses(
        self,
        responses_file: str,
        output_file: str,
        ground_truth_key: str = "ground_truth",
        test_mode: bool = False
    ) -> Dict:
        """
        Evaluate all responses in a file and save results.
        
        Args:
            responses_file: Path to file containing LLM responses
            output_file: Path to save evaluation results
            ground_truth_key: Key in the data containing ground truth answer
            test_mode: If True, only evaluate the first response and print detailed information
        """
        import time
        import os
        start_time = time.time()
        
        with open(responses_file, 'r') as f:
            data = json.load(f)
            
        results = {
            "total_questions": 0,
            "total_responses": 0,
            "correct_answers": 0,
            "missing_answers": 0,
            "detailed_results": []
        }
        
        # Create intermediate results directory
        if not test_mode:
            # Extract dataset name from responses_file
            dataset_name = os.path.basename(responses_file).replace('_responses.json', '')
            # Create reward model string
            reward_models = []
            if self.use_nemotron:
                reward_models.append("nemotron")
            if self.use_rise:
                reward_models.append("rise")
            reward_str = "_".join(reward_models) if reward_models else "no_reward"
            
            # Create directory name
            intermediate_dir = os.path.join(
                os.path.dirname(output_file),
                f"{dataset_name}_{reward_str}_intermediate"
            )
            os.makedirs(intermediate_dir, exist_ok=True)
            print(f"\nSaving intermediate results to: {intermediate_dir}")
        
        # In test mode, only process the first item
        if test_mode:
            data = data[:1]
            print("\n=== Running in Test Mode ===")
            print(f"Will evaluate only the first question with {len(data[0]['responses'])} responses\n")
        
        # Create progress bar for questions
        question_pbar = tqdm(data, desc="Questions", position=0)
        
        for item in question_pbar:
            question_id = item.get("id")
            prompt = item.get("question")
            ground_truth = item.get(ground_truth_key)
            responses = item.get("responses", [])
            
            # Skip if any required field is missing
            if not all([question_id is not None, prompt, ground_truth, responses]):
                print(f"Skipping question {question_id} due to missing data")
                continue
                
            results["total_questions"] += 1
            results["total_responses"] += len(responses)
            
            if test_mode:
                print(f"\nQuestion ID: {question_id}")
                print(f"Question: {prompt}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Number of Responses: {len(responses)}\n")
            
            question_results = {
                "id": question_id,
                "question": prompt,
                "ground_truth": ground_truth,
                "responses": [],
                "self_consistency": None,
                "best_of_n": None
            }
            
            # Create progress bar for responses within this question
            response_pbar = tqdm(responses, desc=f"Responses for Q{question_id}", position=1, leave=False)
            
            for response in response_pbar:
                # Count tokens
                token_count = len(response.split())
                
                # Parse response and get final answer
                answer_match = self.answer_pattern.search(response)
                final_answer = answer_match.group(1) if answer_match else None
                
                is_correct = False
                
                if test_mode:
                    print("\n=== Response Details ===")
                    print(f"Raw Response: {response}")
                    print(f"Final Answer: {final_answer}")
                    print(f"Token Count: {token_count}")
                
                if final_answer:
                    is_correct = self.evaluate_answer(final_answer, str(ground_truth))
                    if is_correct:
                        results["correct_answers"] += 1
                    if test_mode:
                        print(f"Is Correct: {is_correct}")
                else:
                    results["missing_answers"] += 1
                    if test_mode:
                        print("No final answer found in response")
                    
                # Compute reward scores
                reward_scores = {}
                if self.use_nemotron:
                    reward_start = time.time()
                    try:
                        reward_scores["nemotron"] = self.get_nemotron_reward(prompt, response)
                        if test_mode:
                            print(f"Nemotron Reward: {reward_scores['nemotron']}")
                            print(f"Nemotron Evaluation Time: {time.time() - reward_start:.2f} seconds")
                    except Exception as e:
                        print(f"Error getting Nemotron reward: {e}")
                        reward_scores["nemotron"] = 0.0
                if self.use_rise:
                    reward_start = time.time()
                    try:
                        reward_scores["rise"] = self.get_rise_reward(response)
                        if test_mode:
                            print(f"RISE Reward: {reward_scores['rise']}")
                            print(f"RISE Evaluation Time: {time.time() - reward_start:.2f} seconds")
                    except Exception as e:
                        print(f"Error getting RISE reward: {e}")
                        reward_scores["rise"] = 0.0
                    
                question_results["responses"].append({
                    "final_answer": final_answer,
                    "is_correct": is_correct,
                    "token_count": token_count,
                    "reward_scores": reward_scores
                })
                
            # Evaluate self-consistency
            if test_mode:
                print("\n=== Self-Consistency Evaluation ===")
                consistency_start = time.time()
            
            # Get all answers from responses
            answers = [r["final_answer"] for r in question_results["responses"] if r["final_answer"]]
            
            if answers:
                # Find majority answer
                unique_answers, counts = np.unique(answers, return_counts=True)
                majority_idx = np.argmax(counts)
                majority_answer = unique_answers[majority_idx]
                
                # Check if majority answer is correct
                is_majority_correct = self.evaluate_answer(majority_answer, str(ground_truth))
                
                question_results["self_consistency"] = {
                    "majority_answer": str(majority_answer),  # Convert to string for JSON
                    "is_majority_correct": is_majority_correct,
                    "vote_counts": {str(k): int(v) for k, v in zip(unique_answers, counts)}  # Convert to string and int
                }
                
                if test_mode:
                    print(f"Majority Answer: {majority_answer}")
                    print(f"Is Majority Correct: {is_majority_correct}")
                    print(f"Vote Counts: {question_results['self_consistency']['vote_counts']}")
                    print(f"Self-Consistency Evaluation Time: {time.time() - consistency_start:.2f} seconds")
            
            # Evaluate best-of-N
            if test_mode:
                print("\n=== Best-of-N Evaluation ===")
                best_of_n_start = time.time()
            
            best_score = -float('inf')
            best_response = None
            best_is_correct = False
            
            for response_data in question_results["responses"]:
                # Combine correctness and reward scores
                score = (1.0 if response_data["is_correct"] else -1.0)
                if self.use_nemotron:
                    score += response_data["reward_scores"].get("nemotron", 0)
                if self.use_rise:
                    score += response_data["reward_scores"].get("rise", 0)
                    
                if score > best_score:
                    best_score = score
                    best_response = response_data
                    best_is_correct = response_data["is_correct"]
            
            question_results["best_of_n"] = {
                "best_response": best_response,
                "best_score": float(best_score),  # Convert to float for JSON
                "is_best_correct": best_is_correct
            }
            
            if test_mode:
                print(f"Best Response: {best_response['final_answer']}")
                print(f"Best Score: {best_score}")
                print(f"Is Best Correct: {best_is_correct}")
                print(f"Best-of-N Evaluation Time: {time.time() - best_of_n_start:.2f} seconds")
                
            results["detailed_results"].append(question_results)
            
            # Save intermediate results after each question (except in test mode)
            if not test_mode:
                intermediate_file = os.path.join(intermediate_dir, f"q{question_id}.json")
                with open(intermediate_file, 'w') as f:
                    json.dump(question_results, f, indent=2)
                print(f"\nSaved intermediate results for question {question_id} to {intermediate_file}")
            
        # Calculate accuracy
        if results["total_responses"] > 0:
            results["accuracy"] = results["correct_answers"] / results["total_responses"]
        else:
            results["accuracy"] = 0.0
            
        # Calculate self-consistency accuracy
        if results["detailed_results"]:
            self_consistency_correct = sum(1 for r in results["detailed_results"] 
                                         if r["self_consistency"] and r["self_consistency"]["is_majority_correct"])
            results["self_consistency_acc"] = self_consistency_correct / results["total_questions"]
            
            # Calculate best-of-N accuracy
            best_of_n_correct = sum(1 for r in results["detailed_results"] if r["best_of_n"]["is_best_correct"])
            results["best_of_n_accuracy"] = best_of_n_correct / results["total_questions"]
        else:
            results["self_consistency_acc"] = 0.0
            results["best_of_n_accuracy"] = 0.0
            
        if test_mode:
            print("\n=== Final Results ===")
            print(f"Total Questions: {results['total_questions']}")
            print(f"Total Responses: {results['total_responses']}")
            print(f"Correct Answers: {results['correct_answers']}")
            print(f"Missing Answers: {results['missing_answers']}")
            print(f"Accuracy: {results['accuracy']:.2%}")
            print(f"Self-Consistency Accuracy: {results['self_consistency_acc']:.2%}")
            print(f"Best-of-N Accuracy: {results['best_of_n_accuracy']:.2%}")
            print(f"\nTotal Evaluation Time: {time.time() - start_time:.2f} seconds")
            
        # Save final results
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
    parser.add_argument("--ground_truth_key", type=str, default="ground_truth",
                      help="Key containing ground truth answer")
    parser.add_argument("--use_nemotron", action="store_true",
                      help="Use Nemotron reward model")
    parser.add_argument("--use_rise", action="store_true",
                      help="Use RISE reward model")
    parser.add_argument("--nemotron_api_base", type=str, default="https://integrate.api.nvidia.com/v1",
                      help="Base URL for Nemotron API")
    parser.add_argument("--test_mode", action="store_true",
                      help="Only evaluate the first response for testing")
    
    args = parser.parse_args()
    
    evaluator = LLMEvaluator(
        use_nemotron=args.use_nemotron,
        use_rise=args.use_rise,
        nemotron_api_base=args.nemotron_api_base
    )
    results = evaluator.evaluate_responses(
        args.responses_file,
        args.output_file,
        args.ground_truth_key,
        args.test_mode
    )
    
    print(f"Evaluation complete. Results saved to {args.output_file}")
    print(f"Base Accuracy: {results['accuracy']:.2%}")
    print(f"Self-Consistency Score: {results['self_consistency_acc']:.2%}")
    print(f"Best-of-N Accuracy: {results['best_of_n_accuracy']:.2%}")
    print(f"Total questions: {results['total_questions']}")
    print(f"Total responses: {results['total_responses']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Missing answers: {results['missing_answers']}")

if __name__ == "__main__":
    main() 