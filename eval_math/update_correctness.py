import json
import os
import re
import numpy as np
from tqdm import tqdm
import argparse
import time

# --- Constants for answer parsing (from LLMEvaluator) ---
BOXED_ANSWER_PATTERN = re.compile(r"(?:\$\$)?\\boxed{((?:[^{}]|{[^{}]*})*)}(?:\$\$)?")
OLD_ANSWER_PATTERN = re.compile(r"Therefore, the answer is\\s*([-.\d]+)") # Made more general for numbers

def parse_llm_response_for_answer(response_text: str) -> str | None:
    """
    Extracts the final answer from an LLM response string.
    Priority is given to \\boxed{} format, then "Therefore, the answer is X"
    """
    final_answer = None
    
    # Try boxed format first
    boxed_match = BOXED_ANSWER_PATTERN.search(response_text)
    if boxed_match:
        final_answer = boxed_match.group(1).strip()
        # Handle potential LaTeX fractions (simple conversion)
        if "\\frac" in final_answer:
            try:
                # Remove \\frac{}{} and split numerator/denominator
                # This is a simplified parser for {num}{den}
                parts = final_answer.replace("\\frac{", "").replace("}", "").split("}{")
                if len(parts) == 2:
                    num = float(parts[0])
                    den = float(parts[1])
                    if den != 0:
                        final_answer = str(num / den)
                    # else: keep as original fraction string if denominator is zero
            except ValueError:
                pass # Keep as original string if conversion fails
    else:
        # Try "Therefore, the answer is X" format
        old_match = OLD_ANSWER_PATTERN.search(response_text)
        if old_match:
            final_answer = old_match.group(1).strip()
            
    return final_answer

def evaluate_math_answer(prediction: str, ground_truth: str, math_equal_func) -> bool:
    """
    Evaluates if the predicted math answer matches the ground truth using math_equal.
    """
    if prediction is None:
        return False
    try:
        return math_equal_func(prediction, str(ground_truth), timeout=True)
    except Exception as e:
        print(f"Error in answer evaluation (pred='{prediction}', gt='{ground_truth}'): {e}")
        return False

def update_evaluation_file(responses_filepath: str, evaluation_filepath: str, ground_truth_key_input: str, math_equal_func):
    """
    Updates an existing evaluation file with re-parsed answers and correctness.
    Preserves existing reward scores. Recalculates aggregate metrics.
    """
    start_run_time = time.time()
    print(f"Loading responses from: {responses_filepath}")
    with open(responses_filepath, 'r') as f:
        responses_data_list = json.load(f)

    print(f"Loading existing evaluation from: {evaluation_filepath}")
    if not os.path.exists(evaluation_filepath):
        print(f"Error: Evaluation file {evaluation_filepath} not found.")
        return
    with open(evaluation_filepath, 'r') as f:
        eval_results = json.load(f)

    responses_map = {item.get("id"): item for item in responses_data_list}

    if "detailed_results" not in eval_results:
        print("Error: 'detailed_results' not found in evaluation file.")
        return

    overall_correct_answers = 0
    overall_missing_answers = 0
    overall_total_responses = 0

    for question_eval_data in tqdm(eval_results.get("detailed_results", []), desc="Updating Questions"):
        q_id = question_eval_data.get("id")
        if not q_id:
            print(f"Skipping item in evaluation data due to missing ID: {question_eval_data.get('question', 'N/A')[:50]}")
            continue

        corresponding_question_from_responses_file = responses_map.get(q_id)
        if not corresponding_question_from_responses_file:
            print(f"Warning: No responses found for question ID {q_id} in responses file. Skipping update for this question.")
            continue

        raw_responses_for_q = corresponding_question_from_responses_file.get("responses", [])
        eval_responses_for_q = question_eval_data.get("responses", [])
        
        ground_truth = question_eval_data.get("ground_truth") 
        if not ground_truth: 
             ground_truth = corresponding_question_from_responses_file.get(ground_truth_key_input) or \
                            corresponding_question_from_responses_file.get("answer") or \
                            corresponding_question_from_responses_file.get("solution")

        if not ground_truth:
            print(f"Warning: No ground truth found for question ID {q_id}. Skipping.")
            continue
        
        question_eval_data["ground_truth"] = ground_truth

        if len(raw_responses_for_q) != len(eval_responses_for_q):
            print(f"Warning: Mismatch in number of responses for Q {q_id}. "
                  f"Responses file: {len(raw_responses_for_q)}, Eval file: {len(eval_responses_for_q)}. "
                  "Skipping response updates for this question.")
            overall_total_responses += len(eval_responses_for_q)
            overall_correct_answers += sum(1 for r in eval_responses_for_q if r.get("is_correct"))
            overall_missing_answers += sum(1 for r in eval_responses_for_q if r.get("final_answer") is None)
            continue

        for i, eval_response_item in enumerate(eval_responses_for_q):
            overall_total_responses += 1
            raw_response_text = raw_responses_for_q[i] 

            parsed_answer = parse_llm_response_for_answer(raw_response_text)
            eval_response_item["final_answer"] = parsed_answer
            eval_response_item["token_count"] = len(raw_response_text.split())

            is_correct = False
            if parsed_answer is not None:
                is_correct = evaluate_math_answer(parsed_answer, ground_truth, math_equal_func)
            
            eval_response_item["is_correct"] = is_correct

            if is_correct:
                overall_correct_answers +=1
            if parsed_answer is None:
                overall_missing_answers +=1
            
        q_answers_for_sc = [r["final_answer"] for r in question_eval_data["responses"] if r["final_answer"] is not None]
        if q_answers_for_sc:
            unique_q_answers, q_counts = np.unique(q_answers_for_sc, return_counts=True)
            q_majority_idx = np.argmax(q_counts)
            q_majority_answer = str(unique_q_answers[q_majority_idx])
            q_is_majority_correct = evaluate_math_answer(q_majority_answer, ground_truth, math_equal_func)
            
            question_eval_data["self_consistency"] = {
                "majority_answer": q_majority_answer,
                "is_majority_correct": q_is_majority_correct,
                "vote_counts": {str(k): int(v) for k, v in zip(unique_q_answers, q_counts)}
            }
        else:
            question_eval_data["self_consistency"] = { "majority_answer": None, "is_majority_correct": False, "vote_counts": {} }
        
        q_best_score = -float('inf')
        q_best_response_item = None
        q_is_best_correct = False

        for resp_data_item in question_eval_data["responses"]:
            score = (1.0 if resp_data_item.get("is_correct") else -1.0)
            
            if "reward_scores" in resp_data_item and isinstance(resp_data_item["reward_scores"], dict):
                for reward_val in resp_data_item["reward_scores"].values():
                    if isinstance(reward_val, (int, float)):
                        score += reward_val 
            
            if score > q_best_score:
                q_best_score = score
                q_best_response_item = resp_data_item 
                q_is_best_correct = resp_data_item.get("is_correct", False)
        
        question_eval_data["best_of_n"] = {
            "best_response": q_best_response_item,
            "best_score": float(q_best_score) if q_best_score != -float('inf') else None,
            "is_best_correct": q_is_best_correct
        }

    eval_results["total_questions"] = len(eval_results.get("detailed_results", []))
    eval_results["total_responses"] = overall_total_responses
    eval_results["correct_answers"] = overall_correct_answers
    eval_results["missing_answers"] = overall_missing_answers

    if overall_total_responses > 0:
        eval_results["accuracy"] = overall_correct_answers / overall_total_responses
    else:
        eval_results["accuracy"] = 0.0

    if eval_results["total_questions"] > 0:
        sc_correct_count = sum(1 for r in eval_results["detailed_results"] if r.get("self_consistency") and r["self_consistency"].get("is_majority_correct"))
        eval_results["self_consistency_acc"] = sc_correct_count / eval_results["total_questions"]
        
        bon_correct_count = sum(1 for r in eval_results["detailed_results"] if r.get("best_of_n") and r["best_of_n"].get("is_best_correct"))
        eval_results["best_of_n_accuracy"] = bon_correct_count / eval_results["total_questions"]
    else:
        eval_results["self_consistency_acc"] = 0.0
        eval_results["best_of_n_accuracy"] = 0.0

    print(f"Saving updated evaluation to: {evaluation_filepath}")
    with open(evaluation_filepath, 'w') as f:
        json.dump(eval_results, f, indent=2)

    end_run_time = time.time()
    print(f"Update complete in {end_run_time - start_run_time:.2f} seconds.")
    print(f"  Overall Accuracy: {eval_results.get('accuracy', 0.0):.2%}")
    print(f"  Self-Consistency Accuracy: {eval_results.get('self_consistency_acc', 0.0):.2%}")
    print(f"  Best-of-N Accuracy: {eval_results.get('best_of_n_accuracy', 0.0):.2%}")

def main():
    parser = argparse.ArgumentParser(description="Update correctness and parsed answers in an existing LLM evaluation file.")
    parser.add_argument("--responses_file", type=str, required=True,
                        help="Path to the JSON file containing raw LLM responses (problem_id, question, responses list).")
    parser.add_argument("--evaluation_file", type=str, required=True,
                        help="Path to the JSON evaluation file to be updated.")
    parser.add_argument("--ground_truth_key", type=str, default="ground_truth",
                        help="Key in the responses data or eval data for ground truth answer.")
    
    args = parser.parse_args()

    try:
        import sys
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_of_current_dir = os.path.dirname(current_script_dir)
        if parent_of_current_dir not in sys.path:
            sys.path.insert(0, parent_of_current_dir)
        
        if current_script_dir not in sys.path:
             sys.path.insert(0, current_script_dir)

        from grader import math_equal
        print("Successfully imported math_equal from grader.")
    except ImportError as e:
        print(f"Error importing math_equal from grader: {e}")
        print("Please ensure grader.py is accessible.")
        print("Falling back to basic string comparison for answers. THIS IS NOT RELIABLE FOR MATH.")
        def basic_math_equal_fallback(pred, gt, **kwargs): 
            return str(pred).strip() == str(gt).strip()
        math_equal = basic_math_equal_fallback
        
    update_evaluation_file(args.responses_file, args.evaluation_file, args.ground_truth_key, math_equal)

if __name__ == "__main__":
    main() 