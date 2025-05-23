import random
import os
import argparse
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from ..utils.llm.llm_inference import LLMGenerator
from ..data.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from ..data.parser import parse_question, parse_ground_truth
from .python_executor import PythonExecutor


class MathEvaluator:
    """Evaluates model performance on mathematical problems."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_vllm: bool = False,
        num_shots: int = 0,
        prompt_type: str = "tool-integrated",
        apply_chat_template: bool = False,
        pipeline_parallel_size: int = 1,
    ):
        """Initialize the math evaluator.
        
        Args:
            model_name: Name or path of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            use_vllm: Whether to use vLLM for inference
            num_shots: Number of few-shot examples
            prompt_type: Type of prompt to use
            apply_chat_template: Whether to apply chat template
            pipeline_parallel_size: Pipeline parallel size for vLLM
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_vllm = use_vllm
        self.num_shots = num_shots
        self.prompt_type = prompt_type
        self.apply_chat_template = apply_chat_template
        self.pipeline_parallel_size = pipeline_parallel_size
        
        # Initialize model and tokenizer
        if use_vllm:
            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=len(available_gpus) // pipeline_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                trust_remote_code=True,
            )
            self.tokenizer = None
            if apply_chat_template:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
        else:
            self.llm = LLMGenerator(
                provider="openai" if "gpt" in model_name.lower() else "claude",
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.tokenizer = None
            
        # Initialize Python executor
        if "pal" in prompt_type:
            self.executor = PythonExecutor(get_answer_expr="solution()")
        else:
            self.executor = PythonExecutor(get_answer_from_stdout=True)

    def evaluate_problems(
        self,
        problems: List[Dict[str, Any]],
        num_samples: int = 1,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate model performance on a list of problems.
        
        Args:
            problems: List of problems to evaluate
            num_samples: Number of samples per problem
            output_file: Optional file to save results
            
        Returns:
            Dictionary containing evaluation results
        """
        samples = []
        for problem in tqdm(problems, desc="Processing problems"):
            # Parse question and ground truth
            question = parse_question(problem)
            if not question:
                continue
                
            gt_cot, gt_ans = parse_ground_truth(problem)
            
            # Construct prompt
            prompt = construct_prompt(
                problem,
                num_shots=self.num_shots,
                prompt_type=self.prompt_type
            )
            
            # Generate responses
            responses = []
            for _ in range(num_samples):
                if self.use_vllm:
                    output = self.llm.generate(
                        prompt,
                        SamplingParams(
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            n=1,
                        )
                    )
                    response = output[0].outputs[0].text
                else:
                    response = self.llm.generate(prompt)
                    
                # Execute response if needed
                if "pal" in self.prompt_type:
                    result = self.executor.execute(response)
                    response = result if result else response
                    
                responses.append(response)
            
            # Create sample
            sample = {
                "idx": problem.get("idx", len(samples)),
                "question": question,
                "gt_cot": gt_cot,
                "gt": gt_ans,
                "responses": responses,
                "prompt": prompt,
            }
            
            # Add additional fields
            for key in [
                "level", "type", "unit", "solution_type",
                "choices", "solution", "ques_type", "ans_type",
                "answer_type", "dataset", "subfield", "field",
                "theorem", "answer"
            ]:
                if key in problem:
                    sample[key] = problem[key]
                    
            samples.append(sample)
            
        # Evaluate samples
        results = self._evaluate_samples(samples)
        
        # Save results if output file specified
        if output_file:
            save_jsonl(samples, output_file)
            with open(output_file.replace(".jsonl", "_metrics.json"), "w") as f:
                json.dump(results, f, indent=4)
                
        return results
    
    def _evaluate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a list of samples.
        
        Args:
            samples: List of samples to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        total = len(samples)
        correct = 0
        execution_success = 0
        
        for sample in samples:
            # Check if any response is correct
            is_correct = False
            for response in sample["responses"]:
                if self._is_correct(response, sample["gt"]):
                    is_correct = True
                    break
                    
            if is_correct:
                correct += 1
                
            # Check execution success
            if "pal" in self.prompt_type:
                for response in sample["responses"]:
                    try:
                        self.executor.execute(response)
                        execution_success += 1
                        break
                    except:
                        continue
                        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        execution_rate = execution_success / total if total > 0 else 0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "execution_success": execution_success,
            "execution_rate": execution_rate
        }
    
    def _is_correct(self, response: str, ground_truth: str) -> bool:
        """Check if a response matches the ground truth.
        
        Args:
            response: Model response
            ground_truth: Ground truth answer
            
        Returns:
            Whether the response is correct
        """
        # Clean response
        response = response.strip().lower()
        ground_truth = ground_truth.strip().lower()
        
        # Handle multiple choice
        if all(c in "abcde" for c in ground_truth):
            return response in ground_truth
            
        # Handle numerical answers
        try:
            resp_num = float(response)
            gt_num = float(ground_truth)
            return abs(resp_num - gt_num) < 1e-6
        except:
            pass
            
        # Handle exact match
        return response == ground_truth


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-4", type=str)
    parser.add_argument("--data_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    args = parser.parse_args()
    
    # Load problems
    problems = load_jsonl(args.data_file)
    
    # Initialize evaluator
    evaluator = MathEvaluator(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_vllm=args.use_vllm,
        num_shots=args.num_shots,
        prompt_type=args.prompt_type,
        apply_chat_template=args.apply_chat_template,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )
    
    # Evaluate problems
    results = evaluator.evaluate_problems(
        problems=problems,
        num_samples=args.num_samples,
        output_file=args.output_file
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total problems: {results['total']}")
    print(f"Correct answers: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Execution success rate: {results['execution_rate']:.2%}")


if __name__ == "__main__":
    main() 