import os
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import re
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor
import anthropic
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Import existing utility functions
from math_utils import parse_ground_truth, is_equiv
from reward_labeling import get_nemotron_reward, get_rise_reward

class MathEvaluator:
    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
        api_key: Optional[str] = None,
        num_samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_nemotron: bool = False,
        use_rise: bool = False
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_nemotron = use_nemotron
        self.use_rise = use_rise
        
        # Initialize model based on type
        if model_type == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif model_type == "gemini":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        elif model_type in ["qwen", "llama"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_chain_of_thought_prompt(self, question: str) -> str:
        return f"""Please solve the following math problem step by step. Show your reasoning and calculations clearly.

Problem: {question}

Let's solve this step by step:"""

    def generate_response(self, prompt: str) -> str:
        if self.model_type == "claude":
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif self.model_type == "gemini":
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return response.text
        else:  # Local models (Qwen, Llama)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_response(self, response: str, ground_truth: str) -> float:
        # Base correctness reward
        reward = 1.0 if is_equiv(response, ground_truth) else -1.0
        
        # Add reward model scores if enabled
        if self.use_nemotron:
            reward += get_nemotron_reward(response)
        if self.use_rise:
            reward += get_rise_reward(response)
            
        return reward

    def self_consistency(self, question: str, ground_truth: str) -> Dict[str, Any]:
        responses = []
        for _ in range(self.num_samples):
            prompt = self.get_chain_of_thought_prompt(question)
            response = self.generate_response(prompt)
            responses.append(response)
        
        # Majority voting based on equivalence with ground truth
        votes = [is_equiv(r, ground_truth) for r in responses]
        final_answer = responses[np.argmax(votes)]
        
        return {
            "responses": responses,
            "final_answer": final_answer,
            "is_correct": is_equiv(final_answer, ground_truth)
        }

    def best_of_n(self, question: str, ground_truth: str) -> Dict[str, Any]:
        responses = []
        rewards = []
        
        for _ in range(self.num_samples):
            prompt = self.get_chain_of_thought_prompt(question)
            response = self.generate_response(prompt)
            reward = self.evaluate_response(response, ground_truth)
            
            responses.append(response)
            rewards.append(reward)
        
        # Select response with highest reward
        best_idx = np.argmax(rewards)
        final_answer = responses[best_idx]
        
        return {
            "responses": responses,
            "rewards": rewards,
            "final_answer": final_answer,
            "is_correct": is_equiv(final_answer, ground_truth)
        }

    def evaluate_dataset(
        self,
        dataset_path: str,
        output_path: str,
        method: str = "self_consistency"
    ) -> Dict[str, Any]:
        results = []
        total_correct = 0
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        for item in tqdm(dataset, desc=f"Evaluating {method}"):
            question = item['question']
            ground_truth = parse_ground_truth(item)
            
            if method == "self_consistency":
                result = self.self_consistency(question, ground_truth)
            else:  # best_of_n
                result = self.best_of_n(question, ground_truth)
            
            result['question'] = question
            result['ground_truth'] = ground_truth
            results.append(result)
            
            if result['is_correct']:
                total_correct += 1
        
        accuracy = total_correct / len(dataset)
        
        # Save results
        output = {
            "model": self.model_name_or_path,
            "method": method,
            "accuracy": accuracy,
            "results": results
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["claude", "gemini", "qwen", "llama"])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, help="API key for Claude or Gemini")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--method", type=str, default="self_consistency", choices=["self_consistency", "best_of_n"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--use_nemotron", action="store_true")
    parser.add_argument("--use_rise", action="store_true")
    
    args = parser.parse_args()
    
    evaluator = MathEvaluator(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        api_key=args.api_key,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_nemotron=args.use_nemotron,
        use_rise=args.use_rise
    )
    
    results = evaluator.evaluate_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        method=args.method
    )
    
    print(f"Evaluation complete. Accuracy: {results['accuracy']:.2%}")
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main() 