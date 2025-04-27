import os
import json
import argparse
from typing import Dict, List, Optional
import openai
import anthropic
from google import generativeai as genai
from tqdm import tqdm
import time
import random
from grader import math_equal_process
from concurrent.futures import TimeoutError
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "grok-3-fast-beta", base_url: str = "https://api.x.ai/v1"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"XAI API error: {e}")
            return ""

class ClaudeProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",  # or other Claude model version
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Claude API error: {e}")
            return ""

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ""

class LLMGenerator:
    def __init__(
        self,
        provider: str,
        api_key: str,
        model_name: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        num_samples: int = 1
    ):
        """
        Initialize the LLM generator.
        
        Args:
            provider: Name of the LLM provider ("openai", "claude", "gemini")
            api_key: API key for the chosen provider
            model_name: Name of the model to use (provider-specific)
            base_url: Base URL for the API (for OpenAI-compatible providers)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            num_samples: Number of samples to generate per question
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        
        # Initialize the appropriate provider
        if provider == "openai":
            self.llm = OpenAIProvider(api_key, model_name, base_url)
        elif provider == "claude":
            self.llm = ClaudeProvider(api_key)
        elif provider == "gemini":
            self.llm = GeminiProvider(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Example problem for one-shot chain of thought
        self.example_problem = {
            "question": "If a triangle has sides of length 3, 4, and 5 units, what is its area?",
            "thought": "Let me solve this step by step:\n1) This is a 3-4-5 triangle, which is a right triangle (by the Pythagorean theorem)\n2) For a right triangle, we can use the formula: Area = (base × height) ÷ 2\n3) Using the 3 and 4 as base and height: Area = (3 × 4) ÷ 2\n4) Area = 12 ÷ 2 = 6",
            "answer": "6"
        }

    def load_math_problems(self, data_file: str) -> List[Dict]:
        """
        Load math problems from the specified file.
        
        Args:
            data_file: Path to JSONL file containing math problems (one JSON object per line)
            
        Returns:
            List of problem dictionaries
        """
        problems = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    problems.append(json.loads(line))
        return problems

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM for a given prompt with chain-of-thought.
        """
        system_prompt = """You are a helpful math assistant that solves problems step by step. 
Always follow this exact format in your response:

1. Start with "Let me solve this step by step:"
2. Break down the solution into numbered steps
3. For each step, explain your reasoning clearly
4. End with "Therefore, the answer is X." where X is the final answer

Here's an example of the expected format:

Let me solve this step by step:
1) First step explanation
2) Second step explanation
...
N) Final step explanation
Therefore, the answer is X."""
        
        cot_prompt = f"""Here's an example of how to solve a math problem step by step:

Question: {self.example_problem['question']}
Let me solve this step by step:
{self.example_problem['thought']}
Therefore, the answer is {self.example_problem['answer']}.

Now, solve this new problem:
Question: {prompt}
Let me solve this step by step:"""

        return self.llm.generate(system_prompt, cot_prompt, self.temperature, self.max_tokens)

    def generate_responses(
        self,
        data_file: str,
        output_file: str,
        num_runs: int = 1,
        start_id: int = 0,
        existing_results: List[Dict] = None
    ) -> List[Dict]:
        """
        Generate responses for all problems and save results incrementally.
        
        Args:
            data_file: Path to input data file
            output_file: Path to save results
            num_runs: Number of times to run each problem
            start_id: Start from this question ID
            existing_results: List of existing results to append to
            
        Returns:
            List of results with generated responses
        """
        problems = self.load_math_problems(data_file)
        results = existing_results if existing_results else []
        
        # Find the last processed ID
        last_id = max([r.get('id', 0) for r in results]) if results else -1
        start_idx = max(start_id, last_id + 1)
        
        print(f"Starting from problem ID {start_idx}")
        
        for problem in tqdm(problems[start_idx:], desc="Generating responses"):
            responses = []
            for _ in range(num_runs):
                response = self.generate_response(problem["question"])
                responses.append(response)
                # Add delay to avoid rate limiting
                time.sleep(random.uniform(0.5, 1.0))
            
            result = {
                "id": problem.get("id", start_idx),
                "question": problem.get("question"),
                "ground_truth": problem.get("answer"),
                "responses": responses
            }
            results.append(result)
            
            # Save intermediate results after each problem
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            start_idx += 1
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate LLM responses for math problems")
    parser.add_argument("--provider", type=str, required=True, 
                      choices=["openai", "claude", "gemini"],
                      help="LLM provider name")
    parser.add_argument("--model_name", type=str,
                      help="Model name to use (provider-specific)")
    parser.add_argument("--base_url", type=str,
                      help="Base URL for the API (for OpenAI-compatible providers)")
    parser.add_argument("--data_file", type=str, required=True,
                      help="Input data file containing math problems")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Output file path")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048,
                      help="Maximum tokens to generate")
    parser.add_argument("--num_runs", type=int, default=1,
                      help="Number of runs per question")
    parser.add_argument("--start_id", type=int, default=0,
                      help="Start from this question ID")
    parser.add_argument("--api_key", type=str, required=True,
                      help="API key for the chosen provider")
    
    args = parser.parse_args()
    
    generator = LLMGenerator(
        provider=args.provider,
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples=1
    )
    
    # Load existing results if file exists
    existing_results = []
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing results")
        except json.JSONDecodeError:
            print("Warning: Could not load existing results file, starting fresh")
    
    # Generate new results
    results = generator.generate_responses(
        args.data_file,
        args.output_file,
        args.num_runs,
        start_id=args.start_id,
        existing_results=existing_results
    )
    
    print(f"Generation complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 