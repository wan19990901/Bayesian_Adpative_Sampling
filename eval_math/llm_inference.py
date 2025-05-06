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
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv('../.env')

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", base_url: str = "https://api.deepinfra.com/v1/openai"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120
        )
        self.model_name = model_name
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        for attempt in range(self.max_retries):
            try:
                # Create messages list
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Create completion parameters
                completion_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature
                }
                
                # Only add max_tokens if it's not None
                if max_tokens is not None:
                    completion_params["max_tokens"] = max_tokens
                
                response = self.client.chat.completions.create(**completion_params)
                
                # Update token counts
                if hasattr(response, 'usage'):
                    self.total_tokens += response.usage.total_tokens
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                
                # Print detailed response information for debugging
                print("\n=== Response Debug Info ===")
                print(f"Model: {self.model_name}")
                print(f"Attempt: {attempt + 1}/{self.max_retries}")
                print(f"Temperature: {temperature}")
                print(f"Response object type: {type(response)}")
                print(f"Response object: {response}")
                print(f"Choices length: {len(response.choices)}")
                if response.choices:
                    print(f"First choice type: {type(response.choices[0])}")
                    print(f"First choice: {response.choices[0]}")
                    print(f"Message type: {type(response.choices[0].message)}")
                    print(f"Message: {response.choices[0].message}")
                    print(f"Content type: {type(response.choices[0].message.content)}")
                    print(f"Content: {response.choices[0].message.content}")
                print("=== End Debug Info ===\n")
                
                if not response.choices[0].message.content:
                    # Check for reasoning_content
                    if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                        print("Note: Using reasoning_content instead of content")
                        return response.choices[0].message.reasoning_content
                    
                    print(f"Warning: Empty response received from model {self.model_name}")
                    print(f"  Attempt {attempt + 1}/{self.max_retries}")
                    print(f"  System prompt length: {len(system_prompt)}")
                    print(f"  User prompt length: {len(user_prompt)}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return ""
                
                return response.choices[0].message.content
                
            except openai.RateLimitError as e:
                print(f"Rate limit error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: Rate limit exceeded after {self.max_retries} attempts"
                
            except openai.AuthenticationError as e:
                print(f"Authentication error: {e}")
                return f"ERROR: Authentication failed - {str(e)}"
                
            except openai.APIError as e:
                print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: API error - {str(e)}"
                
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: Unexpected error - {str(e)}"
        
        return "ERROR: Maximum retries exceeded"

    def get_token_stats(self) -> Dict[str, int]:
        """Return token usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens
        }

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

class DeepInfraProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        # If api_key is not provided, try to get it from environment
        if not api_key:
            api_key = os.getenv('DEEPINFRA_API_KEY')
            if not api_key:
                raise ValueError("DeepInfra API key not provided and DEEPINFRA_API_KEY environment variable not set")
        
        print(f"\nInitializing DeepInfraProvider with:")
        print(f"  Model: {model_name}")
        print(f"  API Key length: {len(api_key)}")
        print(f"  Base URL: https://api.deepinfra.com/v1/openai")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            timeout=120
        )
        self.model_name = model_name
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        for attempt in range(self.max_retries):
            try:
                print(f"\nAttempt {attempt + 1}/{self.max_retries}")
                print(f"System prompt length: {len(system_prompt)}")
                print(f"User prompt length: {len(user_prompt)}")
                print(f"Temperature: {temperature}")
                print(f"Max tokens: {max_tokens}")
                
                # Test API connection first
                print("Testing API connection...")
                test_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    temperature=0.7,
                    max_tokens=10
                )
                print("API connection test successful")
                
                # Now make the actual request
                print("Making actual request...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                print("Request successful")
                # Update token counts
                if hasattr(response, 'usage'):
                    self.total_tokens += response.usage.total_tokens
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                    print(f"Token usage: {response.usage}")
                
                if not response.choices[0].message.content:
                    print(f"Warning: Empty response received from model {self.model_name}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return ""
                
                return response.choices[0].message.content
                
            except openai.RateLimitError as e:
                print(f"Rate limit error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: Rate limit exceeded after {self.max_retries} attempts"
                
            except openai.AuthenticationError as e:
                print(f"Authentication error: {e}")
                return f"ERROR: Authentication failed - {str(e)}"
                
            except openai.APIError as e:
                print(f"API error: {e}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                if hasattr(e, 'response'):
                    print(f"Response status: {e.response.status_code}")
                    print(f"Response body: {e.response.text}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: API error - {str(e)}"
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    print("Traceback:")
                    traceback.print_tb(e.__traceback__)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"ERROR: Unexpected error - {str(e)}"
        
        return "ERROR: Maximum retries exceeded"

    def get_token_stats(self) -> Dict[str, int]:
        """Return token usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens
        }

class LLMGenerator:
    def __init__(
        self,
        provider: str,
        api_key: str,
        model_name: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        num_samples: int = 1,
        system_prompt: str = None,
        example_prompt: str = None
    ):
        """
        Initialize the LLM generator.
        
        Args:
            provider: Name of the LLM provider ("openai", "claude", "gemini", "deepinfra")
            api_key: API key for the chosen provider
            model_name: Name of the model to use (provider-specific)
            base_url: Base URL for the API (for OpenAI-compatible providers)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            num_samples: Number of samples to generate per question
            system_prompt: Optional custom system prompt (defaults to system_prompt.txt)
            example_prompt: Optional custom example prompt (defaults to example_prompt.txt)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.model_name = model_name  # Store model_name as instance variable
        
        # Load prompts from files if not provided
        self.system_prompt = system_prompt or self._load_prompt("system_prompt.txt")
        self.example_prompt = example_prompt or self._load_prompt("example_prompt.txt")
        
        # Initialize the appropriate provider
        if provider == "openai":
            self.llm = OpenAIProvider(api_key, model_name, base_url)
        elif provider == "claude":
            self.llm = ClaudeProvider(api_key)
        elif provider == "gemini":
            self.llm = GeminiProvider(api_key)
        elif provider == "deepinfra":
            self.llm = DeepInfraProvider(api_key, model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt from the prompts directory."""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
        try:
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

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

    def generate_response(self, question: str) -> str:
        """
        Generate a response for a single question using the system prompt and example.
        
        Args:
            question: The math problem question
            
        Returns:
            Generated response string
        """
        # Combine system prompt, example, and question
        user_prompt = f"{self.example_prompt}\n\nQuestion: {question}"
        
        # Remove max_tokens restriction for Grok model
        if self.model_name == "grok-3-mini-fast-beta":
            return self.llm.generate(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=None
            )
        
        return self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

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
        results = existing_results or []
        
        # Load existing results if file exists
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_results = json.load(f)
                print(f"Loaded {len(existing_results)} existing results")
                # Convert list to dict for easier lookup
                results_dict = {r["id"]: r for r in existing_results}
                results = list(results_dict.values())
            except json.JSONDecodeError:
                print("Warning: Could not load existing results file, starting fresh")
                results_dict = {}
        else:
            results_dict = {}
        
        for problem in tqdm(problems[start_id:], desc="Generating responses"):
            problem_id = problem.get("id", len(results))
            
            # Check if we already have this problem in results
            existing_problem = results_dict.get(problem_id)
            if existing_problem:
                # Check if we need to generate more responses
                valid_responses = [r for r in existing_problem["responses"] if r and not r.startswith("ERROR:")]
                if len(valid_responses) >= num_runs:
                    print(f"\nProblem {problem_id}: All {num_runs} responses already present, skipping...")
                    continue
                else:
                    print(f"\nProblem {problem_id}: Found {len(valid_responses)} valid responses, need {num_runs - len(valid_responses)} more")
                    responses = valid_responses
                    remaining_runs = num_runs - len(valid_responses)
            else:
                print(f"\nProblem {problem_id}: No existing responses, generating {num_runs} new responses")
                responses = []
                remaining_runs = num_runs
            
            empty_responses = 0
            error_responses = 0
            
            # Reset token stats for new responses
            if isinstance(self.llm, OpenAIProvider):
                self.llm.total_tokens = 0
                self.llm.total_prompt_tokens = 0
                self.llm.total_completion_tokens = 0
            
            for run_num in range(remaining_runs):
                print(f"  Generating response {run_num + 1}/{remaining_runs} for problem {problem_id}...")
                response = self.generate_response(problem["problem"])
                if not response.strip():
                    empty_responses += 1
                    print(f"  Warning: Empty response for problem {problem_id}")
                elif response.startswith("ERROR:"):
                    error_responses += 1
                    print(f"  Error response for problem {problem_id}: {response}")
                responses.append(response)
                
                # Update the file after each response
                result = {
                    "id": problem_id,
                    "problem": problem["problem"],
                    "solution": problem.get("solution", ""),
                    "answer": problem.get("answer", ""),
                    "responses": responses
                }
                # Only add token stats for new problems
                if isinstance(self.llm, OpenAIProvider):
                    result["token_stats"] = self.llm.get_token_stats()
                
                # Update results dict and list
                results_dict[problem_id] = result
                results = list(results_dict.values())
                
                # Save after each response
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved progress: {len(responses)}/{num_runs} responses for problem {problem_id}")
            
            if empty_responses > 0:
                print(f"  Warning: {empty_responses} empty responses out of {remaining_runs} for problem {problem_id}")
            if error_responses > 0:
                print(f"  Warning: {error_responses} error responses out of {remaining_runs} for problem {problem_id}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate LLM responses for math problems")
    parser.add_argument("--provider", type=str, required=True, 
                      choices=["openai", "claude", "gemini", "deepinfra"],
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
    
    # Generate new results
    results = generator.generate_responses(
        args.data_file,
        args.output_file,
        args.num_runs,
        start_id=args.start_id
    )
    
    print(f"Generation complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 