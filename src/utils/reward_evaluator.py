import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from enum import Enum
import os
import requests

SKYWORK_REWARD_MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
SKYWORK_API_MODEL = "Skywork-Reward-V2-Llama-3.1-8B"
SKYWORK_API_URL = "https://api.skywork.ai/v1/score"

class SamplingStrategy(Enum):
    BOS = "bos"  # Bayesian Optimal Stopping
    BEST_OF_N = "best_of_n"  # Best of N sampling

class RewardEvaluator:
    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        backend: str = "local",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        max_length: int = 4096,
        trust_remote_code: bool = False,
        use_chat_template: Optional[bool] = None,
        skywork_api_key: Optional[str] = None,
        skywork_api_model: str = SKYWORK_API_MODEL,
        skywork_api_url: str = SKYWORK_API_URL,
        request_timeout: int = 60,
    ):
        """
        Initialize the reward model evaluator.

        Args:
            model_name (str): HuggingFace model path for the reward model
            device (str, optional): Device to run the model on (cuda/cpu)
        """
        self.backend = backend.lower()
        if self.backend not in {"local", "api"}:
            raise ValueError("backend must be one of: local, api")
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_chat_template = (
            use_chat_template
            if use_chat_template is not None
            else model_name.startswith("Skywork/")
        )
        print(f"Reward backend: {self.backend}")

        if self.backend == "api":
            self.skywork_api_key = (
                skywork_api_key
                or os.getenv("SKY_API_KEY")
                or os.getenv("SKYWORK_API_KEY")
            )
            if not self.skywork_api_key:
                raise ValueError("Set SKY_API_KEY or SKYWORK_API_KEY for backend='api'.")
            self.skywork_api_model = skywork_api_model
            self.skywork_api_url = skywork_api_url
            self.request_timeout = request_timeout
            self.session = requests.Session()
            self.model = None
            self.tokenizer = None
            self.is_regression = True
            return

        print(f"Using device: {self.device}")
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if self.use_chat_template:
            model_kwargs["num_labels"] = 1
        if torch_dtype:
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        elif self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.bfloat16
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

        # Check if this is a regression or classification model
        self.is_regression = self.model.config.num_labels == 1
        print(f"Model type: {'regression' if self.is_regression else 'classification'}")

    def compute_reward(self, prompt: str, response: str) -> float:
        """
        Compute the reward score for a single prompt-response pair.

        Args:
            prompt (str): The input prompt/question
            response (str): The response to evaluate

        Returns:
            float: The reward score
        """
        if self.backend == "api":
            return self._compute_reward_api(prompt, response)
        inputs = self._tokenize_pair(prompt, response)

        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                outputs = self.model(inputs)
            else:
                outputs = self.model(**inputs)

        # Get the reward score
        if self.is_regression:
            reward = outputs.logits.item()
        else:
            # For classification models, use the probability of the positive class
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            reward = probs[0, 1].item()  # Assuming binary classification with positive = 1

        return reward

    def _compute_reward_api(self, prompt: str, response: str) -> float:
        headers = {
            "Authorization": f"Bearer {self.skywork_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.skywork_api_model,
            "input": {
                "prompt": prompt,
                "response": response,
            },
        }
        resp = self.session.post(
            self.skywork_api_url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        score = data.get("data", {}).get("score")
        if score is None:
            raise ValueError(f"Malformed Skywork API response: {data}")
        return float(score)

    def _tokenize_pair(self, prompt: str, response: str):
        if self.use_chat_template:
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            return self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

        return self.tokenizer(
            prompt,
            response,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

    def evaluate_responses(self, prompt: str, responses: List[str]) -> List[float]:
        """
        Evaluate multiple responses for the same prompt.

        Args:
            prompt (str): The input prompt/question
            responses (list): List of response strings to evaluate

        Returns:
            list: List of reward scores corresponding to the responses
        """
        rewards = []
        for response in tqdm(responses):
            reward = self.compute_reward(prompt, response)
            rewards.append(reward)
        return rewards

class BayesianOptimalStopping:
    def __init__(
        self,
        alpha0: float = 1.0,
        nu0: float = 1.0,
        beta0: float = 1.0,
        mu0: float = 0.0,
        cost_per_sample: float = 0.1,
        max_iterations: int = 100
    ):
        """
        Initialize the Bayesian Optimal Stopping model.

        Args:
            alpha0 (float): Initial alpha parameter for the normal-gamma prior
            nu0 (float): Initial nu parameter for the normal-gamma prior
            beta0 (float): Initial beta parameter for the normal-gamma prior
            mu0 (float): Initial mu parameter for the normal-gamma prior
            cost_per_sample (float): Cost per additional sample
            max_iterations (int): Maximum number of iterations to run
        """
        self.alpha0 = alpha0
        self.nu0 = nu0
        self.beta0 = beta0
        self.mu0 = mu0
        self.cost_per_sample = cost_per_sample
        self.max_iterations = max_iterations

    def compute_initial_parameters(self, initial_rewards: np.ndarray) -> Tuple[float, float, float]:
        """Compute initial parameters for BOS."""
        n = len(initial_rewards)
        mean = np.mean(initial_rewards)
        var = np.var(initial_rewards, ddof=1)
        
        alpha = self.alpha0 + n/2
        nu = self.nu0 + n
        beta = self.beta0 + n/2 * var + (n * self.nu0 * (mean - self.mu0)**2) / (2 * (self.nu0 + n))
        mu = (self.nu0 * self.mu0 + n * mean) / (self.nu0 + n)
        
        return mean, mu, np.sqrt(beta / (alpha * nu))

    def update_parameters(self, z_k: float, mu_k: float, sigma_k: float, new_reward: float, k: int) -> Tuple[float, float, float]:
        """Update parameters after observing a new reward."""
        alpha = self.alpha0 + (k + 1)/2
        nu = self.nu0 + k + 1
        beta = self.beta0 + (k + 1)/2 * sigma_k**2 + ((k + 1) * self.nu0 * (mu_k - self.mu0)**2) / (2 * (self.nu0 + k + 1))
        mu = (self.nu0 * self.mu0 + (k + 1) * z_k) / (self.nu0 + k + 1)
        
        return z_k, mu, np.sqrt(beta / (alpha * nu))

    def should_continue_sampling(self, z_k: float, mu_k: float, sigma_k: float, k: int) -> bool:
        """Determine whether to continue sampling based on current parameters."""
        if k >= self.max_iterations:
            return False
            
        z_val = (z_k - mu_k) / sigma_k if sigma_k > 1e-9 else (1000.0 if z_k > mu_k else (-1000.0 if z_k < mu_k else 0.0))
        c_effective = self.cost_per_sample / sigma_k if sigma_k > 1e-9 else float('inf')
        
        # Simplified stopping rule - can be replaced with more sophisticated logic
        return z_val > c_effective

    def run_sampling(
        self,
        initial_rewards: List[float],
        max_samples: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Run the BOS sampling process.

        Args:
            initial_rewards: List of initial reward values
            max_samples: Optional maximum number of samples to collect

        Returns:
            Tuple containing:
            - Number of samples used
            - Maximum reward found
        """
        if len(initial_rewards) < 3:
            raise ValueError("BOS requires at least 3 initial samples")

        all_rewards = list(initial_rewards)
        z_k, mu_k, sigma_k = self.compute_initial_parameters(np.array(initial_rewards))
        k = len(initial_rewards)
        samples_used = k

        max_reward = max(initial_rewards)

        while self.should_continue_sampling(z_k, mu_k, sigma_k, k) and (max_samples is None or k < max_samples):
            # Generate a new sample - in practice, this would come from your model
            new_reward = np.random.normal(mu_k, sigma_k)  # Use current estimate of distribution
            
            all_rewards.append(new_reward)
            
            if new_reward > max_reward:
                max_reward = new_reward
                
            z_k, mu_k, sigma_k = self.update_parameters(z_k, mu_k, sigma_k, new_reward, k)
            k += 1
            samples_used = k

        return samples_used, max_reward

class BestOfNSampler:
    def __init__(self, n: int = 5):
        """
        Initialize the Best-of-N sampler.

        Args:
            n (int): Number of samples to generate and select the best from
        """
        self.n = n

    def run_sampling(
        self,
        initial_rewards: List[float],
        max_samples: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Run Best-of-N sampling.

        Args:
            initial_rewards: List of initial reward values
            max_samples: Optional maximum number of samples to collect

        Returns:
            Tuple containing:
            - Number of samples used
            - Maximum reward found
        """
        if len(initial_rewards) < 1:
            raise ValueError("Best-of-N requires at least 1 initial sample")

        all_rewards = list(initial_rewards)
        k = len(initial_rewards)
        samples_used = k

        # Continue sampling until we have n samples or reach max_samples
        while (max_samples is None or k < max_samples) and k < self.n:
            # Generate a new sample - in practice, this would come from your model
            new_reward = np.random.normal(np.mean(all_rewards), np.std(all_rewards))
            
            all_rewards.append(new_reward)
            k += 1
            samples_used = k

        # Find the best reward
        max_reward = max(all_rewards)
        return samples_used, max_reward

class ResponseSampler:
    def __init__(
        self,
        strategy: Union[SamplingStrategy, str] = SamplingStrategy.BEST_OF_N,
        n: int = 5,
        bos_params: Optional[Dict] = None
    ):
        """
        Initialize the response sampler with a specific strategy.

        Args:
            strategy: Sampling strategy to use (BOS or Best-of-N)
            n: Number of samples for Best-of-N strategy
            bos_params: Parameters for Bayesian Optimal Stopping
        """
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy.lower())
        
        self.strategy = strategy
        
        if strategy == SamplingStrategy.BOS:
            bos_params = bos_params or {}
            self.sampler = BayesianOptimalStopping(**bos_params)
        else:
            self.sampler = BestOfNSampler(n=n)

    def run_sampling(
        self,
        initial_rewards: List[float],
        max_samples: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Run sampling with the selected strategy.

        Args:
            initial_rewards: List of initial reward values
            max_samples: Optional maximum number of samples to collect

        Returns:
            Tuple containing:
            - Number of samples used
            - Maximum reward found
        """
        return self.sampler.run_sampling(initial_rewards, max_samples) 
