import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from enum import Enum

class SamplingStrategy(Enum):
    BOS = "bos"  # Bayesian Optimal Stopping
    BEST_OF_N = "best_of_n"  # Best of N sampling

class RewardEvaluator:
    def __init__(self, model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2", device: Optional[str] = None):
        """
        Initialize the reward model evaluator.

        Args:
            model_name (str): HuggingFace model path for the reward model
            device (str, optional): Device to run the model on (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

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
        # Format inputs based on model requirements
        inputs = self.tokenizer(prompt, response, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the reward score
        if self.is_regression:
            reward = outputs.logits.item()
        else:
            # For classification models, use the probability of the positive class
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            reward = probs[0, 1].item()  # Assuming binary classification with positive = 1

        return reward

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