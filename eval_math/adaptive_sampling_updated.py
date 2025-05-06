import json
import os
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import math
from tqdm import tqdm
from numba import njit

# Add BOS helper functions
@njit
def _t_pdf(x, df):
    if df <= 1e-9: return 0.0
    x_f = float(x)
    df_f = float(df)
    try:
        term1 = (1.0 + x_f**2 / df_f)**(-(df_f + 1.0) / 2.0)
        log_gamma_num = math.lgamma((df_f + 1.0) / 2.0)
        log_gamma_den = math.lgamma(df_f / 2.0)
        log_density = math.log(term1) + log_gamma_num - log_gamma_den - 0.5 * math.log(df_f * math.pi)
        return math.exp(log_density)
    except:
        return 0.0

@njit
def _norm_pdf(x):
    x_f = float(x)
    return math.exp(-0.5 * x_f**2) / math.sqrt(2.0 * math.pi)

@njit
def _t_cdf(x, df):
    if df <= 0: return 0.0 if x < 0 else (1.0 if x > 0 else 0.5)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@njit
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@njit
def _t_ppf(p, df):
    """
    Compute the inverse CDF (PPF) of the t-distribution using a simple approximation.
    This is a Numba-compatible implementation.
    """
    # For small degrees of freedom, use a simple approximation
    if df <= 2:
        return np.sqrt(df * (p**(-2/df) - 1))
    
    # For larger degrees of freedom, use normal approximation
    z = _norm_ppf(p)
    return z * (1 + (1 + z**2)/(4*df) + (3 + 5*z**2 + 2*z**4)/(96*df**2))

@njit
def _norm_ppf(p):
    """
    Compute the inverse CDF (PPF) of the normal distribution.
    This is a Numba-compatible implementation.
    """
    # Simple approximation of normal PPF
    if p < 0.5:
        return -_norm_ppf(1 - p)
    
    t = np.sqrt(-2 * np.log(1 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    
    return t - (c0 + c1*t + c2*t**2)/(1 + d1*t + d2*t**2 + d3*t**3)

@njit
def adaptive_ignore_update(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, alpha=0.03):
    """
    Update parameters with adaptive ignoring of extreme low observations.
    """
    df_k = 2 * alpha0 + k
    threshold_k = mu_k + sigma_k * _t_ppf(alpha, df_k)

    # Ignore update if below threshold, use mu_k as observation
    y_adj = y if y >= threshold_k else mu_k  # effectively ignoring extreme observation
    
    # Update best offer
    z_k_plus = max(z_k, y)

    # Update mu and sigma normally using y_adj
    mu_k_plus = mu_k + (y_adj - mu_k) / (nu0 + k + 1)

    if k < 3:
        nu_k = nu0 + k
        alpha_k = alpha0 + k / 2
        beta_k = beta0 + (k * nu0 / (nu0 + k)) * ((mu_k - mu0)**2)
        sigma_k_plus = np.sqrt((1 + 1 / nu_k) * (2 * beta_k / (2 * alpha_k)))
    else:
        L_k_plus = np.sqrt((1 - (1 / (nu0 + k + 1))**2) / (2 * alpha0 + k + 1))
        sigma_k_plus = L_k_plus * np.sqrt((2 * alpha0 + k) * sigma_k**2 + (y_adj - mu_k)**2)

    return z_k_plus, mu_k_plus, sigma_k_plus

@njit
def update_parameters_bos(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0):
    z_k_plus = max(z_k, y)
    mu_k_plus = mu_k + (y - mu_k) / (nu0 + k + 1.0)
    
    L_k_plus_sq = (1.0 - (1.0/(nu0 + k + 1.0))**2) / (2.0*alpha0 + k + 1.0) if (2.0*alpha0 + k + 1.0) > 1e-9 else float('inf')
    sigma_k_plus_sq = L_k_plus_sq * ((2.0*alpha0 + k) * sigma_k**2 + (y - mu_k)**2)
    sigma_k_plus = np.sqrt(sigma_k_plus_sq) if sigma_k_plus_sq > 1e-9 else 1e-9
    
    return z_k_plus, mu_k_plus, sigma_k_plus

@njit
def compute_initial_parameters_bos(x_values, alpha0, nu0, beta0, mu0):
    k = len(x_values)
    if k != 3: raise ValueError(f"BOS initial parameter computation expects exactly 3 observations. Got {k}.")
    
    x_bar = np.mean(x_values)
    nu_k = nu0 + k
    mu_k = (nu0 * mu0 + np.sum(x_values)) / nu_k
    alpha_k = alpha0 + k/2.0
    
    sum_squared_diff = np.sum((x_values - x_bar)**2)
    beta_k = beta0 + sum_squared_diff + (k * nu0 / (nu0 + k)) * ((x_bar - mu0)**2) if nu0 + k > 1e-9 else beta0
    
    sigma_k_sq = (1.0 + 1.0/nu_k) * (beta_k / alpha_k) if alpha_k > 1e-9 else float('inf')
    sigma_k = np.sqrt(sigma_k_sq) if sigma_k_sq > 1e-9 else 1e-9
    
    z_k = np.max(x_values)
    return z_k, mu_k, sigma_k

class AdaptiveSampler:
    def __init__(self, results_dir: str):
        """
        Initialize the adaptive sampler with the directory containing evaluation results.
        
        Args:
            results_dir: Directory containing the evaluation results and intermediate files
        """
        self.results_dir = results_dir
        self.results = {}
        self.load_results()
        
        # Initialize BOS parameters
        self.alpha0 = 2.0
        self.nu0 = 1.0
        self.beta0 = 1.0
        self.mu0 = 0.0
        self.h_matrix = None
        self.z_grid = None
        self.c_grid = None
    
    def load_results(self):
        """Load all evaluation results from the directory."""
        # Find all intermediate directories
        for dir_name in os.listdir(self.results_dir):
            if dir_name.endswith('_intermediate'):
                # Extract dataset name and reward model from directory name
                # Format: xai_responses_amc23.json_nemotron_intermediate
                parts = dir_name.split('_')
                dataset = '_'.join(parts[:-2])  # Join all parts except last two
                reward_models = parts[-2]  # Second to last part
                
                if dataset not in self.results:
                    self.results[dataset] = {}
                
                self.results[dataset][reward_models] = {
                    'questions': {},
                    'rewards': {}
                }
                
                # Load all question files
                intermediate_dir = os.path.join(self.results_dir, dir_name)
                for file_name in os.listdir(intermediate_dir):
                    if file_name.startswith('q') and file_name.endswith('.json'):
                        question_id = int(file_name[1:-5])  # Remove 'q' and '.json'
                        with open(os.path.join(intermediate_dir, file_name), 'r') as f:
                            question_data = json.load(f)
                            
                        self.results[dataset][reward_models]['questions'][question_id] = question_data
                        
                        # Extract rewards for each model
                        for response in question_data['responses']:
                            for model, reward in response['reward_scores'].items():
                                if model not in self.results[dataset][reward_models]['rewards']:
                                    self.results[dataset][reward_models]['rewards'][model] = []
                                self.results[dataset][reward_models]['rewards'][model].append(reward)
    
    def get_best_response_accuracy(self, responses, n_samples, reward_model):
        """Calculate accuracy based on whether the response with highest reward is correct."""
        if n_samples > len(responses):
            n_samples = len(responses)
        
        if n_samples == 0:
            return 0.0
            
        # Get rewards and correctness for sampled responses
        rewards = [(r['reward_scores'][reward_model], r.get('is_correct', False)) 
                  for r in responses[:n_samples]]
        
        # Find the response with highest reward
        max_reward_idx = max(range(len(rewards)), key=lambda i: rewards[i][0])
        
        # Return whether the highest-reward response is correct
        return 1.0 if rewards[max_reward_idx][1] else 0.0

    def wilson_score_interval(self, p: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Wilson score interval for a proportion.
        
        Args:
            p: The proportion of successes
            n: The total number of trials
            alpha: The significance level (default: 0.05 for 95% confidence)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        z = stats.norm.ppf(1 - alpha/2)
        denominator = 1 + z**2/n
        center = (p + z**2/(2*n))/denominator
        spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denominator
        return center - spread, center + spread
    
    def bcs_sampling(self, question_id: int, dataset: str, reward_models: str,
                    max_ci_width: float = 0.1, t_min: int = 5, t_max: int = 32,
                    alpha_ci: float = 0.05) -> Dict:
        """
        Implement Binary Confidence Stopping (BCS) for reward sampling.
        
        Args:
            question_id: The question ID to sample from
            dataset: The dataset name
            reward_models: The reward model to use
            max_ci_width: Maximum allowed confidence interval width
            t_min: Minimum number of samples
            t_max: Maximum number of samples
            alpha_ci: Confidence level for intervals
            
        Returns:
            Dictionary containing sampling results
        """
        question_data = self.results[dataset][reward_models]['questions'][question_id]
        responses = question_data['responses']
        
        # Initialize tracking variables
        samples_used = 0
        max_reward = float('-inf')
        rewards = []
        
        # Sample until we hit t_max or confidence interval is narrow enough
        for t in range(t_max):
            if t < len(responses):
                reward = responses[t]['reward_scores'][reward_models]
                rewards.append(reward)
                samples_used += 1
                max_reward = max(max_reward, reward)
                
                # Only check confidence interval after minimum samples
                if t >= t_min:
                    # Calculate proportion of rewards close to max
                    close_to_max = sum(1 for r in rewards if abs(r - max_reward) < 1e-6)
                    p = close_to_max / len(rewards)
                    
                    # Calculate Wilson score interval
                    lower, upper = self.wilson_score_interval(p, len(rewards), alpha_ci)
                    
                    # Check if interval width is small enough
                    if upper - lower <= max_ci_width:
                        break
        
        return {
            'samples_used': samples_used,
            'max_reward': max_reward,
            'accuracy': self.get_best_response_accuracy(responses, samples_used, reward_models)
        }
    
    def bos_sampling(self, question_id: int, dataset: str, reward_models: str,
                    cost_per_sample: float = 0.1, t_min: int = 3, t_max: int = 32,
                    use_adaptive_ignore: bool = True, alpha: float = 0.03) -> Dict:
        """
        Implement Bayesian Optimal Stopping for reward sampling.
        Note: BOS requires exactly 3 initial samples regardless of t_min parameter.
        
        Args:
            question_id: The question ID to sample from
            dataset: The dataset name
            reward_models: The reward model to use
            cost_per_sample: Cost per sample for stopping decision
            t_min: Minimum number of samples (ignored, always uses 3 for BOS)
            t_max: Maximum number of samples
            use_adaptive_ignore: Whether to use adaptive ignoring of extreme low observations
            alpha: Tail quantile for ignoring extreme observations (default: 0.03)
            
        Returns:
            Dictionary containing sampling results
        """
        question_data = self.results[dataset][reward_models]['questions'][question_id]
        responses = question_data['responses']
        
        # BOS requires exactly 3 initial samples
        if len(responses) < 3:
            return {
                'samples_used': len(responses),
                'max_reward': max(r['reward_scores'][reward_models] for r in responses),
                'accuracy': self.get_best_response_accuracy(responses, len(responses), reward_models)
            }
        
        # Get exactly 3 initial samples for BOS
        initial_rewards = [responses[i]['reward_scores'][reward_models] for i in range(3)]
        initial_rewards = np.array(initial_rewards)
        
        # Initialize BOS parameters
        z_k, mu_k, sigma_k = compute_initial_parameters_bos(
            initial_rewards, self.alpha0, self.nu0, self.beta0, self.mu0
        )
        
        samples_used = 3  # Start with 3 samples
        max_reward = max(initial_rewards)
        
        # Sample until stopping condition is met
        for k in range(3, min(len(responses), t_max)):  # Start from 3rd sample
            z_val = (z_k - mu_k) / sigma_k if sigma_k > 1e-9 else float('inf')
            c_effective = cost_per_sample / sigma_k if sigma_k > 1e-9 else float('inf')
            
            # Check stopping condition
            if c_effective >= z_val:
                break
                
            reward = responses[k]['reward_scores'][reward_models]
            
            # Use adaptive ignoring if enabled
            if use_adaptive_ignore:
                z_k, mu_k, sigma_k = adaptive_ignore_update(
                    z_k, mu_k, sigma_k, reward, k,
                    self.alpha0, self.nu0, self.beta0, self.mu0, alpha
                )
            else:
                z_k, mu_k, sigma_k = update_parameters_bos(
                    z_k, mu_k, sigma_k, reward, k,
                    self.alpha0, self.nu0, self.beta0, self.mu0
                )
            
            max_reward = max(max_reward, reward)
            samples_used += 1
        
        return {
            'samples_used': samples_used,
            'max_reward': max_reward,
            'accuracy': self.get_best_response_accuracy(responses, samples_used, reward_models)
        }

    def compare_sampling_methods(self, dataset: str, reward_models: str,
                               ci_widths: List[float] = [0.05, 0.1, 0.2],
                               t_min: int = 5, t_max: int = 32,
                               alpha_ci: float = 0.05,
                               use_adaptive_ignore: bool = True,
                               alpha: float = 0.03) -> Dict:
        """Compare different sampling methods."""
        results = {
            'fixed': {},
            'bcs': {},
            'bos': {},
            'bos_adaptive': {}
        }
        
        question_ids = list(self.results[dataset][reward_models]['questions'].keys())
        
        # Test fixed sampling
        for n in [t_min, t_max]:
            samples = []
            max_rewards = []
            accuracies = []
            
            for qid in tqdm(question_ids, desc=f'Fixed sampling (n={n})'):
                responses = self.results[dataset][reward_models]['questions'][qid]['responses'][:n]
                rewards = [r['reward_scores'][reward_models] for r in responses]
                samples.append(len(rewards))
                max_rewards.append(max(rewards))
                accuracies.append(self.get_best_response_accuracy(responses, len(rewards), reward_models))
            
            results['fixed'][n] = {
                'avg_samples': np.mean(samples),
                'avg_max_reward': np.mean(max_rewards),
                'std_max_reward': np.std(max_rewards),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
        
        # Test BCS sampling (t_min can be different from BOS's 3 samples)
        for width in ci_widths:
            samples = []
            max_rewards = []
            accuracies = []
            
            for qid in tqdm(question_ids, desc=f'BCS sampling (width={width})'):
                bcs_result = self.bcs_sampling(qid, dataset, reward_models,
                                            max_ci_width=width, t_min=t_min,
                                            t_max=t_max, alpha_ci=alpha_ci)
                samples.append(bcs_result['samples_used'])
                max_rewards.append(bcs_result['max_reward'])
                accuracies.append(bcs_result['accuracy'])
            
            results['bcs'][width] = {
                'avg_samples': np.mean(samples),
                'avg_max_reward': np.mean(max_rewards),
                'std_max_reward': np.std(max_rewards),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
        
        # Test BOS sampling (always starts with 3 samples)
        cost_values = [0.05, 0.1, 0.2]
        for cost in cost_values:
            # Regular BOS
            samples = []
            max_rewards = []
            accuracies = []
            
            for qid in tqdm(question_ids, desc=f'BOS sampling (cost={cost})'):
                bos_result = self.bos_sampling(qid, dataset, reward_models,
                                           cost_per_sample=cost, t_max=t_max,
                                           use_adaptive_ignore=False)  # Regular BOS
                samples.append(bos_result['samples_used'])
                max_rewards.append(bos_result['max_reward'])
                accuracies.append(bos_result['accuracy'])
            
            results['bos'][cost] = {
                'avg_samples': np.mean(samples),
                'avg_max_reward': np.mean(max_rewards),
                'std_max_reward': np.std(max_rewards),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
            
            # Adaptive BOS with tail filtering
            samples = []
            max_rewards = []
            accuracies = []
            
            for qid in tqdm(question_ids, desc=f'Adaptive BOS sampling (cost={cost})'):
                bos_result = self.bos_sampling(qid, dataset, reward_models,
                                           cost_per_sample=cost, t_max=t_max,
                                           use_adaptive_ignore=True, alpha=alpha)  # Adaptive BOS
                samples.append(bos_result['samples_used'])
                max_rewards.append(bos_result['max_reward'])
                accuracies.append(bos_result['accuracy'])
            
            results['bos_adaptive'][cost] = {
                'avg_samples': np.mean(samples),
                'avg_max_reward': np.mean(max_rewards),
                'std_max_reward': np.std(max_rewards),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
        
        return results

def main():
    # Example usage
    sampler = AdaptiveSampler('results')
    
    # Compare sampling methods
    results = sampler.compare_sampling_methods(
        dataset='xai_responses_amc23.json',
        reward_models='nemotron',
        ci_widths=[0.05, 0.1, 0.2],
        t_min=5,
        t_max=32,
        use_adaptive_ignore=True,
        alpha=0.03
    )
    
    # Save results
    with open('sampling_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nFixed Sampling Results:")
    for n, stats in results['fixed'].items():
        print(f"\nn = {n}:")
        print(f"Average samples used: {stats['avg_samples']:.2f}")
        print(f"Average max reward: {stats['avg_max_reward']:.4f}")
        print(f"Std of max reward: {stats['std_max_reward']:.4f}")
        print(f"Average accuracy: {stats['avg_accuracy']:.4f}")
        print(f"Std of accuracy: {stats['std_accuracy']:.4f}")
    
    print("\nBCS Sampling Results:")
    for width, stats in results['bcs'].items():
        print(f"\nCI width = {width}:")
        print(f"Average samples used: {stats['avg_samples']:.2f}")
        print(f"Average max reward: {stats['avg_max_reward']:.4f}")
        print(f"Std of max reward: {stats['std_max_reward']:.4f}")
        print(f"Average accuracy: {stats['avg_accuracy']:.4f}")
        print(f"Std of accuracy: {stats['std_accuracy']:.4f}")
    
    print("\nBOS Sampling Results:")
    for cost, stats in results['bos'].items():
        print(f"\nCost per sample = {cost}:")
        print(f"Average samples used: {stats['avg_samples']:.2f}")
        print(f"Average max reward: {stats['avg_max_reward']:.4f}")
        print(f"Std of max reward: {stats['std_max_reward']:.4f}")
        print(f"Average accuracy: {stats['avg_accuracy']:.4f}")
        print(f"Std of accuracy: {stats['std_accuracy']:.4f}")
    
    print("\nAdaptive BOS Sampling Results:")
    for cost, stats in results['bos_adaptive'].items():
        print(f"\nCost per sample = {cost}:")
        print(f"Average samples used: {stats['avg_samples']:.2f}")
        print(f"Average max reward: {stats['avg_max_reward']:.4f}")
        print(f"Std of max reward: {stats['std_max_reward']:.4f}")
        print(f"Average accuracy: {stats['avg_accuracy']:.4f}")
        print(f"Std of accuracy: {stats['std_accuracy']:.4f}")

if __name__ == "__main__":
    main() 