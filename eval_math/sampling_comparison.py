import json
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from llm_evaluator import LLMEvaluator
import math
from numba import njit

# --- Bayesian Optimal Stopping (BOS) Helper Functions (Updated) ---
@njit
def _t_pdf(x, df):
    """
    Compute the probability density function (PDF) of the t-distribution.
    """
    return (1 + x**2/df)**(-(df+1)/2) * math.gamma((df+1)/2) / (math.sqrt(df*math.pi) * math.gamma(df/2))

@njit
def _norm_pdf(x):
    """
    Compute the probability density function (PDF) of the normal distribution.
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@njit
def _t_cdf(x, df):
    """
    Compute the cumulative distribution function (CDF) of the t-distribution.
    """
    if x == 0:
        return 0.5
    elif x > 0:
        return 1 - 0.5 * (1 - math.erf(x/math.sqrt(2)))
    else:
        return 0.5 * (1 - math.erf(-x/math.sqrt(2)))

@njit
def _norm_cdf(x):
    """
    Compute the cumulative distribution function (CDF) of the normal distribution.
    """
    return 0.5 * (1 + math.erf(x/math.sqrt(2)))

@njit
def update_parameters(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0):
    """
    Update the parameters z_k, mu_k, and sigma_k based on the new observation y.
    """
    # Update z_k
    z_k_plus = max(z_k, y)
    
    # Update mu_k
    mu_k_plus = mu_k + (y - mu_k) / (nu0 + k + 1)
    
    # Update sigma_k
    if k < 3:  # k_0 = 3
        # For k < k_0, use the given formula
        nu_k = nu0 + k
        alpha_k = alpha0 + k/2
        beta_k = beta0 + (k * nu0 / (nu0 + k)) * ((mu_k - mu0)**2)
        sigma_k_plus = math.sqrt((1 + 1/nu_k) * (2*beta_k / (2*alpha_k)))
    else:
        # For k >= k_0, use the update formula
        L_k_plus = math.sqrt((1 - (1/(nu0 + k + 1))**2) / (2*alpha0 + k + 1))
        sigma_k_plus = L_k_plus * math.sqrt((2*alpha0 + k) * sigma_k**2 + (y - mu_k)**2)
    
    return z_k_plus, mu_k_plus, sigma_k_plus

@njit
def compute_initial_parameters(x_values, alpha0, nu0, beta0, mu0):
    """
    Compute initial parameters based on the first k_0 observations.
    """
    k = len(x_values)
    if k < 3:
        raise ValueError("Need at least 3 observations to compute initial parameters")
    
    # Compute mean
    x_bar = np.mean(x_values)
    
    # Compute mu_k
    mu_k = (nu0 * mu0 + k * x_bar) / (nu0 + k)
    
    # Compute beta_k
    sum_squared_diff = np.sum((x_values - x_bar)**2)
    beta_k = beta0 + sum_squared_diff + (k * nu0 / (nu0 + k)) * ((x_bar - mu0)**2)
    
    # Compute alpha_k
    alpha_k = alpha0 + k/2
    
    # Compute nu_k
    nu_k = nu0 + k
    
    # Compute sigma_k
    sigma_k = math.sqrt((1 + 1/nu_k) * (2*beta_k / (2*alpha_k)))
    
    # Compute z_k (best offer so far)
    z_k = np.max(x_values)
    
    return z_k, mu_k, sigma_k

@njit
def H_myopic_jit(recall, sigma_flag, z, k, alpha0):
    """
    Compute the myopic H-function value.
    """
    if recall == 1:
        if sigma_flag == 0:
            df = 2 * alpha0 + k
            return ((df + z**2) / (df - 1)) * _t_pdf(z, df) - z * (1 - _t_cdf(z, df))
        else:  # sigma known
            return _norm_pdf(z) - z * (1 - _norm_cdf(z))
    else:  # recall = 0
        return -z

# --- SamplingComparison Class ---
class SamplingComparison:
    def __init__(self, results_file: str):
        """
        Initialize the sampling comparison.

        Args:
            results_file: Path to the evaluation results file
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        # Assuming evaluator is not needed here anymore, or handle its init
        # self.evaluator = LLMEvaluator(use_nemotron=True, use_rise=False)

        # BOS Prior Parameters (can be tuned)
        self.alpha0 = 0.5
        self.nu0 = 1.0
        self.beta0 = 1.0
        self.mu0 = 0.0

        # Pre-compute h_matrix for dynamic sampling
        self.G = 200  # Grid size
        self.n = 30   # Max iterations
        self.h_matrix, self.z_grid = self._compute_h_matrix()

    def _compute_h_matrix(self):
        """
        Compute the h-index matrix for dynamic sampling.
        """
        # Build grids
        c = np.zeros(self.G+2)
        z = np.zeros(self.G+2)
        for j in range(self.G+2):
            c[j] = self.G * (0.85)**j  # rho = 0.85
            z[j] = 30 * ((1-0.75)*(2*j - self.G - 1)/(self.G-1) + 0.75*((2*j - self.G - 1)/(self.G-1))**3)
        c[self.G+1] = 0

        # Compute h_matrix
        h_matrix = np.zeros((self.n+1, self.G+2))
        for k in range(self.n-1, 0, -1):
            for j_z in range(self.G, 0, -1):
                h_val = H_myopic_jit(1, 0, z[j_z], k, self.alpha0)
                if k < self.n-1:
                    for j_u in range(self.G, 0, -1):
                        mu_u = 1 / (self.nu0 + k + 1)
                        L = math.sqrt((1 - mu_u**2) / (2*self.alpha0 + k + 1))
                        s = L * math.sqrt(2*self.alpha0 + k + z[j_u]**2)
                        z_new = (max(z[j_z], z[j_u]) - z[j_u]*mu_u) / s
                        if k == self.n-2:
                            H_u = H_myopic_jit(1, 0, z_new, k+1, self.alpha0)
                        else:
                            j_1 = self.G
                            while j_1 > 1 and z_new < z[j_1]:
                                j_1 -= 1
                            if j_1 == self.G:
                                j_1 = self.G-1
                            theta_z = (z_new - z[j_1]) / (z[j_1+1] - z[j_1])
                            H_u = (1-theta_z)*h_matrix[k+1,j_1] + theta_z*h_matrix[k+1,j_1+1]
                        density = _t_pdf(z[j_u], df=2*self.alpha0+k)
                        dz = (z[j_u+1] - z[j_u-1]) / 2
                        h_val += s * max(0, H_u) * density * dz
                h_matrix[k, j_z] = h_val

        return h_matrix, z

    def _get_h_value(self, k: int, z_val: float) -> float:
        """
        Get the interpolated h(k, z) value from h_matrix.
        """
        j = self.G
        while j > 0 and z_val < self.z_grid[j]:
            j -= 1
        if j == self.G:
            j = self.G-1

        if (self.z_grid[j+1] - self.z_grid[j]) == 0:
            theta = 0
        else:
            theta = (z_val - self.z_grid[j]) / (self.z_grid[j+1] - self.z_grid[j])
        
        return (1-theta) * self.h_matrix[k, j] + theta * self.h_matrix[k, j+1]

    def _run_bos_sampling(self, responses: List[Dict], cost_threshold: float, use_myopic_h: bool) -> Dict:
        """
        Internal helper to run BOS-based sampling (dynamic or greedy).

        Args:
            responses: List of responses for a single question.
            cost_threshold: The cost 'c' for stopping.
            use_myopic_h: If True, uses H_myopic_jit for stopping condition (greedy).
                          If False, uses a more conservative stopping condition for dynamic.

        Returns:
            Dictionary with accuracy, avg_reward, samples_used.
        """
        if not responses:
            return {"accuracy": 0.0, "avg_reward": 0.0, "samples_used": 0}

        rewards = np.array([r["reward_scores"]["nemotron"] for r in responses], dtype=np.float64)
        n_total = len(rewards)

        if n_total == 0:
             return {"accuracy": 0.0, "avg_reward": 0.0, "samples_used": 0}

        # Fallback for fewer than 3 samples: use Best-of-N among available
        if n_total < 3:
            best_idx = np.argmax(rewards)
            best_response = responses[best_idx]
            return {
                "accuracy": 1.0 if best_response["is_correct"] else 0.0,
                "avg_reward": best_response["reward_scores"]["nemotron"],
                "samples_used": n_total
            }

        # Initialize BOS with first 3 samples
        initial_scores = rewards[:3]
        z_k, mu_k, sigma_k = compute_initial_parameters(
            initial_scores, self.alpha0, self.nu0, self.beta0, self.mu0
        )

        samples_used = 3
        # Track the response corresponding to the best reward z_k
        current_best_response_idx = np.argmax(initial_scores)
        best_response = responses[current_best_response_idx]

        for i in range(3, n_total):
            if sigma_k <= 1e-9: # Avoid division by zero if std dev is tiny
                break # Stop if variance collapses

            z_val = (z_k - mu_k) / sigma_k
            c_value = cost_threshold / sigma_k

            # Calculate h-value (stopping threshold)
            if use_myopic_h:
                # Greedy uses myopic H value directly
                h_val = H_myopic_jit(recall=1, sigma_flag=0, z=z_val, k=i, alpha0=self.alpha0)
            else:
                # Dynamic uses full DP solution
                h_val = self._get_h_value(k=i, z_val=z_val)

            # Decision: continue or stop
            if h_val <= c_value:
                stop = True
                break
            else:
                # Continue: process next sample
                new_reward = rewards[i]

                # Update parameters (pass sigma_k which is std dev)
                z_k, mu_k, sigma_k = update_parameters(
                    z_k, mu_k, sigma_k, new_reward, i, # k is the index (number samples before this one)
                    self.alpha0, self.nu0, self.beta0, self.mu0
                )

                samples_used += 1
                # Update best_response if the new sample gave the max reward z_k
                if new_reward == z_k:
                     best_response = responses[i]

        # Return results based on the best response found when stopping
        return {
            "accuracy": 1.0 if best_response["is_correct"] else 0.0,
            "avg_reward": best_response["reward_scores"]["nemotron"], # Reward of the selected best
            "samples_used": samples_used
        }

    def dynamic_sampling(self, responses: List[Dict], confidence_threshold: float) -> Dict:
        """
        Use Dynamic Bayesian Optimal Stopping (approximated with Myopic H).
        Confidence threshold acts as the cost 'c'.
        """
        return self._run_bos_sampling(responses, cost_threshold=confidence_threshold, use_myopic_h=False)

    def greedy_dynamic_sampling(self, responses: List[Dict], cost_threshold: float) -> Dict:
        """
        Use Greedy Bayesian Optimal Stopping (Myopic H).
        Improvement threshold acts as the cost 'c'.
        """
        return self._run_bos_sampling(responses, cost_threshold=cost_threshold, use_myopic_h=True)

    # --- Other Sampling Methods (Unchanged) ---
    def random_sampling(self, responses: List[Dict]) -> Dict:
        """
        Randomly select 1 sample.
        """
        if not responses:
            return {"accuracy": 0.0, "avg_reward": 0.0, "samples_used": 0}

        selected_idx = np.random.choice(len(responses))
        selected = responses[selected_idx]

        return {
            "accuracy": 1.0 if selected["is_correct"] else 0.0,
            "avg_reward": selected["reward_scores"]["nemotron"],
            "samples_used": 1
        }

    def self_consistency(self, responses: List[Dict]) -> Dict:
        """
        Use self-consistency (majority voting) to select the answer.
        Reports reward of the *first* response matching the majority answer.
        """
        if not responses:
            return {"accuracy": 0.0, "avg_reward": 0.0, "samples_used": 0}

        answers = [r.get("final_answer") for r in responses if r.get("final_answer") is not None]
        if not answers:
            # If no responses have final answers, fallback (e.g., return stats of first response)
             if responses:
                 first_resp = responses[0]
                 return {"accuracy": 1.0 if first_resp["is_correct"] else 0.0,
                         "avg_reward": first_resp["reward_scores"]["nemotron"],
                         "samples_used": len(responses)}
             else:
                 return {"accuracy": 0.0, "avg_reward": 0.0, "samples_used": 0}


        unique_answers, counts = np.unique(answers, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_answer = unique_answers[majority_idx]

        # Find the first response with the majority answer to report its stats
        majority_response = None
        for r in responses:
            if r.get("final_answer") == majority_answer:
                 majority_response = r
                 break # Found the first one

        # Handle case where majority answer was found but no corresponding response exists (shouldn't happen)
        if majority_response is None:
             # Fallback: use stats of the first response overall
             majority_response = responses[0]


        return {
            "accuracy": 1.0 if majority_response["is_correct"] else 0.0,
            "avg_reward": majority_response["reward_scores"]["nemotron"],
            "samples_used": len(responses) # SC always uses all samples
        }

    def best_of_n(self, responses: List[Dict]) -> Dict:
        """
        Use all samples and pick the best one based on reward.
        """
        if not responses:
            return {"accuracy": 0.0, "avg_reward": 0.0, "samples_used": 0}

        rewards = [r["reward_scores"]["nemotron"] for r in responses]
        if not rewards: # Handle case where responses exist but rewards are missing
             return {"accuracy": 0.0, "avg_reward": 0.0, "samples_used": len(responses)}

        best_idx = np.argmax(rewards)
        best_response = responses[best_idx]

        return {
            "accuracy": 1.0 if best_response["is_correct"] else 0.0,
            "avg_reward": best_response["reward_scores"]["nemotron"],
            "samples_used": len(responses) # Best-of-N uses all samples
        }

    # --- Comparison Framework (Updated parameter names) ---
    def compare_methods(self, cost_thresholds: List[float] = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]) -> Dict:
        """
        Compare different sampling methods.

        Args:
            cost_thresholds: List of cost thresholds 'c' for both dynamic and greedy sampling.

        Returns:
            Dictionary containing results for each method.
        """
        results = {
            "random": {"accuracy": [], "avg_reward": [], "samples_used": []},
            "self_consistency": {"accuracy": [], "avg_reward": [], "samples_used": []},
            "best_of_n": {"accuracy": [], "avg_reward": [], "samples_used": []},
            "dynamic": {t: {"accuracy": [], "avg_reward": [], "samples_used": []} for t in cost_thresholds},
            "greedy": {t: {"accuracy": [], "avg_reward": [], "samples_used": []} for t in cost_thresholds}
        }

        # Process each question
        for question_data in tqdm(self.results.get("detailed_results", []), desc="Processing questions"):
            responses = question_data.get("responses", [])
            if not responses: continue # Skip if no responses for this question

            # Random sampling
            res = self.random_sampling(responses)
            results["random"]["accuracy"].append(res["accuracy"])
            results["random"]["avg_reward"].append(res["avg_reward"])
            results["random"]["samples_used"].append(res["samples_used"])

            # Self-consistency
            res = self.self_consistency(responses)
            results["self_consistency"]["accuracy"].append(res["accuracy"])
            results["self_consistency"]["avg_reward"].append(res["avg_reward"])
            results["self_consistency"]["samples_used"].append(res["samples_used"])

            # Best-of-N
            res = self.best_of_n(responses)
            results["best_of_n"]["accuracy"].append(res["accuracy"])
            results["best_of_n"]["avg_reward"].append(res["avg_reward"])
            results["best_of_n"]["samples_used"].append(res["samples_used"])

            # Dynamic sampling (BOS Approx)
            for t in cost_thresholds:
                res = self.dynamic_sampling(responses, t)
                results["dynamic"][t]["accuracy"].append(res["accuracy"])
                results["dynamic"][t]["avg_reward"].append(res["avg_reward"])
                results["dynamic"][t]["samples_used"].append(res["samples_used"])

            # Greedy dynamic sampling (Myopic BOS)
            for t in cost_thresholds:
                res = self.greedy_dynamic_sampling(responses, t)
                results["greedy"][t]["accuracy"].append(res["accuracy"])
                results["greedy"][t]["avg_reward"].append(res["avg_reward"])
                results["greedy"][t]["samples_used"].append(res["samples_used"])

        # Calculate averages safely (handle empty lists)
        for method, method_results in results.items():
            if method in ["dynamic", "greedy"]:
                for t, threshold_results in method_results.items():
                    results[method][t]["accuracy"] = np.mean(threshold_results["accuracy"]) if threshold_results["accuracy"] else 0.0
                    results[method][t]["avg_reward"] = np.mean(threshold_results["avg_reward"]) if threshold_results["avg_reward"] else 0.0
                    results[method][t]["samples_used"] = np.mean(threshold_results["samples_used"]) if threshold_results["samples_used"] else 0.0
            else:
                results[method]["accuracy"] = np.mean(method_results["accuracy"]) if method_results["accuracy"] else 0.0
                results[method]["avg_reward"] = np.mean(method_results["avg_reward"]) if method_results["avg_reward"] else 0.0
                results[method]["samples_used"] = np.mean(method_results["samples_used"]) if method_results["samples_used"] else 0.0

        return results

    # --- Plotting and Saving (Adjust labels/keys) ---
    def plot_results(self, results: Dict, output_dir: str):
        """
        Plot comparison results.
        """
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style

        # Plot accuracy vs samples used
        plt.figure(figsize=(12, 7))

        # Static methods (points)
        plt.scatter(results["random"]["samples_used"], results["random"]["accuracy"],
                   label=f'Random (1 sample)', color='blue', s=100, marker='s', zorder=5)
        plt.scatter(results["best_of_n"]["samples_used"], results["best_of_n"]["accuracy"],
                   label=f'Best-of-N ({results["best_of_n"]["samples_used"]:.1f} samples)', color='green', s=100, marker='^', zorder=5)
        plt.scatter(results["self_consistency"]["samples_used"], results["self_consistency"]["accuracy"],
                   label=f'Self-Consistency ({results["self_consistency"]["samples_used"]:.1f} samples)', color='red', s=100, marker='o', zorder=5)

        # Dynamic methods (lines)
        dynamic_samples = [results["dynamic"][t]["samples_used"] for t in sorted(results["dynamic"].keys())]
        dynamic_accuracy = [results["dynamic"][t]["accuracy"] for t in sorted(results["dynamic"].keys())]
        if dynamic_samples:
             plt.plot(dynamic_samples, dynamic_accuracy, 'o-', label='Dynamic Sampling (BOS Approx)', color='purple', linewidth=2)

        greedy_samples = [results["greedy"][t]["samples_used"] for t in sorted(results["greedy"].keys())]
        greedy_accuracy = [results["greedy"][t]["accuracy"] for t in sorted(results["greedy"].keys())]
        if greedy_samples:
             plt.plot(greedy_samples, greedy_accuracy, 's--', label='Greedy Sampling (Myopic BOS)', color='orange', linewidth=2)


        plt.xlabel('Average Samples Used', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Sampling Method Performance: Accuracy vs. Samples Used', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_samples.png'), dpi=300)
        plt.close()

        # Plot reward vs samples used
        plt.figure(figsize=(12, 7))

        # Static methods (points)
        plt.scatter(results["random"]["samples_used"], results["random"]["avg_reward"],
                   label=f'Random (1 sample)', color='blue', s=100, marker='s', zorder=5)
        plt.scatter(results["best_of_n"]["samples_used"], results["best_of_n"]["avg_reward"],
                   label=f'Best-of-N ({results["best_of_n"]["samples_used"]:.1f} samples)', color='green', s=100, marker='^', zorder=5)
        plt.scatter(results["self_consistency"]["samples_used"], results["self_consistency"]["avg_reward"],
                   label=f'Self-Consistency ({results["self_consistency"]["samples_used"]:.1f} samples)', color='red', s=100, marker='o', zorder=5)

        # Dynamic methods (lines)
        dynamic_rewards = [results["dynamic"][t]["avg_reward"] for t in sorted(results["dynamic"].keys())]
        if dynamic_samples: # Reuse samples from accuracy plot
             plt.plot(dynamic_samples, dynamic_rewards, 'o-', label='Dynamic Sampling (BOS Approx)', color='purple', linewidth=2)

        greedy_rewards = [results["greedy"][t]["avg_reward"] for t in sorted(results["greedy"].keys())]
        if greedy_samples: # Reuse samples from accuracy plot
             plt.plot(greedy_samples, greedy_rewards, 's--', label='Greedy Sampling (Myopic BOS)', color='orange', linewidth=2)

        plt.xlabel('Average Samples Used', fontsize=12)
        plt.ylabel('Average Nemotron Reward', fontsize=12)
        plt.title('Sampling Method Performance: Reward vs. Samples Used', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_vs_samples.png'), dpi=300)
        plt.close()

    def save_results(self, results: Dict, output_file: str):
        """
        Save results to a JSON file.
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = json.loads(json.dumps(results, cls=NpEncoder))
            json.dump(serializable_results, f, indent=2)

# Helper class to encode numpy types for JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare different sampling methods using BOS")
    parser.add_argument("--results_file", type=str, required=True,
                      help="Path to evaluation results file (JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save comparison results and plots")
    parser.add_argument("--cost_thresholds", type=float, nargs='+',
                      default=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
                      help="List of cost thresholds 'c' for both Dynamic and Greedy Sampling")

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found at {args.results_file}")
        return
    if not args.output_dir:
         print("Error: Output directory must be specified.")
         return

    print(f"Loading results from: {args.results_file}")
    print(f"Saving comparison to: {args.output_dir}")
    print(f"Cost thresholds 'c' for both methods: {args.cost_thresholds}")


    try:
        comparison = SamplingComparison(args.results_file)
        results = comparison.compare_methods(args.cost_thresholds)

        # Save and plot results
        results_save_path = os.path.join(args.output_dir, 'comparison_results.json')
        comparison.save_results(results, results_save_path)
        print(f"Comparison results saved to {results_save_path}")

        comparison.plot_results(results, args.output_dir)
        print(f"Plots saved to {args.output_dir}")

        # Print summary
        print("\n=== Results Summary ===")
        print(f"\nRandom Sampling (1 sample):")
        print(f"  Accuracy: {results['random']['accuracy']:.3f}")
        print(f"  Avg Reward: {results['random']['avg_reward']:.3f}")
        print(f"  Avg Samples: {results['random']['samples_used']:.1f}")

        print(f"\nBest-of-N (all samples):")
        print(f"  Accuracy: {results['best_of_n']['accuracy']:.3f}")
        print(f"  Avg Reward: {results['best_of_n']['avg_reward']:.3f}")
        print(f"  Avg Samples: {results['best_of_n']['samples_used']:.1f}")

        print(f"\nSelf-Consistency:")
        print(f"  Accuracy: {results['self_consistency']['accuracy']:.3f}")
        print(f"  Avg Reward: {results['self_consistency']['avg_reward']:.3f}")
        print(f"  Avg Samples: {results['self_consistency']['samples_used']:.1f}")

        print(f"\nDynamic Sampling (BOS Approx):")
        for t in sorted(results["dynamic"].keys()):
            print(f"  Cost Threshold c={t}:")
            print(f"    Accuracy: {results['dynamic'][t]['accuracy']:.3f}")
            print(f"    Avg Reward: {results['dynamic'][t]['avg_reward']:.3f}")
            print(f"    Avg Samples: {results['dynamic'][t]['samples_used']:.2f}") # More precision for samples

        print(f"\nGreedy Sampling (Myopic BOS):")
        for t in sorted(results["greedy"].keys()):
            print(f"  Cost Threshold c={t}:") # Changed label
            print(f"    Accuracy: {results['greedy'][t]['accuracy']:.3f}")
            print(f"    Avg Reward: {results['greedy'][t]['avg_reward']:.3f}")
            print(f"    Avg Samples: {results['greedy'][t]['samples_used']:.2f}") # More precision for samples

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.results_file}")
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {args.results_file}. Check file format.")
    except KeyError as e:
         print(f"Error: Missing expected key in results data: {e}. Ensure 'detailed_results', 'responses', 'reward_scores', 'nemotron', 'is_correct', 'final_answer' exist.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         # Optionally re-raise for debugging: raise


if __name__ == "__main__":
    main() 