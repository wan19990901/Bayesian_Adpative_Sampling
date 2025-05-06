import json
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from llm_evaluator import LLMEvaluator
import math
from numba import njit
from h_index_construction import (
    _t_pdf, _norm_pdf, _t_cdf, _norm_cdf, _norm_ppf, _t_ppf,
    H_myopic_jit, h_index_full, h_index_value
)

# --- Bayesian Optimal Stopping (BOS) Helper Functions (Updated) ---
@njit
def update_parameters(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, use_adaptive_ignore=False, alpha=0.01):
    """
    Update the parameters z_k, mu_k, and sigma_k based on the new observation y.
    
    Args:
        use_adaptive_ignore: If True, uses adaptive ignore update strategy
        alpha: Threshold for adaptive ignore (default: 0.01)
    """
    if use_adaptive_ignore:
        return adaptive_ignore_update(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, alpha)
    
    # Original update logic
    z_k_plus = max(z_k, y)
    mu_k_plus = mu_k + (y - mu_k) / (nu0 + k + 1)
    L_k_plus = math.sqrt((1 - (1/(nu0 + k + 1))**2) / (2*alpha0 + k + 1))
    sigma_k_plus = L_k_plus * math.sqrt((2*alpha0 + k) * sigma_k**2 + (y - mu_k)**2)
    
    return z_k_plus, mu_k_plus, sigma_k_plus

@njit
def adaptive_ignore_update(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, alpha=0.025):
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
    L_k_plus = math.sqrt((1 - (1 / (nu0 + k + 1))**2) / (2 * alpha0 + k + 1))
    sigma_k_plus = L_k_plus * math.sqrt((2 * alpha0 + k) * sigma_k**2 + (y_adj - mu_k)**2)

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

# --- SamplingComparison Class ---
class SamplingComparison:
    def __init__(self, n_total: int, results_file: str = None):
        """
        Initialize the sampling comparison.

        Args:
            n_total: Maximum number of samples to consider
            results_file: Path to the evaluation results file (optional in debug mode)
        """
        # BOS Prior Parameters (can be tuned)
        self.alpha0 = -0.5
        self.nu0 = 0
        self.beta0 = 0.0
        self.mu0 = 0.0

        if results_file:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            
            # Get actual number of responses from the first question
            first_question = next(iter(self.results.get("detailed_results", [])), None)
            if not first_question or not first_question.get("responses"):
                raise ValueError("No responses found in the results file")
            
            actual_n_total = len(first_question["responses"])
            self.n_total = min(n_total, actual_n_total)
            if n_total > actual_n_total:
                print(f"Warning: Requested n_total ({n_total}) exceeds available samples ({actual_n_total}). Using {self.n_total} samples.")
        else:
            # In debug mode, just use the provided n_total
            self.n_total = n_total

        # Pre-compute h_matrix for dynamic sampling
        self.G = 100  # Grid size
        self.n = self.n_total  # Use n_total for consistency
        self.h_matrix, self.z_grid = self._compute_h_matrix()

    def _compute_h_matrix(self):
        """
        Compute the h-index matrix for dynamic sampling.
        Skip calculations for k < 3 due to prior specification issues.
        Rows 0 and 1 (k=0,1) will be zeros, row 2 (k=3) will be computed.
        """
        print("Starting h_matrix computation...")
        # Use the h_index_construction module to compute the matrix
        h_matrix, z_grid = h_index_full(
            recall=1,  # With recall
            mu_flag=0,  # Unknown mean
            sigma_flag=0,  # Unknown variance
            alpha0=self.alpha0,
            nu0=self.nu0,
            n=self.n,
            G=self.G
        )
        
        print("Note: H matrix values for k < 3 (indices 0,1) are set to 0 due to prior specification issues")
        return h_matrix, z_grid

    def _get_h_value(self, k: int, z_val: float) -> float:
        """
        Get the interpolated h(k, z) value from h_matrix.
        """
        return h_index_value(self.h_matrix, self.z_grid, k, z_val)

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

                # Update parameters with adaptive ignore enabled
                z_k, mu_k, sigma_k = update_parameters(
                    z_k, mu_k, sigma_k, new_reward, i,
                    self.alpha0, self.nu0, self.beta0, self.mu0,
                    use_adaptive_ignore=True, alpha=0.01  # Enable adaptive ignore with alpha=0.01
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

    def save_h_matrix(self, output_file: str):
        """
        Save the H matrix to a CSV file.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save h_matrix
        np.savetxt(output_file, self.h_matrix, delimiter=',')
        
        # Save z_grid in a separate file with '_z_grid' suffix
        z_grid_file = output_file.replace('.csv', '_z_grid.csv')
        np.savetxt(z_grid_file, self.z_grid, delimiter=',')
        
        print(f"H matrix saved to {output_file}")
        print(f"Z grid saved to {z_grid_file}")

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
    parser.add_argument("--results_file", type=str,
                      help="Path to evaluation results file (JSON). Required unless in debug mode.")
    parser.add_argument("--output_dir", type=str, default="results/",
                      help="Directory to save comparison results and plots (default: results/)")
    parser.add_argument("--cost_thresholds", type=float, nargs='+',
                      default=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
                      help="List of cost thresholds 'c' for both Dynamic and Greedy Sampling")
    parser.add_argument("--n_total", type=int, required=True,
                      help="Maximum number of samples to consider")
    parser.add_argument("--debug_mode", action="store_true",
                      help="Only compute and save H matrix")
    parser.add_argument("--h_matrix_output", type=str,
                      help="Output file for H matrix in CSV format")

    args = parser.parse_args()

    # --- Input Validation ---
    if not args.debug_mode and not args.results_file:
        print("Error: results_file is required unless in debug mode")
        return

    if args.debug_mode:
        print(f"Debug mode: Computing H matrix for n_total={args.n_total}")
    else:
        print(f"Loading results from: {args.results_file}")
    print(f"Saving comparison to: {args.output_dir}")
    print(f"Cost thresholds 'c' for both methods: {args.cost_thresholds}")
    print(f"Using maximum of {args.n_total} samples")

    try:
        comparison = SamplingComparison(n_total=args.n_total, results_file=args.results_file)
        
        if args.debug_mode:
            if not args.h_matrix_output:
                args.h_matrix_output = os.path.join(args.output_dir, f'h_matrix_{args.n_total}.csv')
            comparison.save_h_matrix(args.h_matrix_output)
            print("Debug mode: Only H matrix was computed and saved")
            return

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
