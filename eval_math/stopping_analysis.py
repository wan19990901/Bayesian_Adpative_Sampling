import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
from h_index_construction import (
    _t_pdf, _norm_pdf, _t_cdf, _norm_cdf, _norm_ppf, _t_ppf,
    H_myopic_jit, h_index_full, h_index_value
)

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
    sigma_k = np.sqrt((1 + 1/nu_k) * (2*beta_k / (2*alpha_k)))
    
    # Compute z_k (best offer so far)
    z_k = np.max(x_values)
    
    return z_k, mu_k, sigma_k

def update_parameters(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, use_adaptive_ignore=False, alpha=0.01):
    """
    Update the parameters z_k, mu_k, and sigma_k based on the new observation y.
    """
    if use_adaptive_ignore:
        return adaptive_ignore_update(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, alpha)
    
    # Original update logic
    z_k_plus = max(z_k, y)
    mu_k_plus = mu_k + (y - mu_k) / (nu0 + k + 1)
    L_k_plus = np.sqrt((1 - (1/(nu0 + k + 1))**2) / (2*alpha0 + k + 1))
    sigma_k_plus = L_k_plus * np.sqrt((2*alpha0 + k) * sigma_k**2 + (y - mu_k)**2)
    
    return z_k_plus, mu_k_plus, sigma_k_plus

def adaptive_ignore_update(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, alpha=0.01):
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
    L_k_plus = np.sqrt((1 - (1 / (nu0 + k + 1))**2) / (2 * alpha0 + k + 1))
    sigma_k_plus = L_k_plus * np.sqrt((2 * alpha0 + k) * sigma_k**2 + (y_adj - mu_k)**2)

    return z_k_plus, mu_k_plus, sigma_k_plus

def analyze_stopping_points(responses_file: str, costs: list, output_dir: str = "results/stopping_analysis"):
    """
    Analyze how stopping at different points affects the value function with multiple costs.
    Compares adaptive and non-adaptive parameter updates.
    """
    # Load results
    with open(responses_file, 'r') as f:
        results = json.load(f)
    
    # Initialize arrays to store metrics
    n_samples = 32  # Maximum number of samples to consider
    avg_rewards = np.zeros(n_samples)
    avg_correct = np.zeros(n_samples)
    value_functions = {cost: np.zeros(n_samples) for cost in costs}
    sample_counts = np.zeros(n_samples)
    
    # BOS parameters
    alpha0 = -0.5
    nu0 = 0
    beta0 = 0.0
    mu0 = 0.0
    
    # Initialize arrays for adaptive and non-adaptive results
    adaptive_results = {cost: {"stopping_points": [], "values": []} for cost in costs}
    non_adaptive_results = {cost: {"stopping_points": [], "values": []} for cost in costs}
    
    # Process each question
    for question_data in tqdm(results.get("detailed_results", []), desc="Analyzing stopping points"):
        responses = question_data.get("responses", [])
        if not responses:
            continue
            
        # Get rewards and correctness
        rewards = [r["reward_scores"]["nemotron"] for r in responses]
        correct = [r["is_correct"] for r in responses]
        
        # For each possible stopping point
        for k in range(min(n_samples, len(responses))):
            # Get best reward up to this point
            best_reward = max(rewards[:k+1])
            best_idx = rewards[:k+1].index(best_reward)
            is_correct = correct[best_idx]
            
            # Calculate value function for each cost
            for cost in costs:
                value = best_reward - (k+1) * cost
                value_functions[cost][k] += value
            
            # Update other metrics
            avg_rewards[k] += best_reward
            avg_correct[k] += is_correct
            sample_counts[k] += 1
            
            # Run BOS with both adaptive and non-adaptive updates
            if k >= 2:  # Need at least 3 samples for BOS
                # Initialize parameters
                initial_scores = rewards[:3]
                z_k, mu_k, sigma_k = compute_initial_parameters(
                    initial_scores, alpha0, nu0, beta0, mu0
                )
                
                # Process remaining samples
                for i in range(3, k+1):
                    if sigma_k <= 1e-9:  # Avoid division by zero
                        break
                        
                    # Calculate stopping threshold
                    z_val = (z_k - mu_k) / sigma_k
                    c_value = cost / sigma_k
                    h_val = H_myopic_jit(recall=1, sigma_flag=0, z=z_val, k=i, alpha0=alpha0)
                    
                    # Update parameters (both adaptive and non-adaptive)
                    z_k_adaptive, mu_k_adaptive, sigma_k_adaptive = update_parameters(
                        z_k, mu_k, sigma_k, rewards[i], i,
                        alpha0, nu0, beta0, mu0,
                        use_adaptive_ignore=True, alpha=0.01
                    )
                    
                    z_k_non_adaptive, mu_k_non_adaptive, sigma_k_non_adaptive = update_parameters(
                        z_k, mu_k, sigma_k, rewards[i], i,
                        alpha0, nu0, beta0, mu0,
                        use_adaptive_ignore=False
                    )
                    
                    # Record stopping points
                    for cost_val in costs:
                        if h_val <= cost_val / sigma_k_adaptive:
                            adaptive_results[cost_val]["stopping_points"].append(i+1)
                            adaptive_results[cost_val]["values"].append(best_reward - (i+1) * cost_val)
                        if h_val <= cost_val / sigma_k_non_adaptive:
                            non_adaptive_results[cost_val]["stopping_points"].append(i+1)
                            non_adaptive_results[cost_val]["values"].append(best_reward - (i+1) * cost_val)
    
    # Calculate averages
    for k in range(n_samples):
        if sample_counts[k] > 0:
            avg_rewards[k] /= sample_counts[k]
            avg_correct[k] /= sample_counts[k]
            for cost in costs:
                value_functions[cost][k] /= sample_counts[k]
    
    # Create a single figure with two subplots
    plt.figure(figsize=(20, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Left subplot: All costs
    plt.subplot(1, 2, 1)
    # Define a more distinctive color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, cost in enumerate(costs):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(range(1, n_samples+1), value_functions[cost], f'{marker}-', 
                label=f'Cost = {cost}', linewidth=3, markersize=6, color=color)
        # Mark optimal point for each cost
        optimal_idx = np.argmax(value_functions[cost])
        optimal_value = value_functions[cost][optimal_idx]
        plt.plot(optimal_idx + 1, optimal_value, '*', markersize=12, color=color,
                 label=f'Optimal (stop={optimal_idx + 1}, value={optimal_value:.3f})')
    
    plt.xlabel('Number of Samples', fontsize=16)
    plt.ylabel('Value Function\n(Reward - Samples × Cost)', fontsize=16)
    plt.title('Value Function vs. Number of Samples\nfor Different Costs', fontsize=18, pad=20)
    plt.legend(fontsize=14, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Right subplot: Cost = 0.075
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_samples+1), value_functions[0.075], 'o-', color='purple', linewidth=3,
             label='Value Function', markersize=6)
    
    # Add vertical lines for optimal stopping points
    adaptive_stops = adaptive_results[0.075]["stopping_points"]
    non_adaptive_stops = non_adaptive_results[0.075]["stopping_points"]
    
    mean_adaptive = np.mean(adaptive_stops) if adaptive_stops else 0
    mean_non_adaptive = np.mean(non_adaptive_stops) if non_adaptive_stops else 0
    
    plt.axvline(mean_adaptive, color='blue', linestyle='--', linewidth=2.5,
                label=f'Adaptive (stop={mean_adaptive:.1f})')
    plt.axvline(mean_non_adaptive, color='orange', linestyle='--', linewidth=2.5,
                label=f'Non-Adaptive (stop={mean_non_adaptive:.1f})')
    
    # Add points at the optimal stopping points
    adaptive_value = value_functions[0.075][int(mean_adaptive)-1]
    non_adaptive_value = value_functions[0.075][int(mean_non_adaptive)-1]
    
    plt.plot(mean_adaptive, adaptive_value, 'bo', markersize=12,
             label=f'Adaptive (value={adaptive_value:.3f})')
    plt.plot(mean_non_adaptive, non_adaptive_value, 'ro', markersize=12,
             label=f'Non-Adaptive (value={non_adaptive_value:.3f})')
    
    # Mark optimal point
    optimal_idx = np.argmax(value_functions[0.075])
    optimal_value = value_functions[0.075][optimal_idx]
    plt.plot(optimal_idx + 1, optimal_value, 'k*', markersize=18,
             label=f'Optimal (stop={optimal_idx + 1}, value={optimal_value:.3f})')
    
    plt.xlabel('Number of Samples', fontsize=16)
    plt.ylabel('Value Function\n(Reward - Samples × 0.075)', fontsize=16)
    plt.title('Value Function vs. Number of Samples\n(Cost = 0.075)', fontsize=18, pad=20)
    plt.legend(fontsize=14, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=2, 
              bbox_transform=plt.gca().transAxes, handletextpad=0.5, columnspacing=1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'value_vs_stopping_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy vs stopping point
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, n_samples+1), avg_correct, 'o-', color='blue', linewidth=2)
    plt.xlabel('Stopping Point (Number of Samples)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Stopping Point', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_stopping.png'), dpi=300)
    plt.close()
    
    # Plot reward vs stopping point
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, n_samples+1), avg_rewards, 'o-', color='green', linewidth=2)
    plt.xlabel('Stopping Point (Number of Samples)', fontsize=12)
    plt.ylabel('Average Best Reward', fontsize=12)
    plt.title('Best Reward vs Stopping Point', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_vs_stopping.png'), dpi=300)
    plt.close()
    
    # Print summary for each cost
    print("\n=== Stopping Point Analysis Summary ===")
    for cost in costs:
        best_stopping = np.argmax(value_functions[cost]) + 1
        print(f"\nCost = {cost}:")
        print(f"  Best stopping point: {best_stopping} samples")
        print(f"  Value at best stopping point: {value_functions[cost][best_stopping-1]:.3f}")
        print(f"  Accuracy at best stopping point: {avg_correct[best_stopping-1]:.3f}")
        print(f"  Average reward at best stopping point: {avg_rewards[best_stopping-1]:.3f}")
        
        # Print adaptive vs non-adaptive comparison
        if adaptive_results[cost]["stopping_points"]:
            avg_adaptive = np.mean(adaptive_results[cost]["stopping_points"])
            avg_non_adaptive = np.mean(non_adaptive_results[cost]["stopping_points"])
            print(f"  Average stopping point (Adaptive): {avg_adaptive:.1f} samples")
            print(f"  Average stopping point (Non-Adaptive): {avg_non_adaptive:.1f} samples")
    
    return {
        "value_functions": value_functions,
        "accuracies": avg_correct,
        "rewards": avg_rewards,
        "adaptive_results": adaptive_results,
        "non_adaptive_results": non_adaptive_results
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze stopping points with different costs")
    parser.add_argument("--results_file", type=str, required=True,
                      help="Path to evaluation results file (JSON)")
    parser.add_argument("--output_dir", type=str, default="results/stopping_analysis",
                      help="Directory to save analysis plots")
    parser.add_argument("--costs", type=float, nargs='+',
                      default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
                      help="List of costs to analyze")

    args = parser.parse_args()

    print(f"Loading results from: {args.results_file}")
    print(f"Saving analysis to: {args.output_dir}")
    print(f"Analyzing costs: {args.costs}")

    try:
        analyze_stopping_points(args.results_file, args.costs, args.output_dir)
        print(f"\nAnalysis complete. Plots saved to {args.output_dir}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.results_file}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.results_file}. Check file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 