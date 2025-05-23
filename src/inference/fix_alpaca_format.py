import json
import os
from dotenv import load_dotenv
import numpy as np
import math
from numba import njit
from h_index_construction import _t_pdf, _norm_pdf, _t_cdf, _norm_cdf, _norm_ppf, _t_ppf

# Load environment variables from parent directory
load_dotenv('../.env')

# --- BOS Helper Functions from sampling_comparison.py ---
@njit
def update_parameters(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, use_adaptive_ignore=False, alpha=0.01):
    """
    Update the parameters z_k, mu_k, and sigma_k based on the new observation y.
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
def adaptive_ignore_update(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0, alpha=0.01):
    """
    Update parameters with adaptive ignoring of extreme low observations.
    """
    df_k = 2 * alpha0 + k
    threshold_k = mu_k + sigma_k * _t_ppf(alpha, df_k)

    # Ignore update if below threshold, use mu_k as observation
    y_adj = y if y >= threshold_k else mu_k
    
    # Update best offer
    z_k_plus = max(z_k, y)

    # Update mu and sigma normally using y_adj
    mu_k_plus = mu_k + (y_adj - mu_k) / (nu0 + k + 1)
    L_k_plus = math.sqrt((1 - (1 / (nu0 + k + 1))**2) / (2 * alpha0 + k + 1))
    sigma_k_plus = L_k_plus * math.sqrt((2 * alpha0 + k) * sigma_k**2 + (y_adj - mu_k)**2)

    return z_k_plus, mu_k_plus, sigma_k_plus

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

def get_dynamic_stopping_index(scores, cost_threshold=0.1):
    """
    Use BOS to determine the optimal stopping index for a sequence of scores.
    
    Args:
        scores: List of reward scores
        cost_threshold: Cost threshold for stopping (default: 0.1)
        
    Returns:
        Index of the best response according to BOS
    """
    if len(scores) < 3:
        return np.argmax(scores) if scores else 0
    
    # BOS Prior Parameters
    alpha0 = -0.5
    nu0 = 0
    beta0 = 0.0
    mu0 = 0.0
    
    # Initialize with first 3 samples
    initial_scores = scores[:3]
    z_k, mu_k, sigma_k = compute_initial_parameters(
        initial_scores, alpha0, nu0, beta0, mu0
    )
    
    best_idx = np.argmax(initial_scores)
    current_best_score = initial_scores[best_idx]
    
    # Process remaining samples
    for i in range(3, len(scores)):
        if sigma_k <= 1e-9:  # Avoid division by zero
            break
            
        new_score = scores[i]
        
        # Update parameters with adaptive ignore
        z_k, mu_k, sigma_k = update_parameters(
            z_k, mu_k, sigma_k, new_score, i,
            alpha0, nu0, beta0, mu0,
            use_adaptive_ignore=True, alpha=0.01
        )
        
        # Update best index if new score is better
        if new_score > current_best_score:
            best_idx = i
            current_best_score = new_score
    
    return best_idx

def fix_alpaca_format(input_file, output_file, scores_file=None, use_dynamic=True, cost_threshold=0.1):
    """
    Fix the format of Alpaca responses with flexible selection strategy.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        scores_file: Path to scores file for dynamic selection
        use_dynamic: Whether to use dynamic BOS selection
        cost_threshold: Cost threshold for BOS (if use_dynamic is True)
    """
    # Read the input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Read scores if provided
    scores_data = None
    if scores_file:
        with open(scores_file, 'r') as f:
            scores_data = json.load(f)
    
    # Convert the format
    fixed_data = []
    for idx, item in enumerate(data):
        if not isinstance(item['responses'], list) or len(item['responses']) == 0:
            print(f"Warning: Unexpected format for item {idx}")
            continue
            
        # Get response index based on selection strategy
        if use_dynamic and scores_data and str(idx + 1) in scores_data:
            # Get scores for this question
            question_scores = scores_data[str(idx + 1)]
            # Use BOS to select best index
            response_idx = int(get_dynamic_stopping_index(question_scores, cost_threshold))  # Convert to Python int
            selection_method = 'dynamic'
        elif scores_data and str(idx + 1) in scores_data:
            # Get scores for this question
            question_scores = scores_data[str(idx + 1)]
            # Select the highest scoring response
            response_idx = int(np.argmax(question_scores))  # Convert to Python int
            selection_method = 'best'
        else:
            # Default to first response if no scores
            response_idx = 0
            selection_method = 'first'
        
        # Select the response
        if isinstance(item['responses'][0], list):
            if response_idx < len(item['responses'][0]):
                fixed_item = {
                    'instruction': item['instruction'],
                    'output': item['responses'][0][response_idx],
                    'generator': item['generator'],
                    'selected_index': response_idx,
                    'total_responses': int(len(item['responses'][0])),  # Convert to Python int
                    'selection_method': selection_method
                }
            else:
                print(f"Warning: Response index {response_idx} out of range for item {idx}")
                continue
        else:
            if response_idx < len(item['responses']):
                fixed_item = {
                    'instruction': item['instruction'],
                    'output': item['responses'][response_idx],
                    'generator': item['generator'],
                    'selected_index': response_idx,
                    'total_responses': int(len(item['responses'])),  # Convert to Python int
                    'selection_method': selection_method
                }
            else:
                print(f"Warning: Response index {response_idx} out of range for item {idx}")
                continue
            
        fixed_data.append(fixed_item)
    
    # Write the fixed data
    with open(output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"Fixed format saved to {output_file}")

if __name__ == "__main__":
    # Example usage with dynamic sampling
    input_file = "results/alpaca_eval/grok3_responses_alpaca.json"
    output_file = "results/alpaca_eval/grok3_responses_alpaca_best.json"
    scores_file = "results/alpaca_eval/skywork_reward_grok3_alpaca.json"
    
    # Use dynamic BOS selection with cost threshold 0.1
    fix_alpaca_format(input_file, output_file, scores_file, use_dynamic=False)