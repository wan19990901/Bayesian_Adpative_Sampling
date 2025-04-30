import json
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Dict, List
from tqdm import tqdm

class RewardAnalyzer:
    def __init__(self, results_dir: str):
        """
        Initialize the reward analyzer with the directory containing evaluation results.
        
        Args:
            results_dir: Directory containing the evaluation results and intermediate files
        """
        self.results_dir = results_dir
        self.results = {}
        self.load_results()
        
    def load_results(self):
        """Load all evaluation results from the directory."""
        # Find all intermediate directories
        for dir_name in os.listdir(self.results_dir):
            if dir_name.endswith('_intermediate'):
                dataset_name = dir_name.split('_')[0]
                reward_models = '_'.join(dir_name.split('_')[1:-1])
                
                if dataset_name not in self.results:
                    self.results[dataset_name] = {}
                
                self.results[dataset_name][reward_models] = {
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
                            
                        self.results[dataset_name][reward_models]['questions'][question_id] = question_data
                        
                        # Extract rewards for each model
                        for response in question_data['responses']:
                            for model, reward in response['reward_scores'].items():
                                if model not in self.results[dataset_name][reward_models]['rewards']:
                                    self.results[dataset_name][reward_models]['rewards'][model] = []
                                self.results[dataset_name][reward_models]['rewards'][model].append(reward)
    
    def analyze_normality_per_question(self):
        """Analyze normality of rewards for each question."""
        normality_results = {}
        
        for dataset, models in self.results.items():
            normality_results[dataset] = {}
            
            for reward_models, data in models.items():
                normality_results[dataset][reward_models] = {}
                
                for model in data['rewards'].keys():
                    normality_results[dataset][reward_models][model] = {
                        'normal_questions': 0,
                        'non_normal_questions': 0,
                        'question_details': []
                    }
                    
                    for question_id, question_data in data['questions'].items():
                        # Get rewards for this question and model
                        rewards = []
                        for response in question_data['responses']:
                            if model in response['reward_scores']:
                                rewards.append(response['reward_scores'][model])
                        
                        if len(rewards) >= 3:  # Need at least 3 samples for normality test
                            rewards = np.array(rewards)
                            stat, p_value = stats.normaltest(rewards)
                            
                            is_normal = p_value > 0.05  # Using 0.05 significance level
                            
                            if is_normal:
                                normality_results[dataset][reward_models][model]['normal_questions'] += 1
                            else:
                                normality_results[dataset][reward_models][model]['non_normal_questions'] += 1
                            
                            normality_results[dataset][reward_models][model]['question_details'].append({
                                'question_id': question_id,
                                'p_value': float(p_value),
                                'is_normal': str(is_normal),
                                'mean': float(np.mean(rewards)),
                                'std': float(np.std(rewards))
                            })
        
        # Save results
        with open(os.path.join(self.results_dir, 'normality_analysis.json'), 'w') as f:
            json.dump(normality_results, f, indent=2)
        
        # Print summary
        for dataset, models in normality_results.items():
            print(f"\n=== Normality Analysis for {dataset} ===")
            for reward_models, model_data in models.items():
                print(f"\n{reward_models}:")
                for model, results in model_data.items():
                    total = results['normal_questions'] + results['non_normal_questions']
                    if total > 0:
                        normal_percent = (results['normal_questions'] / total) * 100
                        print(f"  {model}:")
                        print(f"    Normal Questions: {results['normal_questions']} ({normal_percent:.1f}%)")
                        print(f"    Non-Normal Questions: {results['non_normal_questions']} ({100-normal_percent:.1f}%)")
    
    def analyze_reward_by_correctness(self):
        """Analyze reward distribution for correct vs incorrect answers."""
        correctness_results = {}
        
        for dataset, models in self.results.items():
            correctness_results[dataset] = {}
            
            for reward_models, data in models.items():
                correctness_results[dataset][reward_models] = {}
                
                for model in data['rewards'].keys():
                    correct_rewards = []
                    incorrect_rewards = []
                    
                    for question_id, question_data in data['questions'].items():
                        ground_truth = str(question_data['ground_truth'])
                        
                        for response in question_data['responses']:
                            if model in response['reward_scores']:
                                reward = response['reward_scores'][model]
                                answer = response.get('final_answer', '')
                                
                                if answer == ground_truth:
                                    correct_rewards.append(reward)
                                else:
                                    incorrect_rewards.append(reward)
                    
                    if correct_rewards and incorrect_rewards:
                        correct_rewards = np.array(correct_rewards)
                        incorrect_rewards = np.array(incorrect_rewards)
                        
                        # Calculate statistics
                        stats = {
                            'correct': {
                                'mean': float(np.mean(correct_rewards)),
                                'std': float(np.std(correct_rewards)),
                                'min': float(np.min(correct_rewards)),
                                'max': float(np.max(correct_rewards)),
                                'count': len(correct_rewards)
                            },
                            'incorrect': {
                                'mean': float(np.mean(incorrect_rewards)),
                                'std': float(np.std(incorrect_rewards)),
                                'min': float(np.min(incorrect_rewards)),
                                'max': float(np.max(incorrect_rewards)),
                                'count': len(incorrect_rewards)
                            }
                        }
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(correct_rewards, incorrect_rewards)
                        stats['t_test'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': str(p_value < 0.05)
                        }
                        
                        correctness_results[dataset][reward_models][model] = stats
                        
                        # Plot distributions
                        plt.figure(figsize=(10, 6))
                        plt.hist(correct_rewards, bins=30, alpha=0.5, label='Correct Answers', density=True)
                        plt.hist(incorrect_rewards, bins=30, alpha=0.5, label='Incorrect Answers', density=True)
                        plt.title(f'Reward Distribution by Correctness\n{dataset} - {reward_models} - {model}')
                        plt.xlabel('Reward')
                        plt.ylabel('Density')
                        plt.legend()
                        plt.savefig(os.path.join(self.results_dir, f'{dataset}_{reward_models}_{model}_correctness_dist.png'))
                        plt.close()
        
        # Save results
        with open(os.path.join(self.results_dir, 'correctness_analysis.json'), 'w') as f:
            json.dump(correctness_results, f, indent=2)
        
        # Print summary
        for dataset, models in correctness_results.items():
            print(f"\n=== Reward Analysis by Correctness for {dataset} ===")
            for reward_models, model_data in models.items():
                print(f"\n{reward_models}:")
                for model, stats in model_data.items():
                    print(f"  {model}:")
                    print(f"    Correct Answers (n={stats['correct']['count']}):")
                    print(f"      Mean: {stats['correct']['mean']:.4f}")
                    print(f"      Std: {stats['correct']['std']:.4f}")
                    print(f"    Incorrect Answers (n={stats['incorrect']['count']}):")
                    print(f"      Mean: {stats['incorrect']['mean']:.4f}")
                    print(f"      Std: {stats['incorrect']['std']:.4f}")
                    print(f"    T-test: t={stats['t_test']['t_statistic']:.4f}, p={stats['t_test']['p_value']:.4f}")
                    print(f"    Significant Difference: {stats['t_test']['significant']}")

    def plot_normal_reward_distributions(self):
        """Plot distributions of rewards that pass normality test."""
        for dataset, models in self.results.items():
            for reward_models, data in models.items():
                for model in data['rewards'].keys():
                    # Collect all rewards and their normality status
                    normal_questions = {}
                    non_normal_questions = {}
                    all_rewards = []
                    
                    for question_id, question_data in data['questions'].items():
                        rewards = []
                        for response in question_data['responses']:
                            if model in response['reward_scores']:
                                rewards.append(response['reward_scores'][model])
                                all_rewards.append(response['reward_scores'][model])
                        
                        if len(rewards) >= 3:
                            rewards = np.array(rewards)
                            stat, p_value = stats.normaltest(rewards)
                            if p_value > 0.05:
                                normal_questions[question_id] = rewards
                            else:
                                non_normal_questions[question_id] = rewards
                    
                    if normal_questions and non_normal_questions:
                        plt.figure(figsize=(15, 5))
                        
                        # 1. Plot randomly selected normal question
                        plt.subplot(1, 3, 1)
                        selected_normal_q = np.random.choice(list(normal_questions.keys()))
                        normal_rewards = normal_questions[selected_normal_q]
                        plt.hist(normal_rewards, bins=20, density=True, alpha=0.6)
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = stats.norm.pdf(x, np.mean(normal_rewards), np.std(normal_rewards))
                        plt.plot(x, p, 'k', linewidth=2)
                        plt.title(f'Normal Question {selected_normal_q}\n{model} - {dataset}')
                        plt.xlabel('Reward')
                        plt.ylabel('Density')
                        
                        # 2. Plot randomly selected non-normal question
                        plt.subplot(1, 3, 2)
                        selected_nonnormal_q = np.random.choice(list(non_normal_questions.keys()))
                        plt.hist(non_normal_questions[selected_nonnormal_q], bins=20, density=True, alpha=0.6)
                        plt.title(f'Non-Normal Question {selected_nonnormal_q}\n{model} - {dataset}')
                        plt.xlabel('Reward')
                        plt.ylabel('Density')
                        
                        # 3. Plot all rewards
                        plt.subplot(1, 3, 3)
                        plt.hist(all_rewards, bins=30, density=True, alpha=0.6)
                        plt.title(f'All Questions\n{model} - {dataset}')
                        plt.xlabel('Reward')
                        plt.ylabel('Density')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.results_dir, f'{dataset}_{reward_models}_{model}_normal_vs_nonnormal.png'))
                        plt.close()
    
    def plot_reward_trends_by_correctness(self):
        """Plot trend curves for correct and incorrect answers."""
        for dataset, models in self.results.items():
            for reward_models, data in models.items():
                for model in data['rewards'].keys():
                    plt.figure(figsize=(10, 5))
                    
                    # Plot Q5
                    if 5 in data['questions']:
                        plt.subplot(1, 2, 1)
                        question_data = data['questions'][5]
                        
                        correct_rewards = []
                        incorrect_rewards = []
                        correct_indices = []
                        incorrect_indices = []
                        
                        print(f"\nPlotting Q5")
                        
                        for idx, response in enumerate(question_data['responses']):
                            if model in response['reward_scores']:
                                reward = response['reward_scores'][model]
                                is_correct = response.get('is_correct', False)
                                print(f"Response {idx}: is_correct={is_correct}, reward={reward}")
                                
                                if is_correct:
                                    correct_rewards.append(reward)
                                    correct_indices.append(idx)
                                    print("Marked as correct")
                                else:
                                    incorrect_rewards.append(reward)
                                    incorrect_indices.append(idx)
                                    print("Marked as incorrect")
                        
                        print(f"Found {len(correct_rewards)} correct and {len(incorrect_rewards)} incorrect answers")
                        
                        if correct_rewards:
                            plt.plot(correct_indices, correct_rewards, 'g-', label='Correct', alpha=0.6)
                        if incorrect_rewards:
                            plt.plot(incorrect_indices, incorrect_rewards, 'r-', label='Incorrect', alpha=0.6)
                        plt.title(f'Question 5\n{model} - {dataset}')
                        plt.xlabel('Response Index')
                        plt.ylabel('Reward')
                        plt.legend()
                    
                    # Plot Q48
                    if 48 in data['questions']:
                        plt.subplot(1, 2, 2)
                        question_data = data['questions'][48]
                        
                        correct_rewards = []
                        incorrect_rewards = []
                        correct_indices = []
                        incorrect_indices = []
                        
                        print(f"\nPlotting Q48")
                        
                        for idx, response in enumerate(question_data['responses']):
                            if model in response['reward_scores']:
                                reward = response['reward_scores'][model]
                                is_correct = response.get('is_correct', False)
                                print(f"Response {idx}: is_correct={is_correct}, reward={reward}")
                                
                                if is_correct:
                                    correct_rewards.append(reward)
                                    correct_indices.append(idx)
                                    print("Marked as correct")
                                else:
                                    incorrect_rewards.append(reward)
                                    incorrect_indices.append(idx)
                                    print("Marked as incorrect")
                        
                        print(f"Found {len(correct_rewards)} correct and {len(incorrect_rewards)} incorrect answers")
                        
                        if correct_rewards:
                            plt.plot(correct_indices, correct_rewards, 'g-', label='Correct', alpha=0.6)
                        if incorrect_rewards:
                            plt.plot(incorrect_indices, incorrect_rewards, 'r-', label='Incorrect', alpha=0.6)
                        plt.title(f'Question 48\n{model} - {dataset}')
                        plt.xlabel('Response Index')
                        plt.ylabel('Reward')
                        plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.results_dir, f'{dataset}_{reward_models}_{model}_trends.png'))
                    plt.close()

def main():
    # Example usage
    analyzer = RewardAnalyzer('results')
    
    # Analyze normality per question
    analyzer.analyze_normality_per_question()
    
    # Analyze reward distribution by correctness
    analyzer.analyze_reward_by_correctness()
    
    # Plot normal reward distributions
    analyzer.plot_normal_reward_distributions()
    
    # Plot reward trends by correctness
    analyzer.plot_reward_trends_by_correctness()

if __name__ == "__main__":
    main() 