import json
import os
import argparse
from pathlib import Path

def combine_reward_scores(eval_dir):
    """
    Combine reward scores from skywork_reward_scores.json into evaluation.json
    
    Args:
        eval_dir (str): Directory containing both evaluation.json and skywork_reward_scores.json
    """
    # Load both JSON files
    eval_path = os.path.join(eval_dir, 'evaluation.json')
    skywork_path = os.path.join(eval_dir, 'skywork_reward_scores.json')
    
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    with open(skywork_path, 'r') as f:
        skywork_scores = json.load(f)
    
    # For each question in evaluation.json
    for question in eval_data['detailed_results']:
        question_id = str(question['id'])
        
        # Skip if this question ID is not in skywork scores
        if question_id not in skywork_scores:
            continue
            
        # Get the scores for this question
        scores = skywork_scores[question_id]
        
        # Add scores to each response
        for i, response in enumerate(question['responses']):
            if i < len(scores):  # Make sure we have a score for this response
                if 'reward_scores' not in response:
                    response['reward_scores'] = {}
                response['reward_scores']['skywork'] = scores[i]
    
    # Save the updated evaluation.json
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Combine Skywork reward scores into evaluation.json')
    parser.add_argument('eval_dir', type=str, help='Directory containing evaluation.json and skywork_reward_scores.json')
    args = parser.parse_args()
    
    combine_reward_scores(args.eval_dir)

if __name__ == '__main__':
    main() 