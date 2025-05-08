import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process leaderboard data for specific models')
parser.add_argument('--property', type=str, required=True, help='Property to assign to all models (e.g., first, best, dynamic)')
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv('results/alpaca_eval/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv')

# Define the models we're interested in
target_models = [
    'openai/grok-3-mini-fast-beta',
    'deepinfra/Qwen/Qwen2.5-7B-Instruct',
    'deepinfra/meta-llama/Llama-3.2-3B-Instruct'
]

# Filter the dataframe for our target models
filtered_df = df[df.iloc[:, 0].isin(target_models)]

# Select the columns we want
columns_to_keep = ['length_controlled_winrate', 'win_rate', 'standard_error', 'n_total', 'avg_length']
result_df = filtered_df[columns_to_keep]

# Add the property column with the same value for all models
result_df['property'] = args.property

# Get the model names from the first column of filtered_df
result_df['model'] = filtered_df.iloc[:, 0]

# Save to a new CSV file
result_df.to_csv('model_comparison.csv', index=False)

# Print the results
print("\nProcessed Model Information:")
print(result_df.to_string(index=False)) 