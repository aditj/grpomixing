import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

def calculate_diversity(chain_log):
    """
    Calculates token diversity for a single generation step in a single chain.

    The diversity metric is calculated as 1 - sum(p^2), where p is the vector
    of normalized mixture weights. This is the Gini-Simpson Index, a measure of
    diversity. It is 0 if one token has all the probability, and approaches 1
    as probability is spread evenly.
    """
    if not chain_log.get('is_mixture_phase', False):
        return None
    
    weights = chain_log.get('normalized_mixture_weights')
    if not weights:
        return None
        
    # HHI index is sum of squares of probabilities.
    hhi = sum(p**2 for p in weights)
    # Diversity is 1 - HHI
    diversity = 1 - hhi
    return diversity

def process_log_file(file_path):
    """
    Processes a single detailed_generation.json file to extract diversity metrics.
    """
    with open(file_path, 'r') as f:
        log_data = json.load(f)

    generation_steps = log_data.get('generation_steps', [])
    step_diversities = []

    for step in generation_steps:
        diversities_in_step = []
        for chain in step.get('chains', []):
            if chain.get('is_running', False) and chain.get('is_mixture_phase', False):
                diversity = calculate_diversity(chain)
                if diversity is not None:
                    diversities_in_step.append(diversity)
        
        if diversities_in_step:
            avg_diversity_for_step = np.mean(diversities_in_step)
            step_diversities.append(avg_diversity_for_step)
        else:
            # Append NaN if no running mixture chains in this step, or handle as 0
            step_diversities.append(np.nan)
            
    return step_diversities


def main(args):
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"Error: Directory not found at {log_dir}")
        return

    # Find all detailed generation logs
    log_files = [f for f in os.listdir(log_dir) if f.endswith('_detailed_generation.json')]
    
    if not log_files:
        print(f"No '*_detailed_generation.json' files found in {log_dir}")
        return

    all_rounds_data = {}
    
    # Process each log file
    for file_name in tqdm(log_files, desc="Processing log files"):
        match = re.match(r'(\d+)_detailed_generation\.json', file_name)
        if match:
            round_num = int(match.group(1))
            file_path = os.path.join(log_dir, file_name)
            diversity_data = process_log_file(file_path)
            if diversity_data:
                all_rounds_data[round_num] = diversity_data

    if not all_rounds_data:
        print("No data could be processed.")
        return

    # Sort rounds
    sorted_rounds = sorted(all_rounds_data.keys())
    
    # Create a 2D matrix for the heatmap
    max_steps = 0
    if all_rounds_data:
      max_steps = max(len(d) for d in all_rounds_data.values())

    heatmap_data = np.full((len(sorted_rounds), max_steps), np.nan)

    for i, round_num in enumerate(sorted_rounds):
        data = all_rounds_data[round_num]
        heatmap_data[i, :len(data)] = data

    # Plotting
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, xticklabels=50, yticklabels=sorted_rounds, cmap="viridis", cbar_kws={'label': 'Token Diversity (1 - HHI)'})
    plt.xlabel("Generation Step")
    plt.ylabel("Training Round")
    plt.title("Token Diversity across Training Rounds and Generation Steps")
    
    # Adjust y-ticks to show round numbers
    y_tick_labels = [str(r) for r in sorted_rounds]
    if len(y_tick_labels) > 20: # Heuristic to avoid crowded labels
        tick_indices = np.linspace(0, len(y_tick_labels) - 1, 20, dtype=int)
        y_ticks = tick_indices + 0.5 # Center ticks
        y_tick_labels = [y_tick_labels[i] for i in tick_indices]
        plt.yticks(y_ticks, y_tick_labels, rotation=0)
    else:
        plt.yticks(np.arange(len(sorted_rounds)) + 0.5, y_tick_labels, rotation=0)


    output_path = os.path.join(log_dir, 'token_diversity_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot token diversity from detailed generation logs.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing the '*_detailed_generation.json' files.")
    
    args = parser.parse_args()
    main(args) 