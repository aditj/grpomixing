import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../output_new')
ENVIRONMENTS = ['acre', 'color_cube_rotation', 'family_relationships', 'graph_color', 'number_sequence']
FIGURE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'visualizations')

# Create output directory for figures
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

def read_eval_logs(exp_dir):
    """Read evaluation logs from an experiment directory."""
    eval_logs_dir = os.path.join(exp_dir, 'eval_logs')
    if not os.path.exists(eval_logs_dir):
        return None
    
    log_files = glob.glob(os.path.join(eval_logs_dir, 'metrics_*.json'))
    if not log_files:
        return None

    # Sort files by step number
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    steps = []
    accuracies = []

    for log_file in log_files:
        step = int(os.path.basename(log_file).split('_')[1].split('.')[0])
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            # Use the top-level accuracy field
            accuracy = data.get('accuracy', 0)
            steps.append(step)
            accuracies.append(accuracy)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {log_file}: {e}")
            continue

    return {'steps': steps, 'accuracy': accuracies}

def find_experiment_dirs(environment, method, chains):
    """Find experiment directories matching the given criteria."""
    
    if chains == 5:
        # For 5 chains, look for num_chains_5 pattern
        if method == "different_tokens":
            pattern = f"mixture_grpo_{environment}.reasoning_gym_1.5b_different_tokens_*num_chains_5*"
        elif method == "dirichlet":
            pattern = f"mixture_grpo_{environment}.reasoning_gym_1.5b_dirichlet_*num_chains_5*"
        elif method == "vanilla":
            pattern = f"mixture_grpo_{environment}.reasoning_gym_1.5b_vanilla_*num_chains_5*"
    elif chains == 10:
        # For 10 chains, look for max_token patterns
        if method == "different_tokens":
            # Try three different patterns for different_tokens
            pattern1 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_*max_token*different_tokens*"
            pattern2 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_different_tokens_*max_token*"
            pattern3 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_*different_tokens*max_token*"
            dirs1 = glob.glob(os.path.join(OUTPUT_DIR, pattern1))
            dirs2 = glob.glob(os.path.join(OUTPUT_DIR, pattern2))
            dirs3 = glob.glob(os.path.join(OUTPUT_DIR, pattern3))
            return dirs1 + dirs2 + dirs3
        elif method == "dirichlet":
            # Try two different patterns for dirichlet
            pattern1 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_dirichlet_*max_token*"
            pattern2 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_dirichlet_*max_token*"
            dirs1 = glob.glob(os.path.join(OUTPUT_DIR, pattern1))
            dirs2 = glob.glob(os.path.join(OUTPUT_DIR, pattern2))
            return dirs1 + dirs2
        elif method == "vanilla":
            # Try two different patterns for vanilla
            pattern1 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_vanilla_*max_token*"
            pattern2 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_vanilla_*max_token*"
            dirs1 = glob.glob(os.path.join(OUTPUT_DIR, pattern1))
            dirs2 = glob.glob(os.path.join(OUTPUT_DIR, pattern2))
            return dirs1 + dirs2
    
    full_pattern = os.path.join(OUTPUT_DIR, pattern)
    matching_dirs = glob.glob(full_pattern)
    
    return matching_dirs

def plot_environment_accuracy(environment):
    """Plot accuracy comparison for a single environment."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = ['vanilla', 'dirichlet', 'different_tokens']
    chains = [5, 10]
    colors = {
        'vanilla': {'5': '#1f77b4', '10': '#aec7e8'},
        'dirichlet': {'5': '#ff7f0e', '10': '#ffbb78'},
        'different_tokens': {'5': '#2ca02c', '10': '#98df8a'}
    }
    
    legend_handles = []
    
    for method in methods:
        for chain_count in chains:
            exp_dirs = find_experiment_dirs(environment, method, chain_count)
            
            if not exp_dirs:
                print(f"No experiments found for {environment}, {method}, {chain_count} chains")
                continue
            
            # Use the first matching directory
            exp_dir = exp_dirs[0]
            print(f"Using experiment: {os.path.basename(exp_dir)}")
            
            data = read_eval_logs(exp_dir)
            if data is None:
                print(f"No evaluation data found for {exp_dir}")
                continue
            
            steps = data['steps']
            accuracy = data['accuracy']
            
            if not steps or not accuracy:
                print(f"Empty data for {exp_dir}")
                continue
            
            color = colors[method][str(chain_count)]
            line_style = '-' if chain_count == 5 else '--'
            label = f"{method} ({chain_count} chains)"
            
            line = ax.plot(steps, accuracy, color=color, linestyle=line_style, 
                          linewidth=2, marker='o', markersize=4, label=label)
            legend_handles.extend(line)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy vs Training Steps - {environment.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set y-axis to show percentage properly
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(FIGURE_OUTPUT_DIR, f'{environment}_accuracy_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_file}")
    
    plt.close()

def main():
    """Generate visualizations for all environments."""
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figure output directory: {FIGURE_OUTPUT_DIR}")
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory {OUTPUT_DIR} does not exist")
        return
    
    print("\nGenerating visualizations for environments:")
    for env in ENVIRONMENTS:
        print(f"\nProcessing {env}...")
        try:
            plot_environment_accuracy(env)
        except Exception as e:
            print(f"Error processing {env}: {e}")
    
    print(f"\nAll visualizations saved to: {FIGURE_OUTPUT_DIR}")

if __name__ == "__main__":
    main() 