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
    """Read evaluation logs from an experiment directory and return max accuracy."""
    eval_logs_dir = os.path.join(exp_dir, 'eval_logs')
    if not os.path.exists(eval_logs_dir):
        return None
    
    log_files = glob.glob(os.path.join(eval_logs_dir, 'metrics_*.json'))
    if not log_files:
        return None

    # Sort files by step number
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    max_accuracy = 0
    accuracies = []

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            # Use the top-level accuracy field
            accuracy = data.get('accuracy', 0)
            accuracies.append(accuracy)
            max_accuracy = max(max_accuracy, accuracy)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {log_file}: {e}")
            continue

    return max_accuracy if accuracies else None

def find_experiment_dirs(environment, method, chains):
    """Find experiment directories matching the given criteria."""
    
   
    pattern = f"mixture_grpo_{environment}.reasoning_gym_1.5b*{method}*num_chains_{chains}*"

    full_pattern = os.path.join(OUTPUT_DIR, pattern)
    matching_dirs = glob.glob(full_pattern)
    
    return matching_dirs

def plot_environment_bar_chart(environment):
    """Plot bar chart of best accuracies for a single environment."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = ['vanilla', 'dirichlet', 'different_tokens']
    chains = [5, 10]
    
    # Colors for different methods
    colors = {
        'vanilla': '#1f77b4',
        'dirichlet': '#ff7f0e', 
        'different_tokens': '#2ca02c'
    }
    
    # Data collection
    data = {}
    labels = []
    values = []
    bar_colors = []
    
    for method in methods:
        for chain_count in chains:
            exp_dirs = find_experiment_dirs(environment, method, chain_count)
            
            if not exp_dirs:
                continue
            mean_accuracy = 0
            accuracies = []
            for exp_dir in exp_dirs:
                max_accuracy = read_eval_logs(exp_dir)
                if max_accuracy is not None:
                    mean_accuracy += max_accuracy
                    accuracies.append(max_accuracy)
            mean_accuracy /= len(exp_dirs)
            std_accuracy = np.std(accuracies)
            label = f"{method}\n({chain_count} chains)"
            labels.append(label)
            values.append(mean_accuracy)
            bar_colors.append(colors[method])
            print(f"{environment} - {method} ({chain_count} chains): {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    
    if not values:
        print(f"No data found for {environment}")
        return
    
    # Create bar plot
    bars = ax.bar(labels, values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    ax.set_ylabel('Best Accuracy (%)', fontsize=12)
    ax.set_title(f'Best Accuracy Comparison - {environment.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis to start from 0 and add some padding at the top
    ax.set_ylim(0, max(values) * 1.1)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(FIGURE_OUTPUT_DIR, f'{environment}_best_accuracy_bars.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart: {output_file}")
    
    plt.close()

def plot_combined_bar_chart():
    """Plot a combined bar chart showing best accuracies across all environments."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    methods = ['vanilla', 'dirichlet', 'different_tokens']
    chains = [5, 10]
    
    # Colors for different methods
    colors = {
        'vanilla': '#1f77b4',
        'dirichlet': '#ff7f0e', 
        'different_tokens': '#2ca02c'
    }
    
    # Collect all data
    all_data = {}
    
    for env in ENVIRONMENTS:
        all_data[env] = {}
        for method in methods:
            for chain_count in chains:
                exp_dirs = find_experiment_dirs(env, method, chain_count)
                
                if exp_dirs:
                    exp_dir = exp_dirs[0]
                    max_accuracy = read_eval_logs(exp_dir)
                    if max_accuracy is not None:
                        all_data[env][f"{method}_{chain_count}"] = max_accuracy
    
    # Prepare data for grouped bar chart
    method_chain_combinations = []
    for method in methods:
        for chain_count in chains:
            method_chain_combinations.append(f"{method}_{chain_count}")
    
    x = np.arange(len(ENVIRONMENTS))
    width = 0.12  # Width of bars
    
    # Create bars for each method-chain combination
    for i, combo in enumerate(method_chain_combinations):
        parts = combo.split('_')
        method = '_'.join(parts[:-1])  # Join all parts except the last as method name
        chain_str = parts[-1]         # Last part is chain count
        chain_count = int(chain_str)
        
        values = []
        for env in ENVIRONMENTS:
            value = all_data[env].get(combo, 0)
            values.append(value)
        
        # Offset for grouped bars
        offset = (i - len(method_chain_combinations)/2 + 0.5) * width
        
        # Choose color and pattern
        color = colors[method]
        alpha = 0.8 if chain_count == 5 else 0.5
        
        bars = ax.bar(x + offset, values, width, 
                     label=f'{method} ({chain_count} chains)',
                     color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            if value > 0:  # Only show label if there's data
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Best Accuracy (%)', fontsize=12)
    ax.set_title('Best Accuracy Comparison Across All Environments', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([env.replace('_', ' ').title() for env in ENVIRONMENTS])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(FIGURE_OUTPUT_DIR, 'all_environments_best_accuracy_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined bar chart: {output_file}")
    
    plt.close()

def main():
    """Generate bar chart visualizations for best accuracies."""
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figure output directory: {FIGURE_OUTPUT_DIR}")
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory {OUTPUT_DIR} does not exist")
        return
    
    print("\nGenerating bar charts for best accuracies:")
    
    # Generate individual bar charts for each environment
    for env in ENVIRONMENTS:
        print(f"\nProcessing {env}...")
        try:
            plot_environment_bar_chart(env)
        except Exception as e:
            print(f"Error processing {env}: {e}")
    
    # Generate combined bar chart
    print(f"\nGenerating combined bar chart...")
    try:
        plot_combined_bar_chart()
    except Exception as e:
        print(f"Error generating combined chart: {e}")
    
    print(f"\nAll bar charts saved to: {FIGURE_OUTPUT_DIR}")

if __name__ == "__main__":
    main() 