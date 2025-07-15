import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

def calculate_dot_product(chain_log, embedding_matrix):
    """
    Calculates the dot product of the mixture embedding with itself.
    
    This is equivalent to the squared L2 norm of the mixture embedding vector.
    A higher value indicates a higher "energy" or "confidence" in the
    synthesized thought vector.
    """
    if not chain_log.get('is_mixture_phase', False):
        return None

    weights = chain_log.get('normalized_mixture_weights')
    token_ids = chain_log.get('top_k_token_ids')

    if not weights or not token_ids:
        return None

    try:
        # Get the embeddings for the top-k tokens
        # Ensure token_ids are valid indices
        token_ids = [tid for tid in token_ids if tid < embedding_matrix.shape[0]]
        if len(token_ids) != len(weights):
             # This can happen if tokenizer vocabulary has changed or there are issues with token ids
             return None
        
        top_k_embeddings = embedding_matrix[token_ids]
        
        # Reconstruct the mixture embedding
        # weights need to be a column vector to multiply with embeddings
        weights_tensor = torch.tensor(weights, dtype=top_k_embeddings.dtype).unsqueeze(1)
        mixture_embedding = (top_k_embeddings * weights_tensor).sum(dim=0)
        
        # Calculate the dot product of the mixture embedding with itself
        dot_product = torch.dot(mixture_embedding, mixture_embedding).item()
        
        return dot_product
    except IndexError:
        print(f"Warning: Token ID out of bounds. Skipping a chain.")
        return None


def process_log_file(file_path, embedding_matrix):
    """
    Processes a single detailed_generation.json file to extract dot product metrics.
    """
    with open(file_path, 'r') as f:
        log_data = json.load(f)

    generation_steps = log_data.get('generation_steps', [])
    step_dot_products = []

    for step in generation_steps:
        dot_products_in_step = []
        for chain in step.get('chains', []):
            if chain.get('is_running', False) and chain.get('is_mixture_phase', False):
                dot_product = calculate_dot_product(chain, embedding_matrix)
                if dot_product is not None:
                    dot_products_in_step.append(dot_product)
        
        if dot_products_in_step:
            avg_dot_product_for_step = np.mean(dot_products_in_step)
            step_dot_products.append(avg_dot_product_for_step)
        else:
            step_dot_products.append(np.nan)
            
    return step_dot_products


def main(args):
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"Error: Directory not found at {log_dir}")
        return

    # Load the saved embeddings
    if not os.path.exists(args.embedding_file):
        print(f"Error: Embedding file not found at {args.embedding_file}")
        print("Please run extract_embeddings.py first.")
        return
        
    print("Loading token embeddings...")
    embedding_data = torch.load(args.embedding_file, map_location='cpu')
    embedding_matrix = embedding_data['embeddings']
    print("Embeddings loaded.")

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
            if round_num < 500:
                continue
            file_path = os.path.join(log_dir, file_name)
            dot_product_data = process_log_file(file_path, embedding_matrix)
            if dot_product_data:
                all_rounds_data[round_num] = dot_product_data

    if not all_rounds_data:
        print("No data could be processed.")
        return

    # Sort rounds and create a 2D matrix for the heatmap
    sorted_rounds = sorted(all_rounds_data.keys())
    max_steps = max(len(d) for d in all_rounds_data.values())
    heatmap_data = np.full((len(sorted_rounds), max_steps), np.nan)

    for i, round_num in enumerate(sorted_rounds):
        data = all_rounds_data[round_num]
        heatmap_data[i, :len(data)] = data

    # Plotting
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, xticklabels=50, cmap="inferno", cbar_kws={'label': 'Embedding Dot Product (Squared L2 Norm)'})
    plt.xlabel("Generation Step")
    plt.ylabel("Training Round")
    plt.title("Mixture Embedding Dot Product across Training Rounds")
    
    # Adjust y-ticks to show round numbers
    y_tick_labels = [str(r) for r in sorted_rounds]
    if len(y_tick_labels) > 20:
        tick_indices = np.linspace(0, len(y_tick_labels) - 1, 20, dtype=int)
        y_ticks = tick_indices + 0.5
        y_tick_labels = [y_tick_labels[i] for i in tick_indices]
        plt.yticks(y_ticks, y_tick_labels, rotation=0)
    else:
        plt.yticks(np.arange(len(sorted_rounds)) + 0.5, y_tick_labels, rotation=0)
    os.makedirs("plots", exist_ok=True)
    os.makedirs(os.path.join("plots", log_dir), exist_ok=True)
    output_path = os.path.join("plots", log_dir, 'embedding_dot_product_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mixture embedding dot product from detailed generation logs.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing the '*_detailed_generation.json' files.")
    parser.add_argument("--embedding_file", type=str, default="output/token_embeddings.pt", help="Path to the .pt file with token embeddings.")
    
    args = parser.parse_args()
    main(args) 