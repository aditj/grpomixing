import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

def calculate_token_similarity(chain_log, embedding_matrix):
    """
    Calculates the average pairwise cosine similarity between the embeddings
    of the top-k tokens considered in the mixture phase.

    A high value (near 1) means the candidate tokens are semantically similar.
    A low value (near 0 or negative) means they are dissimilar.
    """
    if not chain_log.get('is_mixture_phase', False):
        return None

    token_ids = chain_log.get('top_k_token_ids')

    # Need at least 2 tokens to compare
    if not token_ids or len(token_ids) < 2:
        return None

    try:
        # Get embeddings and ensure they are valid
        valid_token_ids = [tid for tid in token_ids if tid < embedding_matrix.shape[0]]
        if len(valid_token_ids) < 2:
            return None
        
        top_k_embeddings = embedding_matrix[valid_token_ids].float() # Use float for precision
        
        # Normalize embeddings to get unit vectors for cosine similarity
        norm_embeddings = torch.nn.functional.normalize(top_k_embeddings, p=2, dim=1)
        similarity =   torch.cosine_similarity(norm_embeddings[0], norm_embeddings[1], dim=0)
        return similarity.item()
        
        # We only want the average of the off-diagonal elements.
        # The number of unique pairs is k * (k-1) / 2
        num_pairs = len(valid_token_ids) * (len(valid_token_ids) - 1) / 2
        if num_pairs == 0:
            return None
        
        # Use triu with diagonal=1 to get the upper triangle, excluding the diagonal
        avg_similarity = torch.triu(similarity_matrix, diagonal=1).sum() / num_pairs
        
        return avg_similarity.item()

    except IndexError:
        # This shouldn't happen with the check above, but as a safeguard
        print(f"Warning: Token ID out of bounds. Skipping a chain.")
        return None


def process_log_file(file_path, embedding_matrix):
    """
    Processes a single detailed_generation.json file to extract similarity metrics.
    """
    with open(file_path, 'r') as f:
        log_data = json.load(f)

    generation_steps = log_data.get('generation_steps', [])
    step_similarities = []

    for step in generation_steps:
        similarities_in_step = []
        for chain in step.get('chains', []):
            if chain.get('is_running', False) and chain.get('is_mixture_phase', False):
                similarity = calculate_token_similarity(chain, embedding_matrix)
                if similarity is not None:
                    similarities_in_step.append(similarity)
        
        if similarities_in_step:
            avg_similarity_for_step = np.mean(similarities_in_step)
            step_similarities.append(avg_similarity_for_step)
        else:
            step_similarities.append(np.nan)
            
    return step_similarities


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
            if round_num < args.min_round:
                continue
            file_path = os.path.join(log_dir, file_name)
            similarity_data = process_log_file(file_path, embedding_matrix)
            if similarity_data:
                all_rounds_data[round_num] = similarity_data

    if not all_rounds_data:
        print(f"No data could be processed for rounds >= {args.min_round}.")
        return

    # Sort rounds and create a 2D matrix for the heatmap
    sorted_rounds = sorted(all_rounds_data.keys())
    max_steps = max(len(d) for d in all_rounds_data.values()) if all_rounds_data else 0
    heatmap_data = np.full((len(sorted_rounds), max_steps), np.nan)

    for i, round_num in enumerate(sorted_rounds):
        data = all_rounds_data[round_num]
        heatmap_data[i, :len(data)] = data

    # Plotting
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, xticklabels=50, cmap="viridis", cbar_kws={'label': 'Avg. Pairwise Cosine Similarity'})
    plt.xlabel("Generation Step")
    plt.ylabel("Training Round")
    plt.title("Avg. Pairwise Similarity of Candidate Tokens")
    
    y_tick_labels = [str(r) for r in sorted_rounds]
    if len(y_tick_labels) > 20:
        tick_indices = np.linspace(0, len(y_tick_labels) - 1, 20, dtype=int)
        plt.yticks(tick_indices + 0.5, [y_tick_labels[i] for i in tick_indices], rotation=0)
    else:
        plt.yticks(np.arange(len(sorted_rounds)) + 0.5, y_tick_labels, rotation=0)

    output_path = os.path.join(log_dir, 'token_similarity_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot avg. pairwise cosine similarity of candidate tokens.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing the '*_detailed_generation.json' files.")
    parser.add_argument("--embedding_file", type=str, default="output/token_embeddings.pt", help="Path to the .pt file with token embeddings.")
    parser.add_argument("--min_round", type=int, default=0, help="Minimum training round to include in the plot.")
    
    args = parser.parse_args()
    main(args) 