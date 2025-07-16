import os
import glob

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../output_new')
ENVIRONMENTS = ['acre', 'color_cube_rotation', 'family_relationships', 'graph_color', 'number_sequence']

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

def main():
    """Generate summary of found experiments."""
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    
    methods = ['vanilla', 'dirichlet', 'different_tokens']
    chains = [5, 10]
    
    for env in ENVIRONMENTS:
        print(f"\n{env.replace('_', ' ').title()}:")
        print("-" * 30)
        
        for method in methods:
            for chain_count in chains:
                exp_dirs = find_experiment_dirs(env, method, chain_count)
                
                if exp_dirs:
                    print(f"  ✓ {method} ({chain_count} chains): {len(exp_dirs)} experiment(s)")
                    for exp_dir in exp_dirs:
                        print(f"    - {os.path.basename(exp_dir)}")
                else:
                    print(f"  ✗ {method} ({chain_count} chains): No experiments found")
    
    print(f"\nTotal experiments found: {sum(len(find_experiment_dirs(env, method, chains)) for env in ENVIRONMENTS for method in methods for chains in [5, 10])}")

if __name__ == "__main__":
    main() 