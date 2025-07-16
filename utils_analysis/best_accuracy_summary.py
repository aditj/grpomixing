import os
import json
import glob

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../output_new')
ENVIRONMENTS = ['acre', 'color_cube_rotation', 'family_relationships', 'graph_color', 'number_sequence']

def read_eval_logs(exp_dir):
    """Read evaluation logs from an experiment directory and return max accuracy."""
    eval_logs_dir = os.path.join(exp_dir, 'eval_logs')
    if not os.path.exists(eval_logs_dir):
        return None
    
    log_files = glob.glob(os.path.join(eval_logs_dir, 'metrics_*.json'))
    if not log_files:
        return None

    max_accuracy = 0
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            accuracy = data.get('accuracy', 0)
            max_accuracy = max(max_accuracy, accuracy)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    return max_accuracy

def find_experiment_dirs(environment, method, chains):
    """Find experiment directories matching the given criteria."""
    if chains == 5:
        if method == "different_tokens":
            pattern = f"mixture_grpo_{environment}.reasoning_gym_1.5b_different_tokens_*num_chains_5*"
        elif method == "dirichlet":
            pattern = f"mixture_grpo_{environment}.reasoning_gym_1.5b_dirichlet_*num_chains_5*"
        elif method == "vanilla":
            pattern = f"mixture_grpo_{environment}.reasoning_gym_1.5b_vanilla_*num_chains_5*"
    elif chains == 10:
        if method == "different_tokens":
            pattern1 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_*max_token*different_tokens*"
            pattern2 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_different_tokens_*max_token*"
            pattern3 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_*different_tokens*max_token*"
            dirs1 = glob.glob(os.path.join(OUTPUT_DIR, pattern1))
            dirs2 = glob.glob(os.path.join(OUTPUT_DIR, pattern2))
            dirs3 = glob.glob(os.path.join(OUTPUT_DIR, pattern3))
            return dirs1 + dirs2 + dirs3
        elif method == "dirichlet":
            pattern1 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_dirichlet_*max_token*"
            pattern2 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_dirichlet_*max_token*"
            dirs1 = glob.glob(os.path.join(OUTPUT_DIR, pattern1))
            dirs2 = glob.glob(os.path.join(OUTPUT_DIR, pattern2))
            return dirs1 + dirs2
        elif method == "vanilla":
            pattern1 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_llmjudge_vanilla_*max_token*"
            pattern2 = f"mixture_grpo_{environment}.reasoning_gym_1.5b_vanilla_*max_token*"
            dirs1 = glob.glob(os.path.join(OUTPUT_DIR, pattern1))
            dirs2 = glob.glob(os.path.join(OUTPUT_DIR, pattern2))
            return dirs1 + dirs2
    
    full_pattern = os.path.join(OUTPUT_DIR, pattern)
    return glob.glob(full_pattern)

def main():
    print("="*80)
    print("BEST ACCURACY SUMMARY ACROSS ALL ENVIRONMENTS")
    print("="*80)
    
    methods = ['vanilla', 'dirichlet', 'different_tokens']
    chains = [5, 10]
    
    # Collect all results
    all_results = {}
    
    for env in ENVIRONMENTS:
        all_results[env] = {}
        print(f"\n🎯 {env.replace('_', ' ').title()}")
        print("-" * 50)
        
        best_score = 0
        best_config = ""
        
        for method in methods:
            for chain_count in chains:
                exp_dirs = find_experiment_dirs(env, method, chain_count)
                
                if exp_dirs:
                    exp_dir = exp_dirs[-1]
                    max_accuracy = read_eval_logs(exp_dir)
                    
                    if max_accuracy is not None:
                        all_results[env][f"{method}_{chain_count}"] = max_accuracy
                        print(f"  {method:15} ({chain_count:2} chains): {max_accuracy:6.2f}%")
                        
                        if max_accuracy > best_score:
                            best_score = max_accuracy
                            best_config = f"{method} ({chain_count} chains)"
        
        if best_score > 0:
            print(f"  🏆 Best: {best_config} - {best_score:.2f}%")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE RANKING")
    print("="*80)
    
    # Calculate average performance per method-chain combination
    method_chain_avgs = {}
    for method in methods:
        for chain_count in chains:
            key = f"{method}_{chain_count}"
            scores = []
            for env in ENVIRONMENTS:
                if key in all_results[env]:
                    scores.append(all_results[env][key])
            if scores:
                avg_score = sum(scores) / len(scores)
                method_chain_avgs[key] = avg_score
    
    # Sort by average performance
    sorted_configs = sorted(method_chain_avgs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Average Performance Across All Environments:")
    print("-" * 50)
    for i, (config, avg_score) in enumerate(sorted_configs, 1):
        method, chain_str = config.rsplit('_', 1)
        print(f"{i:2}. {method:15} ({chain_str:2} chains): {avg_score:6.2f}%")
    
    # Environment ranking
    print(f"\n🏆 Best Performance by Environment:")
    print("-" * 50)
    env_best = []
    for env in ENVIRONMENTS:
        best_score = 0
        best_config = ""
        for config, score in all_results[env].items():
            if score > best_score:
                best_score = score
                method, chain_str = config.rsplit('_', 1)
                best_config = f"{method} ({chain_str} chains)"
        env_best.append((env, best_score, best_config))
    
    # Sort environments by best score
    env_best.sort(key=lambda x: x[1], reverse=True)
    
    for i, (env, score, config) in enumerate(env_best, 1):
        print(f"{i}. {env.replace('_', ' ').title():20}: {score:6.2f}% ({config})")

if __name__ == "__main__":
    main() 