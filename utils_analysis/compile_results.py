import os
import json
import pandas as pd
import re
from tqdm import tqdm
import glob
import pdb



def process_results_file(file_path, exp_params):
    """Processes a single JSON results file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        
        # Extract results
        results = data['embedding_mixture']['results']
        
        # Create a list of records
        records = []
        for res in results:
            record = {
                't_e': data['embedding_mixture']['stats']['phase1_avg_rounds'],
                'accuracy': data['embedding_mixture']['accuracy'],
                'problem_id': res['problem_id'],
                'is_correct': res['is_correct'],
                'token_count': res['token_count'],
            }

            record.update(exp_params)
            records.append(record)
            
        return records
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def main():
    """
    Walks through the generation_comparison directory, 
    parses all result JSONs, and aggregates them into a single
    pandas DataFrame, saving it as a Parquet file.
    """
    base_dir = '../soft_thinking/generation_comparison'
    
    # Use glob to find all JSON files
    json_files = glob.glob(f'{base_dir}/**/*.json', recursive=True)
    STOPPING_CRITERIA=["end_token", "entropy_threshold", "max_rounds"]
    ANSWER_HOW=["answer_first", "think_first"]
    SAMPLING_STRATEGY=["nucleus", "top_k", "cluster"]
    AGGREGATION_STRATEGY=   ["weighted_sum", "element_wise_max", "dirichlet", "second_moment"]
    Ks=[1, 2, 4, 8]
    all_records = []
    # pdb.set_trace()
    for file_path in tqdm(json_files, desc="Processing result files"):
        if not any(stopping_criteria in file_path for stopping_criteria in STOPPING_CRITERIA):
            continue
        if not any(answer_how in file_path for answer_how in ANSWER_HOW):
            continue
        if not any(sampling_strategy in file_path for sampling_strategy in SAMPLING_STRATEGY):
            continue
        if not any(aggregation_strategy in file_path for aggregation_strategy in AGGREGATION_STRATEGY):
            continue
        if not any("k"+str(k) in file_path for k in Ks):
            continue
        stopping_criteria = [s for s in STOPPING_CRITERIA if s in file_path][0]
        answer_how = [a for a in ANSWER_HOW if a in file_path][0]
        sampling_strategy = [s for s in SAMPLING_STRATEGY if s in file_path][0]
        aggregation_strategy = [a for a in AGGREGATION_STRATEGY if a in file_path][0]
        k = [int(k) for k in Ks if str(k) in file_path][0]
        exp_params = {
            'stopping_criteria': stopping_criteria,
            'answer_how': answer_how,
            'sampling_strategy': sampling_strategy,
            'aggregation_strategy': aggregation_strategy,
            'k': k
        }
        records = process_results_file(file_path, exp_params)
        all_records.extend(records)
        
    if not all_records:
        print("No records found. Exiting.")
        return
        
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Save to Parquet
    output_path = 'compiled_results.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Successfully processed {len(json_files)} files.")
    print(f"Aggregated {len(df)} records into '{output_path}'.")
    print("\nDataFrame Info:")
    df.info()
    print("\nDataFrame Head:")
    print(df.head())


if __name__ == "__main__":
    main() 