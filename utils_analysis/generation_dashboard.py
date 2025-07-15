from flask import Flask, jsonify, render_template, request
import os
import json
import glob
import re
import numpy as np

app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'GRPO_based_soft_thinking', 'output')

@app.route('/')
def index():
    return render_template('generation_dashboard.html')

@app.route('/api/experiments')
def get_experiments():
    try:
        experiments = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
        return jsonify(sorted(experiments))
    except FileNotFoundError:
        return jsonify({"error": f"Output directory not found at {OUTPUT_DIR}"}), 404

def read_generation_logs(exp_dir):
    log_files = glob.glob(os.path.join(exp_dir, 'training_logs', '*_detailed_generation.json'))
    print(log_files)
    if not log_files:
        return None

    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[0]))

    metrics = {
        'steps': [],
        'avg_mixture_proportion': [],
        'avg_normalized_prob_std': [],
        'avg_sequence_length': [],
        'phase_transition_counts': []
    }
    
    for log_file in log_files:
        step = int(os.path.basename(log_file).split('_')[0])
        with open(log_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        metrics['steps'].append(step)
        
        # Process generation_steps
        num_chains = data.get('num_chains', 1)
        mixture_steps = 0
        total_steps = 0
        normalized_prob_stds = []
        
        for step_data in data.get('generation_steps', []):
            total_steps += 1
            for chain_data in step_data.get('chains', []):
                if chain_data.get('is_mixture_phase'):
                    mixture_steps += 1
                if 'normalized_mixture_weights' in chain_data:
                    weights = chain_data['normalized_mixture_weights']
                    if len(weights) > 1:
                        normalized_prob_stds.append(np.std(weights))

        metrics['avg_mixture_proportion'].append(mixture_steps / (total_steps * num_chains) if (total_steps * num_chains) > 0 else 0)
        metrics['avg_normalized_prob_std'].append(np.mean(normalized_prob_stds) if normalized_prob_stds else 0)

        # Process final_sequences
        seq_lengths = data.get('final_sequences', {}).get('sequence_lengths', [])
        metrics['avg_sequence_length'].append(np.mean(seq_lengths) if seq_lengths else 0)

        # Process phase_transitions
        metrics['phase_transition_counts'].append(len(data.get('phase_transitions', [])))

    return metrics

@app.route('/api/generation_data')
def get_generation_data():
    exp_name = request.args.get('exp')

    if not exp_name:
        return jsonify({"error": "exp parameter is required"}), 400

    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    
    response = {
        'exp': {
            'name': exp_name,
            'generation_stats': read_generation_logs(exp_dir)
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5002) 