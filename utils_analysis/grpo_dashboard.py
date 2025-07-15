from flask import Flask, jsonify, render_template, request
import os
import json
import glob
import pdb
import re
app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../output_new')

@app.route('/')
def index():
    return render_template('grpo_dashboard.html')

@app.route('/api/experiments')
def get_experiments():
    try:
        experiments = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
        return jsonify(sorted(experiments))
    except FileNotFoundError:
        return jsonify({"error": f"Output directory not found at {OUTPUT_DIR}"}), 404

def read_training_logs(exp_dir):
    log_file = os.path.join(exp_dir, 'training_logs', 'train_logs.json')
    if not os.path.exists(log_file):
        return None
    with open(log_file, 'r') as f:
        data = json.load(f)

    metrics = {
        'steps': [],
        'loss': [],
        'kl': [],
        'response_length': [],
        'reward_std': [],
        'grad_norm': [],
        'learning_rate': [],
        'avg_thinking_tokens': []
    }
    
    # data is a dict of dicts, where keys are step numbers (as strings)
    steps = sorted([int(k) for k in data.keys()])
    for step in steps:
        step_str = str(step)
        metrics['steps'].append(step)
        metrics['loss'].append(data[step_str].get('loss'))
        metrics['kl'].append(data[step_str].get('kl'))
        metrics['response_length'].append(data[step_str].get('response_length'))
        metrics['reward_std'].append(data[step_str].get('reward_std'))
        metrics['grad_norm'].append(data[step_str].get('grad_norm'))
        metrics['learning_rate'].append(data[step_str].get('learning_rate'))

        # Calculate avg thinking tokens from generation logs
        avg_thinking_tokens = 0
        try:
            gen_log_file = os.path.join(exp_dir, 'training_logs', f'{step}_generations.txt')
            if os.path.exists(gen_log_file):
                with open(gen_log_file, 'r') as f_gen:
                    content = f_gen.read()
                
                responses = re.split(r"#### GENERATION \d+ RESPONSE ####", content)[1:]
                thinking_tokens_counts = []
                for resp_block in responses:
                    # Extract just the response text
                    resp_text = re.split(r"#### GENERATION \d+ SCORES ####", resp_block)[0]
                    thinking_tokens_counts.append(len(resp_text.split()))
                    
                if thinking_tokens_counts:
                    avg_thinking_tokens = sum(thinking_tokens_counts) / len(thinking_tokens_counts)
        except Exception as e:
            print(f"Could not process generation log for step {step}: {e}")

        metrics['avg_thinking_tokens'].append(avg_thinking_tokens)

    return metrics

def read_eval_logs(exp_dir):
    log_files = glob.glob(os.path.join(exp_dir, 'eval_logs', 'metrics_*.json'))
    if not log_files:
        return None

    # sort files by step number
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    metrics = {
        'steps': [],
        'accuracy': [],
    }
    eval_metrics = {}

    for log_file in log_files:
        step = int(os.path.basename(log_file).split('_')[1].split('.')[0])
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        metrics['steps'].append(step)
        metrics['accuracy'].append(data.get('accuracy'))
        
        if 'metrics' in data:
            for key, value in data['metrics'].items():
                if key not in eval_metrics:
                    eval_metrics[key] = {'steps': [], 'values': []}
                eval_metrics[key]['steps'].append(step)
                eval_metrics[key]['values'].append(value)

    return {'summary': metrics, 'details': eval_metrics}


@app.route('/api/experiment_data')
def get_experiment_data():
    exp1_name = request.args.get('exp1')
    exp2_name = request.args.get('exp2')

    if not exp1_name:
        return jsonify({"error": "exp1 parameter is required"}), 400

    exp1_dir = os.path.join(OUTPUT_DIR, exp1_name)
    
    response = {
        'exp1': {
            'name': exp1_name,
            'training': read_training_logs(exp1_dir),
            'evaluation': read_eval_logs(exp1_dir)
        }
    }

    if exp2_name:
        exp2_dir = os.path.join(OUTPUT_DIR, exp2_name)
        response['exp2'] = {
            'name': exp2_name,
            'training': read_training_logs(exp2_dir),
            'evaluation': read_eval_logs(exp2_dir)
        }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5001) 