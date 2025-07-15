#!/bin/bash
#SBATCH --job-name=new_generation_method_loss_on_all_tokens
#SBATCH --gres=gpu:h100:1     # Request 1× H100 GPU
#SBATCH --time=24:00:00               # HH:MM:SS
#SBATCH --array=0-2  # 4 experiments (see arrays below)
#SBATCH --output=std/%x_%A_%a.out   # Stdout/stderr per-array-task

# Optional: load modules / activate environment
source vinv/bin/activate
cd GRPO_based_soft_thinking

K_vals=(2 2 2 2 2 2 2) 
tasks=("maze" "emoji_mystery" "mini_sudoku"  "family_relationships"  "number_sequence"  "acre"  "color_cube_rotation" "graph_color"  "shortest_path" "prime_factorization"  )
task_index=$((SLURM_ARRAY_TASK_ID % 6))
task=${tasks[$task_index]}
dataset="${task}.reasoning_gym"
K_index=$((SLURM_ARRAY_TASK_ID / 6))
K=${K_vals[$K_index]}

num_chains=10
vanilla_true_vals=(0)
dirichlet_true_vals=(1)

num_steps=1000
max_length=768
eval_iterations=50
max_prompt_length=1024
max_completion_length=1024
temperature=0.9

vanilla_true=0
dirichlet_true=0
loss_on_all_tokens=0

if [ $loss_on_all_tokens -eq 1 ]; then
    loss_on_all_tokens_string="all_tokens"
else
    loss_on_all_tokens_string="max_token"
fi
if [ $vanilla_true -eq 1 ]; then
    experiment_name="${dataset}_1.5b_vanilla_${num_steps}_steps_${loss_on_all_tokens_string}_eval_${eval_iterations}_eval_rg_word_count"
elif [ $dirichlet_true -eq 1 ]; then
    experiment_name="${dataset}_1.5b_dirichlet_${num_steps}_steps_${loss_on_all_tokens_string}_eval_${eval_iterations}_eval_rg_word_count"
else
    experiment_name="${dataset}_1.5b_different_tokens_${num_steps}_steps_${loss_on_all_tokens_string}_eval_${eval_iterations}_eval_rg_word_count"
fi

learning_rate=5e-6
model_name="Qwen/Qwen2.5-1.5B-Instruct"
echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID with T_e=$T_E, k=$K on GPU $CUDA_VISIBLE_DEVICES and experiment_name=$experiment_name and model_name=$model_name and num_chains=$num_chains"
#experiment_name=${experiment_names[$SLURM_ARRAY_TASK_ID]}
python main.py \
  --output_dir output_new/mixture_grpo_${experiment_name}_${K}\
  --model_name $model_name \
  --mixture_k $K \
  --num_chains $num_chains \
  --num_train_iters $num_steps \
  --learning_rate $learning_rate \
  --experiment_name $experiment_name \
  --normal_generation $vanilla_true \
  --dataset $dataset \
  --eval_iterations $eval_iterations \
  --loss_on_all_tokens $loss_on_all_tokens \
  --temperature $temperature \
  --max_prompt_length $max_prompt_length \
  --max_completion_length $max_completion_length

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID" 