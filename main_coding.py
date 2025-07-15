"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
import torch.nn.functional as F
import llms
import utils
import evaluator
import rldatasets
import pdb
import multiprocessing as mp
from main import generate_with_embeddings, compute_loss, parse_args
mp.set_start_method("spawn", force=True)
def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_loader: DataLoader for test set
        eval_class: Evaluator for computing rewards
        device: Device to run on
        args: Training arguments
        round_num: Current training round number
        
    Returns:
        total_scores: Dictionary of average metrics
        accuracy: Accuracy on test set
    """
    print("Running evaluation on test set...")
    
    # Track metrics across all test examples
    total_scores = defaultdict(float)
    num_examples = 0
    total_accuracy = 0.0

    # Create log file for this evaluation round
    log_file = os.path.join(args.output_dir, f'eval_metrics_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        # Run through test set
        for question, answer, function_signature in tqdm(test_loader, desc="Evaluating on test set"):
            # Generate completions using same function as training
            if args.normal_generation:
                prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
                    model, tokenizer, question, function_signature, device, args
                )
            else:
                prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text, generation_log, token_embeddings_list = generate_completions(
                    model, tokenizer, question, function_signature, device, args
                )
            
            # Score completions using evaluator
            prompt_for_question = f"Using the following function signature {function_signature}, generate a python function that solves the problem: " + question
            mock_prompts = [[{'content': prompt_for_question}]] * len(completions_text)
            mock_completions = [[{'content': completion}] for completion in completions_text]
            # Make answer array same length as completions
            answers = [answer] * len(completions_text)
            rewards_per_func, metrics = eval_class.compute_rewards(
                prompts=mock_prompts,
                completions=mock_completions, 
                function_signature=function_signature,
                answer=answers,
                device=device
            )
            
            # Track accuracy and accumulate metrics
            total_accuracy += metrics['accuracy']
                
            for k, v in metrics.items():
                total_scores[k] += v
            num_examples += 1

            # Log this example
            f.write("\n" + "="*50 + "\n")
            f.write(f"Q# {num_examples}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Function Signature: {function_signature}\n")
            f.write(f"Response: {completions_text[0]}\n") # Log first completion
            f.write(f"Ground Truth: {answer}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"Total Score: {rewards_per_func.sum().item()}\n")
            
            # Log generation details if verbose
            if args.verbose and generation_log:
                f.write(f"Phase transitions: {generation_log.get('phase_transitions', [])}\n")
                f.write(f"Final sequence lengths: {generation_log.get('final_sequences', {}).get('sequence_lengths', [])}\n")

    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    accuracy = total_accuracy / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir,f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'accuracy': accuracy}, f, indent=4)

    if args.verbose:
        print("\nEvaluation Results:")
        print("-" * 20)
        print(f"Accuracy: {accuracy:.2f}%")
        for metric, value in avg_scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 20)

    return avg_scores, accuracy

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    question: str,
    function_signature: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str, dict | None, torch.Tensor | None]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        question: The input question/prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
        generation_log: Dictionary containing detailed generation information
    """
    # 1. Prepare prompting
    prompt_for_question = f"Using the following function signature {function_signature}, generate a python function that solves the problem: " + question
    prompt = [
        {'role': 'system', 'content': args.system_prompt},
        {'role': 'user', 'content': prompt_for_question}
    ]
    prompt_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt to max length and repeat for number of generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    
    # Repeat for number of chains/generations
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    # Move tensors to model's device (handles multi-GPU setups)
    model_device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(model_device)
    prompt_mask = prompt_mask.to(model_device)

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id
    )
    token_embeddings_list = None
    if args.normal_generation:
        # Generate completions
        prompt_completion_ids = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            generation_config=generation_config,
        )
    else:   
        # Generate completions
        prompt_completion_ids, generation_log, token_embeddings_list = generate_with_embeddings(
            model,
            tokenizer,
            prompt_ids,
            prompt_mask,
            device,
            args
        )
    # pdb.set_trace()

    # Extract completion ids
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking 
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model_device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=model_device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    if args.normal_generation:
        return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text
    else: 
        return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text, generation_log, token_embeddings_list
    
def score_completions(
    completions_text: list[str],
    question: str,
    answer: str,
    function_signature: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': question,
            'answer': answer,
            'function_signature': function_signature
        },
        'generations': []
    }

    # Format inputs as expected by evaluator
    prompt_for_question = f"Using the following function signature {function_signature}, generate a python function that solves the problem: " + question
    mock_prompts = [[{'content': prompt_for_question}]] * len(completions_text)
    mock_completions = [[{'content': completion}] for completion in completions_text]
    answers = [answer] * len(completions_text)
    
    # Get rewards and metrics from evaluator
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        function_signature=function_signature,
        answer=answers,
        device=device
    )
    rewards = rewards_per_func.sum(dim=1)

    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data

def grpo_loss(
        model: PreTrainedModel,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        question: str,
        answer: str,
        function_signature: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float], float]:
    """
    Compute GRPO loss between the current model and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        tokenizer: Tokenizer for the models
        question: Input question/prompt
        answer: Ground truth answer
        function_signature: Function signature for the problem
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """
    # Generate completions
    if args.normal_generation:
        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
            model, tokenizer, question, function_signature, device, args
        )
    else:
        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text, generation_log, token_embeddings_list = generate_completions(
            model, tokenizer, question, function_signature, device, args
        )

    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, answer, function_signature, eval_class, device, args
    )

    # Add generation log to the log data
    log_data['generation_log'] = generation_log if not args.normal_generation else None

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file,args.dataset)
    
    # Also save detailed generation log as JSON for analysis
    if not args.normal_generation:
        detailed_log_file = os.path.join(training_log_dir, f'{round_num}_detailed_generation.json')
        
        with open(detailed_log_file, 'w') as f:
            json.dump(generation_log, f, indent=2)

    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    if args.normal_generation:
        loss, loss_metrics = compute_loss(
            model, base_model, prompt_completion_ids, None, prompt_ids, completion_ids,
            attention_mask, completion_mask, advantages, args
        )
    else:
        loss, loss_metrics = compute_loss(
            model, base_model, prompt_completion_ids, token_embeddings_list, prompt_ids, completion_ids,
            attention_mask, completion_mask, advantages, args
        )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics

if __name__ == "__main__":

    # Get all args 
    args = parse_args() 
    if args.dataset == "mbpp" or args.dataset == "leetcode":
        args.system_prompt = """
        You are a helpful coding assistant. Generate a python function that solves the problem using the provided function signature.
        First, think step by step and reason about the problem inside the <reasoning></reasoning> tags. Then, generate the function.
        Respond in the following format only. Make sure to think step by step and reason about the problem before generating the function. 
        <reasoning>
        ...
        </reasoning>
        ```python
        ...
        ```
        """
    # Seed everything 
    utils.seed_everything(args.seed)

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    ###############################
    ## Main Experiment settings ##
    ###############################

    ## Set which model to train 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device,use_flash_attention=args.use_flash_attention)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device,use_flash_attention=args.use_flash_attention)

    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.dataset)

    ###############################


    # Setup logging 
    output_dir = args.output_dir #if args.normal_generation else os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)
    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    eval_log_dir = os.path.join(output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)


    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)


    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    for round_num in tqdm(range(args.num_train_iters), desc="Training Progress"):
        args.round_num = round_num
        # Evaluate on test set every so often 
        if (round_num+1) % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

        # Slowly update ref model
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Get next question
        question, answer, function_signature = next(train_loader)

        # Do GRPO - generate chains, score, compute advantage, compute loss 
        total_loss, train_metrics = grpo_loss(model, base_model, tokenizer, question, answer, function_signature, eval_class, device, round_num, train_log_dir, args)
        
        # Gradient accumulation
        total_loss = total_loss # / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()

        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    

        # Logs
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)
       
