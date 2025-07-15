import os
import torch
import argparse
import llms

def extract_and_save_embeddings(args):
    """
    Loads a model and tokenizer, extracts the input embeddings for the entire
    vocabulary, and saves them to a file.
    """
    print(f"Loading model: {args.model_name}...")
    # Use the project's utility function to load model and tokenizer
    # We load on CPU since we are not doing any computation
    # Flash attention is disabled as it's not needed for this task
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device="cpu", use_flash_attention=0)
    print("Model and tokenizer loaded successfully.")

    # Get the input embedding layer from the model
    embedding_layer = model.get_input_embeddings()
    
    # Get the embedding weights as a tensor.
    # We detach it from the computation graph and ensure it's on the CPU.
    embedding_matrix = embedding_layer.weight.detach().cpu()

    # Get the vocabulary (a dictionary mapping token strings to token IDs)
    vocab = tokenizer.get_vocab()

    # Create a dictionary to hold the data we want to save
    embedding_data = {
        'model_name': args.model_name,
        'vocab': vocab,
        'embeddings': embedding_matrix
    }

    # Ensure the output directory exists before saving
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the data object to the specified file
    print(f"Saving embeddings for {len(vocab)} tokens to {args.output_file}...")
    torch.save(embedding_data, args.output_file)
    
    print("Embeddings saved successfully.")
    print(f"Details:")
    print(f" - Model: {args.model_name}")
    print(f" - Vocabulary Size: {len(vocab)}")
    print(f" - Embedding Dimension: {embedding_matrix.shape[1]}")
    print(f" - Output File: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save token embeddings from a pre-trained model.")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct", 
        help="Name or path of the pre-trained model from Hugging Face."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="output/token_embeddings.pt", 
        help="Path to save the output .pt file containing embeddings and vocabulary."
    )
    
    args = parser.parse_args()
    extract_and_save_embeddings(args) 