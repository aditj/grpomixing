import datasets
from tqdm import tqdm
from openai import OpenAI
import json
import argparse
client = OpenAI(api_key='-', base_url="http://slurm-h100-206-119:8000/v1")
def process_mbpp(problem: str, test_cases: str):
    response = client.responses.create(
    model="gpt-4.1-nano",
    input=[
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            "text": "Construct a precise Python function signature to solve the user's problem.\n# Instructions\n- Provide the function signature.\n- Do not solve the problem or provide a solution.\n - Include data types of the arguments but do not include any data types which are not implemented in a base installation of python.\n- Enumerate input and output parameters.\n- State constraints and assumptions.\n\n# Steps\n1. Define the function signature with parameters.\n2. Describe the logic needed to find the longest chain.\n3. Clearly outline the inputs and constraints.\n4. Specify the output format and type.\n5. Include the given test cases using assert statements obeying the function's signature.\n\n# Output Format\nThe output should include a Python function following best practices, detailing its purpose. Provide at least 5 assert statements that verify the correctness of the implementation. Example assert statements should be realistic and include edge cases.\n\nEnsure appropriate comments accompany code logic and test cases, explaining the reasoning behind each test condition."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "input_text",
            "text": f"Problem: {problem} \nTest cases: {test_cases}"
            }
        ]
        }
    ],
    text={
        "format": {
        "type": "json_schema",
        "name": "solve_function_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "function_signature": {
                "type": "string",
                "description": "Signature of the solve function indicating its parameters and functionality."
            },
            "test_cases": {
                "type": "array",
                "description": "List of assertions to test the correctness of the solve function.",
                "items": {
                "type": "string",
                "description": "An assertion statement that verifies the output of the solve function."
                }
            }
            },
            "required": [
            "function_signature",
            "test_cases"
            ],
            "additionalProperties": False
        }
        }
    },
    reasoning={},
    tools=[],
    temperature=1,
    max_output_tokens=2048,
    top_p=1,
        store=True
        )
    response_json = json.loads(response.output_text)
    return response_json['function_signature'], response_json['test_cases']


def process_mbpp_example(example):
    """
    Processes a single example from the MBPP dataset.
    """
    problem = example['text']
    test_cases_str = "\n".join(example['test_list'])
    try:
        signature, test_cases = process_mbpp(problem, test_cases_str)
        return {
            'function_signature': signature,
            'test_cases': test_cases
        }
    except Exception as e:
        print(f"Error processing example: {e}")
        return {
            'function_signature': f'ERROR: {e}',
            'test_cases': []
        }

def process_leetcode_example(example):
    """
    Processes a single example from the LeetCode dataset.
    """
    content = example['content']
    
    # Separate problem from examples
    example_marker = "**Example"
    example_start_index = content.find(example_marker)
    
    if example_start_index != -1:
        problem = content[:example_start_index].strip()
        examples_and_constraints = content[example_start_index:].strip()
    else:
        problem = content.strip()
        examples_and_constraints = ""
        
    # Remove constraints from the examples part
    constraints_marker = "**Constraints:**"
    constraints_start_index = examples_and_constraints.find(constraints_marker)
    
    if constraints_start_index != -1:
        test_cases_str = examples_and_constraints[:constraints_start_index].strip()
    else:
        test_cases_str = examples_and_constraints.strip()

    try:
        signature, test_cases = process_mbpp(problem, test_cases_str)
        return {
            'function_signature': signature,
            'test_cases': test_cases
        }
    except Exception as e:
        print(f"Error processing example: {e}")
        return {
            'function_signature': f'ERROR: {e}',
            'test_cases': []
        }

def main():
    """
    Loads, processes, and uploads a selected dataset.
    """
    parser = argparse.ArgumentParser(description="Process and upload a dataset to the Hugging Face Hub.")
    parser.add_argument('dataset_to_process', type=str, choices=['mbpp', 'leetcode'], help='The dataset to process.')
    parser.add_argument('hub_repo_name', type=str, help='The name for the repository on the Hugging Face Hub.')
    args = parser.parse_args()

    if args.dataset_to_process == 'mbpp':
        dataset_id = "google-research-datasets/mbpp"
        processing_fn = process_mbpp_example
    elif args.dataset_to_process == 'leetcode':
        dataset_id = "greengerong/leetcode"
        processing_fn = process_leetcode_example

    # Load the dataset from Hugging Face
    print(f"Loading {dataset_id} dataset...")
    dataset = datasets.load_dataset(dataset_id)
    print("Dataset loaded.")

    # Process dataset
    print("Processing dataset...")
    processed_dataset_dict = dataset.map(processing_fn, batched=False)

    # Upload the processed dataset to the Hugging Face Hub
    print(f"Uploading processed dataset to '{args.hub_repo_name}' on the Hugging Face Hub...")
    processed_dataset_dict.push_to_hub(args.hub_repo_name)
    print("Dataset successfully processed and uploaded.")

if __name__ == "__main__":
    main() 