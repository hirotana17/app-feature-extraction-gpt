import json
from typing import Dict
import prompt
import os

# Configuration
BASE_DIR = './data'
SYSTEM_PROMPT = prompt.PROMPTFT1
PROMPT_NAME = "promptft1"

def create_training_example(review: Dict) -> Dict:
    """
    Create a training example in the format required for fine-tuning.
    
    Args:
        review (Dict): A review dictionary containing 'input' and 'output' fields
    
    Returns:
        Dict: A training example in the format required for fine-tuning
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": review['input']
        },
        {
            "role": "assistant",
            "content": json.dumps(review['output'])
        }
    ]
    
    return {"messages": messages}

def generate_training_data(input_file: str, output_dir: str) -> None:
    """
    Generate training data for fine-tuning from the input JSON file.
    
    Args:
        input_file (str): Path to the input JSON file containing reviews
        output_dir (str): Path to the output directory
    """
    # Generate output file name
    base_name = os.path.basename(input_file).rsplit('.', 1)[0]  # Remove extension
    output_file = os.path.join(output_dir, f"{base_name}-finetuning-{PROMPT_NAME}.json")
    
    # Read input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    # Generate training examples
    training_examples = [create_training_example(review) for review in reviews]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(training_examples)} training examples")
    print(f"Output saved to: {output_file}")

def process_in_domain_directory():
    """Process all bin directories in the in-domain directory"""
    print("Processing in-domain data...")
    
    for bin_num in range(0, 10):
        bin_dir = os.path.join(BASE_DIR, 'in-domain', f'bin{bin_num}')
        input_dir = os.path.join(bin_dir, 'formatted_original_data')
        output_dir = os.path.join(bin_dir, 'fine_tuning_data')
        
        if not os.path.exists(input_dir):
            print(f"Warning: formatted_original_data directory not found in bin{bin_num}")
            continue
        
        print(f"\nProcessing bin{bin_num}")
        
        # Process train-set.json
        train_input = os.path.join(input_dir, 'train-set.json')
        if os.path.exists(train_input):
            print(f"Processing train-set.json in bin{bin_num}")
            generate_training_data(train_input, output_dir)
        else:
            print(f"Warning: train-set.json not found in bin{bin_num}")
        
        # Process test-set.json
        test_input = os.path.join(input_dir, 'test-set.json')
        if os.path.exists(test_input):
            print(f"Processing test-set.json in bin{bin_num}")
            generate_training_data(test_input, output_dir)
        else:
            print(f"Warning: test-set.json not found in bin{bin_num}")

def process_out_of_domain_directory():
    """Process all category directories in the out-of-domain directory"""
    print("\nProcessing out-of-domain data...")
    
    bin_dirs = [
        "COMMUNICATION",
        "HEALTH_AND_FITNESS",
        "LIFESTYLE",
        "MAPS_AND_NAVIGATION",
        "PERSONALIZATION",
        "PRODUCTIVITY",
        "SOCIAL",
        "TOOLS",
        "TRAVEL_AND_LOCAL",
        "WEATHER",
    ]
    
    for bin_name in bin_dirs:
        bin_dir = os.path.join(BASE_DIR, 'out-of-domain', bin_name)
        input_dir = os.path.join(bin_dir, 'formatted_original_data')
        output_dir = os.path.join(bin_dir, 'finetuning_train_data')
        
        if not os.path.exists(input_dir):
            print(f"Warning: formatted_original_data directory not found in {bin_name}")
            continue
        
        print(f"\nProcessing {bin_name}")
        
        # Process train-set.json
        train_input = os.path.join(input_dir, 'train-set.json')
        if os.path.exists(train_input):
            print(f"Processing train-set.json in {bin_name}")
            generate_training_data(train_input, output_dir)
        else:
            print(f"Warning: train-set.json not found in {bin_name}")
        
        # Process test-set.json
        test_input = os.path.join(input_dir, 'test-set.json')
        if os.path.exists(test_input):
            print(f"Processing test-set.json in {bin_name}")
            generate_training_data(test_input, output_dir)
        else:
            print(f"Warning: test-set.json not found in {bin_name}")

def main():
    # Process in-domain data
    process_in_domain_directory()
    
    # Process out-of-domain data
    process_out_of_domain_directory()

if __name__ == '__main__':
    main()
