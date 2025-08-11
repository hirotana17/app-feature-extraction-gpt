import json
import os
from typing import Dict, List
import ast

def read_conll_file(file_path: str) -> List[Dict]:
    """
    Read a CoNLL format file and return document-level data
    
    Args:
        file_path (str): Path to the CoNLL file
    
    Returns:
        List[Dict]: List of document data
    """
    documents = []
    current_doc = None
    current_feature = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Start of a new document
            if line.startswith('<doc'):
                if current_doc is not None:
                    # Add the last feature from the previous document
                    if current_feature:
                        current_doc['output'].append(' '.join(current_feature))
                        current_feature = []
                    documents.append(current_doc)
                
                current_doc = {
                    'id': '',
                    'package_name': '',
                    'app_name': '',
                    'app_category': [],
                    'google_play_category': [],
                    'input': '',
                    'output': []
                }
                
                # Extract metadata
                for attr in ['id', 'package_name', 'app_name', 'app_category', 'google_play_category']:
                    if f'{attr}=' in line:
                        value = line.split(f'{attr}="')[1].split('"')[0]
                        if attr in ['app_category', 'google_play_category']:
                            # Convert string representation of list to actual list
                            try:
                                value = ast.literal_eval(value)
                            except:
                                value = [value]
                        current_doc[attr] = value
                continue
            
            # End of document
            if line == '</doc>':
                # Add the last feature
                if current_feature:
                    current_doc['output'].append(' '.join(current_feature))
                    current_feature = []
                continue
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse tab-separated data
            parts = line.split('\t')
            if len(parts) >= 11:  # Ensure required columns exist
                original_word = parts[1]  # Original text
                label = parts[10]  # Label
                
                # Build text using original words
                current_doc['input'] += original_word + ' '
                
                # Extract feature names using original text
                if label == 'B-feature':
                    # Save previous feature if exists
                    if current_feature:
                        current_doc['output'].append(' '.join(current_feature))
                    current_feature = [original_word]
                elif label == 'I-feature' and current_feature:
                    current_feature.append(original_word)
                elif label == 'O' and current_feature:
                    # End of feature
                    current_doc['output'].append(' '.join(current_feature))
                    current_feature = []
    
    # Add the last document
    if current_doc is not None:
        if current_feature:
            current_doc['output'].append(' '.join(current_feature))
        documents.append(current_doc)
    
    return documents

def save_json(data: List[Dict], output_file: str):
    """
    Save data as a JSON file
    
    Args:
        data (List[Dict]): Data to save
        output_file (str): Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_file(input_file: str, output_file: str):
    """
    Process a single CoNLL file and save as JSON
    
    Args:
        input_file (str): Path to input CoNLL file
        output_file (str): Path to output JSON file
    """
    # Read data from CoNLL file
    documents = read_conll_file(input_file)
    
    # Check for duplicates
    review_ids = [doc['id'] for doc in documents]
    unique_ids = set(review_ids)
    duplicate_ids = [id for id in set(review_ids) if review_ids.count(id) > 1]
    
    print(f"  Total reviews: {len(documents)}")
    print(f"  Unique reviews: {len(unique_ids)}")
    if duplicate_ids:
        print(f"  Found {len(duplicate_ids)} duplicate reviews")
        print(f"  First few duplicate IDs: {duplicate_ids[:5]}")
    
    # Remove trailing whitespace from text in each document
    for doc in documents:
        doc['input'] = doc['input'].strip()
    
    # Save as JSON file
    save_json(documents, output_file)
    print(f"  Output saved to {output_file}")

def process_in_domain_directory(domain_dir: str):
    """
    Process all bin directories within the in-domain directory
    
    Args:
        domain_dir (str): Path to the in-domain directory
    """
    print("Processing in-domain data...")
    
    # Process bin0 to bin9
    for bin_num in range(0, 10):
        bin_dir = os.path.join(domain_dir, f'bin{bin_num}')
        
        # Skip if bin directory doesn't exist
        if not os.path.exists(bin_dir):
            continue
            
        output_dir = os.path.join(bin_dir, 'formatted_original_data')
        input_dir = os.path.join(bin_dir, 'original_data')
        
        print(f"Processing directory: bin{bin_num}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Warning: original_data directory not found in bin{bin_num}")
            continue
        
        # Process train-set.txt
        train_input = os.path.join(input_dir, 'train-set.txt')
        train_output = os.path.join(output_dir, 'train-set.json')
        if os.path.exists(train_input):
            print(f"Processing train-set.txt in bin{bin_num}")
            process_file(train_input, train_output)
        else:
            print(f"Warning: train-set.txt not found in bin{bin_num}")
        
        # Process test-set.txt
        test_input = os.path.join(input_dir, 'test-set.txt')
        test_output = os.path.join(output_dir, 'test-set.json')
        if os.path.exists(test_input):
            print(f"Processing test-set.txt in bin{bin_num}")
            process_file(test_input, test_output)
        else:
            print(f"Warning: test-set.txt not found in bin{bin_num}")
        
        print()  # Add an empty line for better readability

def process_out_of_domain_directory(domain_dir: str):
    """
    Process all category directories within the out-of-domain directory
    
    Args:
        domain_dir (str): Path to the out-of-domain directory
    """
    print("Processing out-of-domain data...")
    
    # Specific bin directories for out-of-domain
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
        bin_dir = os.path.join(domain_dir, bin_name)
        
        # Skip if category directory doesn't exist
        if not os.path.exists(bin_dir):
            continue
            
        output_dir = os.path.join(bin_dir, 'formatted_original_data')
        input_dir = os.path.join(bin_dir, 'original_data')
        
        print(f"Processing directory: {bin_name}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Warning: original_data directory not found in {bin_name}")
            continue
        
        # Process train-set.txt
        train_input = os.path.join(input_dir, 'train-set.txt')
        train_output = os.path.join(output_dir, 'train-set.json')
        if os.path.exists(train_input):
            print(f"Processing train-set.txt in {bin_name}")
            process_file(train_input, train_output)
        else:
            print(f"Warning: train-set.txt not found in {bin_name}")
        
        # Process test-set.txt
        test_input = os.path.join(input_dir, 'test-set.txt')
        test_output = os.path.join(output_dir, 'test-set.json')
        if os.path.exists(test_input):
            print(f"Processing test-set.txt in {bin_name}")
            process_file(test_input, test_output)
        else:
            print(f"Warning: test-set.txt not found in {bin_name}")
        
        print()  # Add an empty line for better readability

def main():
    # Base directory
    base_dir = './data'
    
    # Process in-domain data
    in_domain_dir = os.path.join(base_dir, 'in-domain')
    process_in_domain_directory(in_domain_dir)
    
    # Process out-of-domain data
    out_domain_dir = os.path.join(base_dir, 'out-of-domain')
    process_out_of_domain_directory(out_domain_dir)

if __name__ == '__main__':
    main()
