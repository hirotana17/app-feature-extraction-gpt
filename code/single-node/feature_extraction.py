import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import prompt

# Load environment variables
load_dotenv()

# Configuration
INPUT_DIR = './data/in-domain/bin0/formatted_original_data'
INPUT_FILE = 'test-set.json'
OUTPUT_DIR = './data/in-domain/bin0/feature_extracted_data'
MODEL_NAME = "ft:gpt-4.1-nano-2025-04-14:personal::BZ4Ybrkd"
SYSTEM_PROMPT = prompt.PROMPTFT1
PROMPT_NAME = "promptft1"

class FeatureExtraction(BaseModel):
    """Schema for feature extraction output"""
    features: List[str] = Field(description="List of feature names extracted from the review")

def extract_features(review_text, review_number=None, ground_truth_features=None):
    """Extract features from a review text"""
    # Initialize LLM
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # Define prompt template (without parser)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        raw_response = llm.invoke(prompt_template.format(review_text=review_text))
        features = []
        content = raw_response.content.strip()
        
        # Try to parse as JSON if it looks like JSON
        if (content.startswith('[') and content.endswith(']')) or (content.startswith('{') and content.endswith('}')):
            try:
                parsed_data = json.loads(content)

                if isinstance(parsed_data, list):
                    features = parsed_data
                elif isinstance(parsed_data, dict) and 'features' in parsed_data:
                    features = parsed_data['features']
            except json.JSONDecodeError:
                pass
                
        # If JSON parsing didn't work or yield results, try line-by-line parsing
        if not features:
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    features.append(line.lstrip('- *').strip())
        
        # Display results
        review_num_str = f"[{review_number}] " if review_number is not None else ""
        ground_truth_str = f" | GT: {ground_truth_features}" if ground_truth_features else ""
        print(f"{review_num_str}Review: {review_text[:100]}... | Extracted: {features}{ground_truth_str}")
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return []

def process_reviews(input_file, output_file):
    """Process reviews and extract features"""
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"\nProcessing {len(reviews)} reviews...")
    
    # Process each review
    for i, review in enumerate(reviews, 1):
        # Add review number
        review['review_number'] = i
        
        # Extract features
        ground_truth = review.get('output', [])
        extracted_features = extract_features(
            review['input'],
            review_number=i,
            ground_truth_features=ground_truth
        )
        
        # Add extracted features to the review
        review['extracted_features'] = extracted_features
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted. Results saved to {output_file}")

def main():
    # Generate output filename
    model_suffix = MODEL_NAME[-8:]  # Get last 8 characters
    input_file = os.path.join(INPUT_DIR, INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.basename(input_file).replace('.json', '')
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}-extracted-{PROMPT_NAME}-{model_suffix}-singleagent.json")
    
    # Execute review processing
    process_reviews(input_file, output_file)

if __name__ == '__main__':
    main()
