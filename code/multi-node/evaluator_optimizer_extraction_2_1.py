import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
INPUT_DIR = './data-gpt/T-FREX/out-of-domain/TOOLS/formatted_original_data'
INPUT_FILE = 'test-set.json'
OUTPUT_DIR = './data-gpt/T-FREX/out-of-domain/TOOLS/feature_extracted_data'
MODEL_NAME = "ft:gpt-4.1-nano-2025-04-14:personal::BYsG4orf"

# Extract Agent Prompt (Same with the PROMPTFT1 <- Fine-tuning prompt)
EXTRACT_PROMPT = """You are an expert at identifying app features from user reviews.

Your task:
1. Identify and extract the most significant app feature mentioned in the review
2. Extract only features that are explicitly mentioned in the text
3. Return exactly one feature name consisting of one, two, or three words

Guidelines:
- Focus on concrete app functionalities (e.g., search, shopping cart, weather forecast alert)
- Do not infer features that are not directly mentioned
- Avoid general descriptions or sentiments
- Case sensitivity matters - extract features exactly as they appear in the review
- Ignore features that require more than three words

Output format:
- Return the features as a string with comma separated
- Example: search, shopping cart, weather forecast alert"""

# Feedback Agent Prompt
FEEDBACK_PROMPT = """You are an expert reviewer of feature extraction results. Your job is to analyze the extracted features and provide constructive feedback.

Review text: {review_text}
Extracted features: {extracted_features}

Your tasks:
1. Analyze the extracted features for accuracy and completeness
2. Identify any missing important features that are clearly mentioned in the review
3. Point out any incorrect or irrelevant features
4. Provide specific suggestions for improvement

Guidelines:
- Be constructive and specific in your feedback
- Focus on features that are clearly mentioned in the review
- Consider the context and importance of each feature
- Suggest improvements that would make the extraction more accurate
- Only suggest features that are explicitly mentioned in the review text

Return the result as a JSON object with these fields:
- missing_features: array of important features that were missed
- incorrect_features: array of features that should be removed
- suggestions: string with specific improvement suggestions
- reasoning: string explaining your analysis"""

# Refined Extractor Prompt
REFINED_EXTRACTOR_PROMPT = """You are an expert at identifying app features from user reviews. You have received feedback on your previous extraction and should improve your results.

Review text: {review_text}
Previous extraction: {previous_features}
Feedback: {feedback}

Your tasks:
1. Consider the feedback provided
2. Extract the most accurate and relevant features
3. Address any issues mentioned in the feedback
4. Ensure all features are actually mentioned in the review

Guidelines:
- Use the feedback to improve your extraction
- Focus on features that are explicitly mentioned in the review
- Keep feature names short and specific (1-3 words maximum)
- Ensure every word in your feature names appears in the review text, case sensitivity matters
- MAXIMUM 2 features total

Return the features as a JSON array of strings. Example: ["feature1", "feature2"]"""

def extract_features_strict(review_text, ground_truth_features=None):
    """Extractor Agent: Extract features using the existing strict approach"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # Fine-tuned model expects simple array format, not structured output
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", EXTRACT_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        raw_response = llm.invoke(prompt_template.format(review_text=review_text))
        content = raw_response.content.strip()
        
        features = []
        
        # Parse response - try multiple formats
        # 1. Try JSON array format (expected from fine-tuned model)
        if (content.startswith('[') and content.endswith(']')) or (content.startswith('{') and content.endswith('}')):
            try:
                parsed_data = json.loads(content)
                if isinstance(parsed_data, list):
                    features = parsed_data
                elif isinstance(parsed_data, dict) and 'features' in parsed_data:
                    features = parsed_data['features']
            except json.JSONDecodeError:
                pass
        
        # 2. If no features found, try comma-separated format
        if not features:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if ',' in line and not line.startswith('[') and not line.startswith('{'):
                    potential_features = [f.strip().strip('"').strip("'") for f in line.split(',')]
                    features.extend([f for f in potential_features if f and len(f) > 1])
        
        # 3. If still no features, try bullet point format
        if not features:
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                    feature = line.lstrip('- *•').strip()
                    if feature and len(feature) > 1:
                        features.append(feature)
        
        # Remove duplicates and clean up
        features = list(set([f.strip() for f in features if f.strip() and len(f.strip()) > 1]))
        
        # Display results
        print(f"Strict Extractor: {features}\n")
        
        return features
        
    except Exception as e:
        print(f"Error in strict extraction: {e}")
        return []

def provide_feedback(review_text, extracted_features):
    """Feedback Agent: Analyze extracted features and provide feedback"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert reviewer of feature extraction results."),
        ("human", FEEDBACK_PROMPT)
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            extracted_features=extracted_features
        ))
        content = response.content.strip()
        
        # Parse JSON response
        try:
            parsed_data = json.loads(content)
            missing_features = parsed_data.get('missing_features', [])
            incorrect_features = parsed_data.get('incorrect_features', [])
            suggestions = parsed_data.get('suggestions', '')
            reasoning = parsed_data.get('reasoning', '')
        except json.JSONDecodeError:
            missing_features = []
            incorrect_features = []
            suggestions = 'Failed to parse feedback'
            reasoning = 'JSON parsing error'
        
        # Display feedback
        print(f"Feedback:")
        print(f"  Missing: {missing_features}")
        print(f"  Incorrect: {incorrect_features}")
        if suggestions:
            print(f"  Suggestions: {suggestions}")
        if reasoning:
            print(f"  Reasoning: {reasoning}")
        
        return {
            'missing_features': missing_features,
            'incorrect_features': incorrect_features,
            'suggestions': suggestions,
            'reasoning': reasoning
        }
    except Exception as e:
        print(f"Error providing feedback: {e}")
        return {
            'missing_features': [],
            'incorrect_features': [],
            'suggestions': 'Error occurred during feedback generation',
            'reasoning': str(e)
        }

def extract_features_refined(review_text, previous_features, feedback):
    """Refined Extractor Agent: Extract features with feedback consideration"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", REFINED_EXTRACTOR_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            previous_features=previous_features,
            feedback=feedback
        ))
        content = response.content.strip()
        
        features = []
        
        # Parse response - try multiple formats
        # 1. Try JSON array format (expected from fine-tuned model)
        if (content.startswith('[') and content.endswith(']')) or (content.startswith('{') and content.endswith('}')):
            try:
                parsed_data = json.loads(content)
                if isinstance(parsed_data, list):
                    features = parsed_data
                elif isinstance(parsed_data, dict) and 'features' in parsed_data:
                    features = parsed_data['features']
            except json.JSONDecodeError:
                pass
        
        # 2. If no features found, try comma-separated format
        if not features:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if ',' in line and not line.startswith('[') and not line.startswith('{'):
                    potential_features = [f.strip().strip('"').strip("'") for f in line.split(',')]
                    features.extend([f for f in potential_features if f and len(f) > 1])
        
        # 3. If still no features, try bullet point format
        if not features:
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                    feature = line.lstrip('- *•').strip()
                    if feature and len(feature) > 1:
                        features.append(feature)
        
        # Remove duplicates and clean up
        features = list(set([f.strip() for f in features if f.strip() and len(f.strip()) > 1]))
        
        # Display results
        print(f"\nRefined Extractor: {features}\n")
        
        return features
        
    except Exception as e:
        print(f"Error in refined extraction: {e}")
        return []

def process_with_feedback(review_text, ground_truth_features):
    """Process review with feedback-based refinement approach"""
    # Step 1: Initial extraction
    initial_features = extract_features_strict(review_text, ground_truth_features)
    
    # Step 2: Provide feedback (without ground truth)
    feedback_result = provide_feedback(review_text, initial_features)
    
    # Step 3: Refined extraction based on feedback
    refined_features = extract_features_refined(review_text, initial_features, feedback_result)
    
    return {
        'initial_features': initial_features,
        'feedback': feedback_result,
        'refined_features': refined_features,
    }

def process_reviews_enhanced(input_file, output_file):
    """Process reviews with feedback-based refinement approach"""
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"\nProcessing {len(reviews)} reviews with feedback-based refinement approach...")
    
    for i, review in enumerate(reviews, 1):
        ground_truth = review.get('output', [])
        
        # Display review header with number
        print(f"Review [{i}]\n")
        print(f"Text: {review['input']}\n")
        print(f"Ground Truth: {ground_truth}\n")
        
        # Process with feedback-based refinement
        result = process_with_feedback(
            review['input'],
            ground_truth_features=ground_truth
        )
        
        # Store all results
        review['initial_features'] = result['initial_features']
        review['feedback'] = result['feedback']
        review['refined_features'] = result['refined_features']
        print("=" * 80)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted. Results saved to {output_file}")

def main():
    # Generate output filename
    model_suffix = MODEL_NAME[-8:]
    input_file = os.path.join(INPUT_DIR, INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.basename(input_file).replace('.json', '')
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}-{model_suffix}-eval-opti.json")
    
    # Execute enhanced review processing
    process_reviews_enhanced(input_file, output_file)

if __name__ == '__main__':
    main() 