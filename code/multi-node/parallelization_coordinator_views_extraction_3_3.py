import json
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
INPUT_DIR = './data-gpt/T-FREX/out-of-domain/TRAVEL_AND_LOCAL/formatted_original_data'
INPUT_FILE = 'test-set.json'
OUTPUT_DIR = './data-gpt/T-FREX/out-of-domain/TRAVEL_AND_LOCAL/feature_extracted_data'
MODEL_NAME = "ft:gpt-4.1-nano-2025-04-14:personal::BZ4Ybrkd"

# Agent 1: Communication/Social Feature Extractor
COMMUNICATION_SOCIAL_PROMPT = """You are an expert at identifying communication and social features from user reviews. Focus specifically on features related to communication, messaging, social networking, and user interaction.

Review text: {review_text}

Your tasks:
1. Extract ONLY communication and social-related features
2. Focus on features related to communication, messaging, social networking, and user interaction
3. Ignore productivity, technical, or data management features
4. Keep feature names short and specific (1-3 words maximum)
5. Ensure every word in your feature names appears in the review text
6. MAXIMUM 2 features total

Guidelines:
- Only extract features that are explicitly mentioned in the review
- Focus on communication and social aspects
- If no communication/social features are mentioned, return empty array

Return the features as a JSON array of strings. Example: ["feature1", "feature2"]"""

# Agent 2: Productivity Feature Extractor
PRODUCTIVITY_PROMPT = """You are an expert at identifying productivity and task management features from user reviews. Focus specifically on features related to productivity, task management, planning, and organization.

Review text: {review_text}

Your tasks:
1. Extract ONLY productivity and task management features
2. Focus on features related to productivity, task management, planning, and organization
3. Ignore communication, technical, or data management features
4. Keep feature names short and specific (1-3 words maximum)
5. Ensure every word in your feature names appears in the review text
6. MAXIMUM 2 features total

Guidelines:
- Only extract features that are explicitly mentioned in the review
- Focus on productivity and organization aspects
- If no productivity features are mentioned, return empty array

Return the features as a JSON array of strings. Example: ["feature1", "feature2"]"""

# Agent 3: Technical/Data Feature Extractor
TECHNICAL_DATA_PROMPT = """You are an expert at identifying technical and data management features from user reviews. Focus specifically on features related to data storage, synchronization, security, technical capabilities, and system features.

Review text: {review_text}

Your tasks:
1. Extract ONLY technical and data management features
2. Focus on features related to data storage, synchronization, security, technical capabilities, and system features
3. Ignore communication, social, or productivity features
4. Keep feature names short and specific (1-3 words maximum)
5. Ensure every word in your feature names appears in the review text
6. MAXIMUM 2 features total

Guidelines:
- Only extract features that are explicitly mentioned in the review
- Focus on technical and data management aspects
- If no technical/data features are mentioned, return empty array

Return the features as a JSON array of strings. Example: ["feature1", "feature2"]"""

# Agent 4: Coordinator/Integrator
COORDINATOR_PROMPT = """You are an expert coordinator that combines and refines feature extraction results from multiple specialized agents. Your job is to create a final, comprehensive list of features.

Review text: {review_text}
Communication/Social features: {communication_features}
Productivity features: {productivity_features}
Technical/Data features: {technical_features}

Your tasks:
1. Review all extracted features from the three specialized agents
2. Remove any duplicates or overlapping features
3. Ensure all features are actually mentioned in the review text
4. Combine the best features from each category
5. Keep feature names short and specific (1-3 words maximum)
6. MAXIMUM 3 features total

Guidelines:
- Only include features that are explicitly mentioned in the review
- Prioritize the most important and clearly mentioned features
- Remove any features that are not actually in the review text
- Ensure feature names are consistent and clear
- If multiple agents extracted the same feature, keep only one instance

Return the final features as a JSON array of strings. Example: ["feature1", "feature2", "feature3"]"""

def extract_communication_social_features(review_text: str) -> List[str]:
    """Agent 1: Extract communication and social features"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", COMMUNICATION_SOCIAL_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(review_text=review_text))
        content = response.content.strip()
        
        features = parse_features_response(content)
        print(f"Communication/Social Agent: {features}")
        return features
        
    except Exception as e:
        print(f"Error in communication/social extraction: {e}")
        return []

def extract_productivity_features(review_text: str) -> List[str]:
    """Agent 2: Extract productivity features"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", PRODUCTIVITY_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(review_text=review_text))
        content = response.content.strip()
        
        features = parse_features_response(content)
        print(f"Productivity Agent: {features}")
        return features
        
    except Exception as e:
        print(f"Error in productivity extraction: {e}")
        return []

def extract_technical_data_features(review_text: str) -> List[str]:
    """Agent 3: Extract technical and data features"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", TECHNICAL_DATA_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(review_text=review_text))
        content = response.content.strip()
        
        features = parse_features_response(content)
        print(f"Technical/Data Agent: {features}")
        return features
        
    except Exception as e:
        print(f"Error in technical/data extraction: {e}")
        return []

def coordinate_final_features(review_text: str, communication_features: List[str], 
                           productivity_features: List[str], technical_features: List[str]) -> List[str]:
    """Agent 4: Coordinate and combine all extracted features"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", COORDINATOR_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            communication_features=communication_features,
            productivity_features=productivity_features,
            technical_features=technical_features
        ))
        content = response.content.strip()
        
        features = parse_features_response(content)
        print(f"Coordinator Agent: {features}")
        return features
        
    except Exception as e:
        print(f"Error in coordination: {e}")
        return []

def parse_features_response(content: str) -> List[str]:
    """Parse features from LLM response"""
    features = []
    
    # 1. Try JSON array format
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
    
    return features

async def process_review_parallel(review_text: str, ground_truth_features: List[str] = None) -> Dict:
    """Process a single review using parallel agents"""
    
    # Step 1: Run all three specialized agents in parallel
    print(f"\nProcessing review with parallel agents...")
    print(f"Text: {review_text[:100]}...")
    
    # Create tasks for parallel execution
    communication_task = asyncio.create_task(
        asyncio.to_thread(extract_communication_social_features, review_text)
    )
    productivity_task = asyncio.create_task(
        asyncio.to_thread(extract_productivity_features, review_text)
    )
    technical_task = asyncio.create_task(
        asyncio.to_thread(extract_technical_data_features, review_text)
    )
    
    # Wait for all agents to complete
    communication_features, productivity_features, technical_features = await asyncio.gather(
        communication_task, productivity_task, technical_task
    )
    
    print(f"\nAgent Results:")
    print(f"  Communication/Social: {communication_features}")
    print(f"  Productivity: {productivity_features}")
    print(f"  Technical/Data: {technical_features}")
    
    # Step 2: Coordinate and combine results
    final_features = await asyncio.to_thread(
        coordinate_final_features,
        review_text,
        communication_features,
        productivity_features,
        technical_features
    )
    
    return {
        'communication_features': communication_features,
        'productivity_features': productivity_features,
        'technical_features': technical_features,
        'final_features': final_features
    }

async def process_reviews_parallel(input_file: str, output_file: str):
    """Process all reviews using parallel agent workflow"""
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"\nProcessing {len(reviews)} reviews with parallel agent workflow...")
    
    for i, review in enumerate(reviews, 1):
        ground_truth = review.get('output', [])
        
        print(f"\n{'='*80}")
        print(f"Review [{i}/{len(reviews)}]")
        print(f"Ground Truth: {ground_truth}")
        
        # Process with parallel agents
        result = await process_review_parallel(
            review['input'],
            ground_truth_features=ground_truth
        )
        
        # Store all results
        review['communication_features'] = result['communication_features']
        review['productivity_features'] = result['productivity_features']
        review['technical_features'] = result['technical_features']
        review['final_features'] = result['final_features']
        
        print(f"Final Features: {result['final_features']}")
        print(f"{'='*80}")
    
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
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}-{model_suffix}-para-coord-views.json")
    
    # Execute parallel review processing
    asyncio.run(process_reviews_parallel(input_file, output_file))

if __name__ == '__main__':
    main()
