import json
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()

# Configuration
INPUT_DIR = './data-gpt/T-FREX/out-of-domain/PRODUCTIVITY/formatted_original_data'
INPUT_FILE = "./data-gpt/T-FREX/in-domain/bin0/feature_extracted_data/test-set-BYrc6PIY-para-coord-tmp1-dft.json"
OUTPUT_DIR = './data-gpt/T-FREX/out-of-domain/PRODUCTIVITY/feature_extracted_data'
MODEL_NAME = "ft:gpt-4.1-nano-2025-04-14:personal::BYkEvstb"
MODEL_DEFAULT = "gpt-4.1-nano-2025-04-14"
TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0.0

EXTRACT_PROMPT = """You are an expert at identifying app features from user reviews.

Your task:
1. Identify and extract the most significant app feature mentioned in the review
2. Extract only features that are explicitly mentioned in the text
3. Return one to three features consisting of one, two, or three words

Guidelines:
- Focus on concrete app functionalities (e.g., search, shopping cart, weather forecast alert)
- Do not infer features that are not directly mentioned
- Avoid general descriptions or sentiments
- Case sensitivity matters - extract features exactly as they appear in the review
- Ignore features that require more than three words

Output format:
- Return the features as a string with comma separated
- Example: search, shopping cart, weather forecast alert"""

def extract_features_agent_1(review_text: str) -> List[str]:
    """Agent 1: Extract features with temperature 0.5"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", EXTRACT_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(review_text=review_text))
        content = response.content.strip()
        
        features = parse_features_response(content)
        # print(f"Agent 1: {features}")
        return features
        
    except Exception as e:
        print(f"Error in Agent 1 extraction: {e}")
        return []

def extract_features_agent_2(review_text: str) -> List[str]:
    """Agent 2: Extract features with temperature 0.5"""
    llm = ChatOpenAI(model_name=MODEL_DEFAULT, temperature=TEMPERATURE)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", EXTRACT_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(review_text=review_text))
        content = response.content.strip()
        
        features = parse_features_response(content)
        # print(f"Agent 2: {features}")
        return features
        
    except Exception as e:
        print(f"Error in Agent 2 extraction: {e}")
        return []

def final_decision_agent(review_text: str, agent1_features: List[str], agent2_features: List[str]) -> List[str]:
    """Final decision agent that combines results from both agents"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=FINAL_TEMPERATURE)
    
    final_decision_prompt = f"""You are an expert at identifying app features from user reviews.
    These features are extracted by two different agents from the review.
    agent1_features: {agent1_features}
    agent2_features: {agent2_features}

    Your task:
    1. Review all extracted features from both agents and the review text
    2. Select the most significant app feature from the extracted features that mentioned in the review
    3. Extract only features that are explicitly mentioned in the text
    4. Return exactly one feature name consisting of one, two, or three words

    Guidelines:
    - Focus on concrete app functionalities (e.g., search, shopping cart, weather forecast alert)
    - Do not infer features that are not directly mentioned
    - Avoid general descriptions or sentiments
    - Case sensitivity matters - extract features exactly as they appear in the review
    - Ignore features that require more than three words

    Output format:
    - Return the final features as a JSON array of strings. Example: ["feature1", "feature2", "feature3"]"""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", final_decision_prompt),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            agent1_features=agent1_features,
            agent2_features=agent2_features
        ))
        content = response.content.strip()
        
        features = parse_features_response(content)
        
        return features
        
    except Exception as e:
        print(f"Error in final decision agent: {e}")
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

async def process_review_final_decision(review_text: str, ground_truth_features: List[str] = None) -> Dict:
    """Process a single review using two agents with final decision agent"""
    
    # Step 1: Run both agents in parallel
    print(f"Review: {review_text}")
    
    # Create tasks for parallel execution
    agent1_task = asyncio.create_task(
        asyncio.to_thread(extract_features_agent_1, review_text)
    )
    agent2_task = asyncio.create_task(
        asyncio.to_thread(extract_features_agent_2, review_text)
    )
    
    # Wait for both agents to complete
    agent1_features, agent2_features = await asyncio.gather(
        agent1_task, agent2_task
    )
    
    print(f"\nAgent Results:")
    print(f"  Agent 1: {agent1_features}")
    print(f"  Agent 2: {agent2_features}")
    
    # Step 2: Final decision agent combines results
    final_features = await asyncio.to_thread(
        final_decision_agent,
        review_text,
        agent1_features,
        agent2_features
    )

    print(f"Final Decision Result: {final_features}")
    
    return {
        'agent1_features': agent1_features,
        'agent2_features': agent2_features,
        'final_features': final_features
    }

async def process_reviews_final_decision(input_file: str, output_file: str):
    """Process all reviews using final decision agent workflow"""
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"\nProcessing {len(reviews)} reviews with final decision agent workflow...")
    
    for i, review in enumerate(reviews, 1):
        ground_truth = review.get('output', [])
        
        print(f"\n{'='*80}")
        print(f"Review [{i}/{len(reviews)}]")
        print(f"Ground Truth: {ground_truth}")
        
        # Process with final decision agents
        result = await process_review_final_decision(
            review['input'],
            ground_truth_features=ground_truth
        )
        
        # Store all results
        review['agent1_features'] = result['agent1_features']
        review['agent2_features'] = result['agent2_features']
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
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}-{model_suffix}-para-coord-tmp1-dft.json")
    
    # Execute final decision review processing
    asyncio.run(process_reviews_final_decision(input_file, output_file))

if __name__ == '__main__':
    main() 