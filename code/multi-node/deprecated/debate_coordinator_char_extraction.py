import json
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
INPUT_DIR = './data-gpt/T-FREX/out-of-domain/LIFESTYLE/formatted_original_data'
INPUT_FILE = 'test-set.json'
OUTPUT_DIR = './data-gpt/T-FREX/out-of-domain/LIFESTYLE/feature_extracted_data'
MODEL_NAME = "ft:gpt-4.1-nano-2025-04-14:personal::BZ6ub2vv"

# Agent A: Senior Business Analyst Perspective
SENIOR_BA_PROMPT = """You are an experienced Senior Business Analyst with extensive experience in feature extraction from user feedback and reviews. You have a positive, optimistic approach and are skilled at identifying a wide range of potential features from user comments.

Review text: {review_text}

Your task:
1. Extract app features from a senior business analyst's perspective
2. Focus on features that provide business value and user satisfaction
3. Consider both explicit and implicit feature requests
4. Keep feature names short and specific (1-3 words maximum)
5. Ensure every word in your feature names appears in the review text
6. MAXIMUM 3 features total

Business Analysis Guidelines:
- Look for features that create business value
- Consider user needs and market opportunities
- Focus on actionable and implementable features
- Pay attention to user feedback patterns
- Consider both current and future business needs

Return the features as a JSON array of strings. Example: ["feature1", "feature2", "feature3"]"""

# Agent B: Business Analyst Manager Perspective
BA_MANAGER_PROMPT = """You are a Business Analyst Manager with a critical and analytical perspective on feature extraction. You have high standards and apply rigorous criteria when evaluating feature suggestions.

Review text: {review_text}

Your task:
1. Extract app features from a business analyst manager's perspective
2. Focus on features that are well-justified and feasible
3. Consider implementation requirements and business impact
4. Keep feature names short and specific (1-3 words maximum)
5. Ensure every word in your feature names appears in the review text
6. MAXIMUM 3 features total

Manager Guidelines:
- Look for features with clear business justification
- Consider implementation feasibility and resource requirements
- Focus on ROI and business impact
- Pay attention to risk assessment and constraints
- Consider strategic alignment with business objectives

Return the features as a JSON array of strings. Example: ["feature1", "feature2", "feature3"]"""

# Debate Prompts
AGENT_A_RESPONSE_PROMPT = """You are Agent A (Senior Business Analyst). You have extracted these features: {agent_a_features}

Agent B (Business Analyst Manager) has responded with: {agent_b_response}

Agent C (Neutral Arbiter) has provided this feedback: {agent_c_feedback}

Your task:
1. Review Agent B's feedback and extracted features
2. Consider Agent C's neutral feedback and suggestions
3. Provide a constructive response addressing their points
4. Defend your business-focused perspective if needed
5. Acknowledge valid analytical points from Agent B
6. Consider Agent C's suggestions for consensus
7. Suggest potential compromises if there are disagreements
8. Keep your response professional and collaborative

Guidelines:
- Be respectful and constructive
- Focus on business value of your extracted features
- Consider feasibility mentioned by Agent B
- Take into account Agent C's neutral perspective
- Aim for consensus while maintaining business perspective
- If you agree with Agent B's points or Agent C's suggestions, acknowledge them

Respond with this JSON format:
{{
    "response_to_b": "Your response to Agent B's feedback and Agent C's suggestions",
    "revised_features": ["feature1", "feature2", "feature3"],
    "reasoning": "Why you made these changes or why you stand by your original features"
}}"""

AGENT_B_RESPONSE_PROMPT = """You are Agent B (Business Analyst Manager). Agent A (Senior Business Analyst) has shared their feature extraction and reasoning with you.

Agent A's features and explanation: {agent_a_response}

Agent C (Neutral Arbiter) has provided this feedback: {agent_c_feedback}

Your task:
1. Review Agent A's extracted features and reasoning from a manager's perspective
2. Consider Agent C's neutral feedback and suggestions (if any)
3. Extract your own features from the review text, focusing on analytical aspects
4. Provide a constructive response addressing Agent A's points
5. Take into account Agent C's suggestions for consensus (if provided)
6. Defend your analytical perspective if needed
7. Acknowledge valid business points from Agent A
8. Suggest potential compromises if there are disagreements
9. Keep your response professional and collaborative

Guidelines:
- Be respectful and constructive
- Focus on analytical rigor of your extracted features
- Consider business value mentioned by Agent A
- Take into account Agent C's neutral perspective (if feedback is provided)
- Aim for consensus while maintaining analytical perspective
- If you agree with Agent A's points or Agent C's suggestions, acknowledge them
- Extract features that are explicitly mentioned in the review text
- Keep feature names short and specific (1-3 words maximum)
- MAXIMUM 3 features total

Respond with this JSON format:
{{
    "response_to_a": "Your response to Agent A's features and reasoning, considering Agent C's feedback if provided",
    "revised_features": ["feature1", "feature2", "feature3"],
    "reasoning": "Why you chose these features and how you responded to Agent A's perspective and Agent C's suggestions"
}}"""

# Agent C: Neutral Arbiter
ARBITER_PROMPT = """You are Agent C, a neutral arbiter with expertise in business analysis. Your role is to make the final decision when Agents A and B cannot reach consensus.

Review text: {review_text}
Agent A (Senior Business Analyst) final features: {agent_a_final_features}
Agent B (Business Analyst Manager) final features: {agent_b_final_features}
Agent A's reasoning: {agent_a_reasoning}
Agent B's reasoning: {agent_b_reasoning}

Your task:
1. Evaluate both perspectives objectively
2. Consider the balance between business value and analytical rigor
3. Make a final decision on which features to include
4. Provide clear reasoning for your decision
5. Keep feature names short and specific (1-3 words maximum)
6. MAXIMUM 3 features total

Arbiter Guidelines:
- Consider both business and analytical perspectives equally
- Prioritize features that serve both business needs and analytical standards
- Remove features that are not explicitly mentioned in the review
- Ensure all selected features appear in the review text
- Provide balanced reasoning that acknowledges both viewpoints

Return the final decision with this JSON format:
{{
    "final_features": ["feature1", "feature2", "feature3"],
    "decision_reasoning": "Detailed explanation of why you chose these features",
    "business_considerations": "How business perspective influenced the decision",
    "analytical_considerations": "How analytical perspective influenced the decision"
}}"""

# Agent C: Neutral Arbiter (Between Iterations)
ARBITER_FEEDBACK_PROMPT = """You are Agent C, a neutral arbiter with expertise in business analysis. Your role is to provide constructive feedback between debate iterations to help Agents A and B reach consensus.

Review text: {review_text}
Agent A (Senior Business Analyst) features and reasoning: {agent_a_output}
Agent B (Business Analyst Manager) features and reasoning: {agent_b_output}

Your task:
1. Evaluate both perspectives objectively
2. Identify areas of agreement and disagreement
3. Provide constructive feedback to help bridge the gap
4. Suggest potential compromises or areas for discussion
5. Keep your feedback balanced and helpful

Arbiter Guidelines:
- Be neutral and constructive
- Acknowledge valid points from both perspectives
- Identify common ground and differences
- Suggest specific areas for discussion
- Help guide the conversation toward consensus
- Focus on the review text and extracted features

Return your feedback with this JSON format:
{{
    "feedback_to_agents": "Your constructive feedback to both agents",
    "areas_of_agreement": ["point1", "point2"],
    "areas_of_disagreement": ["point1", "point2"],
    "suggestions_for_consensus": "Specific suggestions to help reach agreement"
}}"""

def extract_features_agent_a(review_text: str) -> List[str]:
    """Agent A: Extract features from Senior Business Analyst perspective"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SENIOR_BA_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(review_text=review_text))
        content = response.content.strip()
        
        features = parse_features_response(content)
        return features
        
    except Exception as e:
        print(f"Error in Agent A extraction: {e}")
        return []

def extract_features_agent_b(review_text: str) -> List[str]:
    """Agent B: Extract features from Business Analyst Manager perspective"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", BA_MANAGER_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(review_text=review_text))
        content = response.content.strip()
        
        features = parse_features_response(content)
        return features
        
    except Exception as e:
        print(f"Error in Agent B extraction: {e}")
        return []

def agent_a_respond_to_b(review_text: str, agent_a_features: List[str], agent_b_response: str) -> Dict[str, Any]:
    """Agent A responds to Agent B's feedback"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", AGENT_A_RESPONSE_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            agent_a_features=agent_a_features,
            agent_b_response=agent_b_response
        ))
        content = response.content.strip()
        
        return parse_json_response(content)
        
    except Exception as e:
        print(f"Error in Agent A response: {e}")
        return {
            "response_to_b": "Unable to process response",
            "revised_features": agent_a_features,
            "reasoning": "Error occurred during processing"
        }

def agent_b_respond_to_a(review_text: str, agent_b_features: List[str], agent_a_response: str) -> Dict[str, Any]:
    """Agent B responds to Agent A's feedback"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    
    # If this is Agent B's first response (no features yet), use the initial extraction prompt
    if not agent_b_features:
        initial_prompt = """You are Agent B (Business Analyst Manager). Agent A (Senior Business Analyst) has shared their feature extraction and reasoning with you.

Agent A's perspective: {agent_a_response}

Your task:
1. Extract app features from a business analyst manager's perspective
2. Focus on features that are well-justified and feasible
3. Consider implementation requirements and business impact
4. Keep feature names short and specific (1-3 words maximum)
5. Ensure every word in your feature names appears in the review text
6. MAXIMUM 3 features total
7. Provide a constructive response to Agent A's perspective

Manager Guidelines:
- Look for features with clear business justification
- Consider implementation feasibility and resource requirements
- Focus on ROI and business impact
- Pay attention to risk assessment and constraints
- Consider strategic alignment with business objectives

Respond with this JSON format:
{{
    "response_to_a": "Your response to Agent A's perspective and your analytical reasoning",
    "revised_features": ["feature1", "feature2", "feature3"],
    "reasoning": "Why you chose these features from an analytical perspective"
}}"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", initial_prompt),
            ("human", f"Review text: {review_text}")
        ])
    else:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", AGENT_B_RESPONSE_PROMPT),
            ("human", f"Review text: {review_text}")
        ])
    
    try:
        if not agent_b_features:
            # First response - extract features and respond to A
            response = llm.invoke(prompt_template.format(
                review_text=review_text,
                agent_a_response=agent_a_response
            ))
        else:
            # Subsequent response - respond to A's feedback
            response = llm.invoke(prompt_template.format(
                review_text=review_text,
                agent_b_features=agent_b_features,
                agent_a_response=agent_a_response,
                agent_c_feedback="{}"  # Empty feedback for first iteration
            ))
        
        content = response.content.strip()
        
        parsed_response = parse_json_response(content)
        
        return parsed_response
        
    except Exception as e:
        print(f"Error in Agent B response: {e}")
        return {
            "response_to_a": "Unable to process response",
            "revised_features": agent_b_features,
            "reasoning": "Error occurred during processing"
        }

def agent_c_arbitrate(review_text: str, agent_a_final: Dict[str, Any], agent_b_final: Dict[str, Any]) -> Dict[str, Any]:
    """Agent C makes final decision"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.2)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ARBITER_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            agent_a_final_features=agent_a_final.get("revised_features", []),
            agent_b_final_features=agent_b_final.get("revised_features", []),
            agent_a_reasoning=agent_a_final.get("reasoning", ""),
            agent_b_reasoning=agent_b_final.get("reasoning", "")
        ))
        content = response.content.strip()
        
        return parse_json_response(content)
        
    except Exception as e:
        print(f"Error in Agent C arbitration: {e}")
        return {
            "final_features": [],
            "decision_reasoning": "Error occurred during arbitration",
            "business_considerations": "",
            "analytical_considerations": ""
        }

def agent_c_provide_feedback(review_text: str, agent_a_output: str, agent_b_output: str) -> Dict[str, Any]:
    """Agent C provides feedback between iterations"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ARBITER_FEEDBACK_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            agent_a_output=agent_a_output,
            agent_b_output=agent_b_output
        ))
        content = response.content.strip()
        
        return parse_json_response(content)
        
    except Exception as e:
        print(f"Error in Agent C feedback: {e}")
        return {
            "feedback_to_agents": "Unable to process feedback",
            "areas_of_agreement": [],
            "areas_of_disagreement": [],
            "suggestions_for_consensus": ""
        }

def agent_a_respond_to_b_and_c(review_text: str, agent_a_features: List[str], agent_b_response: str, agent_c_feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Agent A responds to Agent B's feedback and Agent C's feedback"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", AGENT_A_RESPONSE_PROMPT),
        ("human", f"Review text: {review_text}")
    ])
    
    try:
        response = llm.invoke(prompt_template.format(
            review_text=review_text,
            agent_a_features=agent_a_features,
            agent_b_response=agent_b_response,
            agent_c_feedback=json.dumps(agent_c_feedback, indent=2)  # Pass feedback as JSON string
        ))
        content = response.content.strip()
        
        return parse_json_response(content)
        
    except Exception as e:
        print(f"Error in Agent A response to B and C: {e}")
        return {
            "response_to_b": "Unable to process response",
            "revised_features": agent_a_features,
            "reasoning": "Error occurred during processing"
        }

def agent_b_respond_to_a_and_c(review_text: str, agent_b_features: List[str], agent_a_response: str, agent_c_feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Agent B responds to Agent A's feedback and Agent C's feedback"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    
    # If this is Agent B's first response (no features yet), use the initial extraction prompt
    if not agent_b_features:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", BA_MANAGER_PROMPT + "\n\nAfter extracting your features, respond to Agent A's perspective."),
            ("human", f"Review text: {review_text}\n\nAgent A's perspective: {agent_a_response}")
        ])
    else:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", AGENT_B_RESPONSE_PROMPT),
            ("human", f"Review text: {review_text}")
        ])
    
    try:
        if not agent_b_features:
            # First response - extract features and respond to A
            response = llm.invoke(prompt_template.format(
                review_text=review_text,
                agent_a_response=agent_a_response
            ))
        else:
            # Subsequent response - respond to A's feedback
            response = llm.invoke(prompt_template.format(
                review_text=review_text,
                agent_b_features=agent_b_features,
                agent_a_response=agent_a_response,
                agent_c_feedback=json.dumps(agent_c_feedback, indent=2) # Pass feedback as JSON string
            ))
        
        content = response.content.strip()
        
        return parse_json_response(content)
        
    except Exception as e:
        print(f"Error in Agent B response to A and C: {e}")
        return {
            "response_to_a": "Unable to process response",
            "revised_features": agent_b_features,
            "reasoning": "Error occurred during processing"
        }

def parse_features_response(content: str) -> List[str]:
    """Parse features from LLM response"""
    features = []
    
    # Remove markdown code blocks if present
    if content.startswith('```json'):
        content = content.replace('```json', '').replace('```', '').strip()
    elif content.startswith('```'):
        content = content.replace('```', '').strip()
    
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

def parse_json_response(content: str) -> Dict[str, Any]:
    """Parse JSON response from LLM"""
    try:
        # Try to find JSON in the response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = content[start_idx:end_idx]
            return json.loads(json_str)
        else:
            # If no JSON found, return as text
            return {"response": content}
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {"response": content, "error": "JSON parsing failed"}

def check_consensus(agent_a_features: List[str], agent_b_features: List[str]) -> bool:
    """Check if agents have reached consensus"""
    # Simple consensus check: if more than 50% of features overlap
    if not agent_a_features or not agent_b_features:
        return False
    
    common_features = set(agent_a_features) & set(agent_b_features)
    total_features = len(set(agent_a_features) | set(agent_b_features))
    
    if total_features == 0:
        return True
    
    consensus_ratio = len(common_features) / total_features
    return consensus_ratio >= 0.5

async def process_review_debate(review_text: str, ground_truth_features: List[str] = None) -> Dict[str, Any]:
    """Process a single review using the debate workflow"""
    
    print(f"\n{'='*60}")
    print(f"DEBATE WORKFLOW START")
    print(f"Review: {review_text[:80]}{'...' if len(review_text) > 80 else ''}")
    print(f"{'='*60}")
    
    # Iteration 1 Start
    print(f"\n📋 ITERATION 1")
    
    # Step 1-1: Agent A extracts features first and explains to B
    print(f"👤 Agent A (Senior Business Analyst) extracting features...")
    agent_a_initial = await asyncio.to_thread(extract_features_agent_a, review_text)
    agent_a_explanation = f"I extracted these features: {agent_a_initial}. As a Senior Business Analyst, I focused on business value, user satisfaction, and market opportunities that these features provide."
    print(f"   Features: {agent_a_initial}")
    
    # Step 1-2: Agent B responds to A's output (agreement or disagreement)
    print(f"👨‍💻 Agent B (Business Analyst Manager) responding...")
    agent_b_response_1 = await asyncio.to_thread(
        agent_b_respond_to_a, review_text, [], agent_a_explanation
    )
    agent_b_features_1 = agent_b_response_1.get('revised_features', [])
    print(f"   Features: {agent_b_features_1}")
    
    # Check consensus after iteration 1
    if check_consensus(agent_a_initial, agent_b_features_1):
        print(f"\n✅ CONSENSUS REACHED after Iteration 1!")
        final_features = list(set(agent_a_initial + agent_b_features_1))[:3]
        decision_reasoning = "Consensus reached through first iteration of debate between business and analytical perspectives."
        consensus_reached = True
        arbitration_required = False
    else:
        # Agent C provides feedback between iterations
        print(f"\n🤝 Agent C providing feedback between iterations...")
        agent_c_feedback = await asyncio.to_thread(
            agent_c_provide_feedback, review_text, agent_a_explanation, agent_b_response_1.get('response_to_a', '')
        )
        print(f"   Areas of agreement: {agent_c_feedback.get('areas_of_agreement', [])}")
        print(f"   Areas of disagreement: {agent_c_feedback.get('areas_of_disagreement', [])}")
        
        # Iteration 2 Start
        print(f"\n📋 ITERATION 2")
        
        # Step 2-1: Agent A responds to B's iteration 1 output and C's feedback
        print(f"👤 Agent A responding to B and C...")
        agent_a_response_2 = await asyncio.to_thread(
            agent_a_respond_to_b_and_c, review_text, agent_a_initial, agent_b_response_1.get('response_to_a', ''), agent_c_feedback
        )
        agent_a_features_2 = agent_a_response_2.get('revised_features', agent_a_initial)
        print(f"   Features: {agent_a_features_2}")
        
        # Step 2-2: Agent B responds to A's iteration 2 output and C's feedback
        print(f"👨‍💻 Agent B responding to A and C...")
        agent_b_response_2 = await asyncio.to_thread(
            agent_b_respond_to_a_and_c, review_text, agent_b_features_1, agent_a_response_2.get('response_to_b', ''), agent_c_feedback
        )
        agent_b_features_2 = agent_b_response_2.get('revised_features', agent_b_features_1)
        print(f"   Features: {agent_b_features_2}")
        
        # Check consensus after iteration 2
        if check_consensus(agent_a_features_2, agent_b_features_2):
            print(f"\n✅ CONSENSUS REACHED after Iteration 2!")
            final_features = list(set(agent_a_features_2 + agent_b_features_2))[:3]
            decision_reasoning = "Consensus reached through second iteration of debate between business and analytical perspectives."
            consensus_reached = True
            arbitration_required = False
        else:
            # Agent C final arbitration
            print(f"\n⚖️  NO CONSENSUS - Agent C arbitrating...")
            agent_c_decision = await asyncio.to_thread(
                agent_c_arbitrate, review_text, agent_a_response_2, agent_b_response_2
            )
            final_features = agent_c_decision.get('final_features', [])
            decision_reasoning = agent_c_decision.get('decision_reasoning', 'Arbitration required due to lack of consensus after 2 iterations.')
            consensus_reached = False
            arbitration_required = True
    
    # Prepare comprehensive result
    result = {
        'initial_features': {
            'agent_a_business': agent_a_initial,
            'agent_b_analytical': agent_b_features_1 if 'agent_b_features_1' in locals() else []
        },
        'debate_rounds': {
            'iteration_1': {
                'step_1_1_agent_a': {
                    'response': agent_a_explanation
                },
                'step_1_2_agent_b': {
                    'response': agent_b_response_1.get('response_to_a', '') if 'agent_b_response_1' in locals() else ''
                }
            }
        },
        'final_features': final_features,
        'decision_reasoning': decision_reasoning,
        'consensus_reached': consensus_reached,
        'arbitration_required': arbitration_required
    }
    
    # Add Agent C feedback if it occurred
    if 'agent_c_feedback' in locals():
        result['agent_c_feedback'] = agent_c_feedback
    
    # Add iteration 2 details if it occurred
    if 'agent_a_response_2' in locals():
        result['debate_rounds']['iteration_2'] = {
            'step_2_1_agent_a': {
                'response': agent_a_response_2.get('response_to_b', '')
            },
            'step_2_2_agent_b': {
                'response': agent_b_response_2.get('response_to_a', '')
            }
        }
    
    # Add arbitration details if it occurred
    if arbitration_required and 'agent_c_decision' in locals():
        result['arbitration_details'] = agent_c_decision
    
    print(f"\n🎯 FINAL RESULT")
    print(f"   Features: {final_features}")
    print(f"   Consensus: {'Yes' if consensus_reached else 'No'}")
    print(f"   Arbitration: {'Required' if arbitration_required else 'Not needed'}")
    print(f"{'='*60}")
    
    return result

async def process_reviews_debate(input_file: str, output_file: str):
    """Process all reviews using the debate workflow"""
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"\n🚀 Processing {len(reviews)} reviews with debate workflow...")
    
    for i, review in enumerate(reviews, 1):
        ground_truth = review.get('output', [])
        
        print(f"\n📝 Review [{i}/{len(reviews)}] - Ground Truth: {ground_truth}")
        
        # Process with debate workflow
        result = await process_review_debate(
            review['input'],
            ground_truth_features=ground_truth
        )
        
        # Store results in the same format as existing system
        review['initial_features'] = result['initial_features']['agent_a_business']  # Agent A's initial extraction
        # Keep original output (ground truth) unchanged
        # review['output'] = result['final_features']  # Don't overwrite ground truth
        review['refined_features'] = result['final_features']  # Final result after debate
        
        # Add debate-specific information
        review['debate_workflow'] = {
            'initial_extraction': result['initial_features'],
            'debate_rounds': result['debate_rounds'],
            'consensus_reached': result['consensus_reached'],
            'arbitration_required': result['arbitration_required'],
            'decision_reasoning': result['decision_reasoning']
        }
        
        if result.get('arbitration_details'):
            review['debate_workflow']['arbitration_details'] = result['arbitration_details']
        
        # Add feedback structure to match existing format
        review['feedback'] = {
            'missing_features': [],
            'incorrect_features': [],
            'suggestions': f"Debate workflow completed. {result['decision_reasoning']}",
            'reasoning': result['decision_reasoning']
        }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Completed! Results saved to {output_file}")

def main():
    # Generate output filename
    model_suffix = MODEL_NAME.split('::')[-1] if '::' in MODEL_NAME else MODEL_NAME.replace(':', '-').replace('/', '-')
    input_file = os.path.join(INPUT_DIR, INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.basename(input_file).replace('.json', '')
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}-{model_suffix}-debate-coord-char.json")
    
    # Execute debate review processing
    asyncio.run(process_reviews_debate(input_file, output_file))

if __name__ == '__main__':
    main() 