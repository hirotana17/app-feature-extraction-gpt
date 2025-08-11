import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Configuration
INPUT_FILE = "./data-gpt/T-FREX/out-of-domain/COMMUNICATION/feature_extracted_data/test-set-BYsg165a-para-coord-tmp1.json"

def load_final_decision_data(file_path: str) -> list:
    """Load the final decision feature extraction data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_evaluation_data(data: list) -> tuple:
    """
    Prepare data for evaluation by converting features lists to sets
    for case-insensitive matching evaluation
    """
    true_features = []
    agent1_features = []
    agent2_features = []
    final_features = []
    
    for review in data:
        # Convert feature lists to sets for case-insensitive matching
        true_set = set(feature.lower() for feature in review['output'])
        agent1_set = set(feature.lower() for feature in review['agent1_features'])
        agent2_set = set(feature.lower() for feature in review['agent2_features'])
        final_set = set(feature.lower() for feature in review['final_features'])
        
        true_features.append(true_set)
        agent1_features.append(agent1_set)
        agent2_features.append(agent2_set)
        final_features.append(final_set)
    
    return true_features, agent1_features, agent2_features, final_features

def calculate_metrics(true_features: list, predicted_features: list) -> dict:
    """
    Calculate precision, recall, and F1 score for feature extraction.
    """
    # Convert sets to lists for sklearn metrics
    true_list = [list(features) for features in true_features]
    pred_list = [list(features) for features in predicted_features]
    
    # Create MultiLabelBinarizer and fit it on all features
    mlb = MultiLabelBinarizer()
    all_features = set()
    for features in true_list + pred_list:
        all_features.update(features)
    mlb.fit([list(all_features)])
    
    # Transform the feature lists to binary arrays
    true_binary = mlb.transform(true_list)
    pred_binary = mlb.transform(pred_list)
    
    # Calculate metrics
    precision = precision_score(true_binary, pred_binary, average='micro', zero_division=0)
    recall = recall_score(true_binary, pred_binary, average='micro', zero_division=0)
    f1 = f1_score(true_binary, pred_binary, average='micro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_final_decision_results(file_path: str) -> dict:
    """
    Evaluate the final decision feature extraction results
    """
    # Load data
    data = load_final_decision_data(file_path)
    
    # Prepare data for evaluation
    true_features, agent1_features, agent2_features, final_features = prepare_evaluation_data(data)
    
    # Calculate metrics for all approaches
    agent1_metrics = calculate_metrics(true_features, agent1_features)
    agent2_metrics = calculate_metrics(true_features, agent2_features)
    final_metrics = calculate_metrics(true_features, final_features)
    
    # Combine results
    results = {
        'agent1_metrics': agent1_metrics,
        'agent2_metrics': agent2_metrics,
        'final_metrics': final_metrics,
        'total_reviews': len(data),
        'total_true_features': sum(len(features) for features in true_features),
        'total_agent1_features': sum(len(features) for features in agent1_features),
        'total_agent2_features': sum(len(features) for features in agent2_features),
        'total_final_features': sum(len(features) for features in final_features)
    }
    
    return results

def save_final_decision_evaluation_results(file_path: str, results: dict):
    """Save final decision evaluation results to a text file"""
    
    # Extract category and create output directory
    path_parts = file_path.split('/')
    category_index = path_parts.index('out-of-domain') + 1
    category = path_parts[category_index]
    
    base_dir = '/'.join(path_parts[:-2])
    evaluation_dir = os.path.join(base_dir, 'evaluation_result')
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Create output filename
    base_filename = os.path.basename(file_path).replace('.json', '')
    output_filename = f"{base_filename}-evaluation.txt"
    output_path = os.path.join(evaluation_dir, output_filename)
    
    # Prepare output text
    output_lines = []
    output_lines.append("Final Decision Feature Extraction Evaluation Results")
    output_lines.append("=" * 70)
    output_lines.append(f"Total Reviews: {results['total_reviews']}")
    output_lines.append(f"Total True Features: {results['total_true_features']}")
    output_lines.append(f"Total Agent 1 Features: {results['total_agent1_features']}")
    output_lines.append(f"Total Agent 2 Features: {results['total_agent2_features']}")
    output_lines.append(f"Total Final Features: {results['total_final_features']}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("AGENT 1 RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['agent1_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['agent1_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['agent1_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("AGENT 2 RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['agent2_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['agent2_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['agent2_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("FINAL DECISION AGENT (FINAL) RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['final_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['final_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['final_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("PERFORMANCE COMPARISON")
    output_lines.append("-" * 40)
    
    # Find best individual agent
    agent_scores = [
        ("Agent 1", results['agent1_metrics']['f1']),
        ("Agent 2", results['agent2_metrics']['f1'])
    ]
    best_agent = max(agent_scores, key=lambda x: x[1])
    
    output_lines.append(f"Best Individual Agent: {best_agent[0]} (F1: {best_agent[1]:.4f})")
    output_lines.append(f"Final Decision Agent F1: {results['final_metrics']['f1']:.4f}")
    
    final_decision_improvement = results['final_metrics']['f1'] - best_agent[1]
    output_lines.append(f"Final Decision vs Best Agent: {final_decision_improvement:+.4f}")
    
    # Calculate average of individual agents
    avg_agent_f1 = (results['agent1_metrics']['f1'] + results['agent2_metrics']['f1']) / 2
    output_lines.append(f"Average Individual Agent F1: {avg_agent_f1:.4f}")
    
    final_decision_vs_avg = results['final_metrics']['f1'] - avg_agent_f1
    output_lines.append(f"Final Decision vs Average Agent: {final_decision_vs_avg:+.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("DETAILED METRICS COMPARISON")
    output_lines.append("-" * 40)
    output_lines.append("Agent          | Precision | Recall   | F1 Score")
    output_lines.append("-" * 70)
    output_lines.append(f"Agent 1        | {results['agent1_metrics']['precision']:.4f}     | {results['agent1_metrics']['recall']:.4f}     | {results['agent1_metrics']['f1']:.4f}")
    output_lines.append(f"Agent 2        | {results['agent2_metrics']['precision']:.4f}     | {results['agent2_metrics']['recall']:.4f}     | {results['agent2_metrics']['f1']:.4f}")
    output_lines.append(f"Final Decision  | {results['final_metrics']['precision']:.4f}     | {results['final_metrics']['recall']:.4f}     | {results['final_metrics']['f1']:.4f}")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Final decision evaluation results saved to: {output_path}")
    return output_path

def main():
    # Check if file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        print("Please run the final decision feature extraction first.")
        return
    
    # Evaluate results
    results = evaluate_final_decision_results(INPUT_FILE)
    
    # Save results
    save_final_decision_evaluation_results(INPUT_FILE, results)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("FINAL DECISION EVALUATION SUMMARY")
    print("=" * 70)
    
    # Find best individual agent
    agent_scores = [
        ("Agent 1", results['agent1_metrics']['f1']),
        ("Agent 2", results['agent2_metrics']['f1'])
    ]
    best_agent = max(agent_scores, key=lambda x: x[1])
    
    print(f"Best Individual Agent: {best_agent[0]} (F1: {best_agent[1]:.4f})")
    print(f"Final Decision Agent F1: {results['final_metrics']['f1']:.4f}")
    
    final_decision_improvement = results['final_metrics']['f1'] - best_agent[1]
    print(f"Final Decision vs Best Agent: {final_decision_improvement:+.4f}")
    
    # Calculate average of individual agents
    avg_agent_f1 = (results['agent1_metrics']['f1'] + results['agent2_metrics']['f1']) / 2
    print(f"Average Individual Agent F1: {avg_agent_f1:.4f}")
    
    final_decision_vs_avg = results['final_metrics']['f1'] - avg_agent_f1
    print(f"Final Decision vs Average Agent: {final_decision_vs_avg:+.4f}")

if __name__ == '__main__':
    main() 