import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Configuration
INPUT_FILE = "./data-gpt/T-FREX/out-of-domain/TOOLS/feature_extracted_data/test-set-BYsG4orf-eval-opti.json"

def load_multiagent_data(file_path: str) -> list:
    """Load the multi-agent feature extraction data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_evaluation_data(data: list) -> tuple:
    """
    Prepare data for evaluation by converting features lists to sets
    for case-insensitive exact matching evaluation
    """
    true_features = []
    initial_features = []
    refined_features = []
    
    for review in data:
        # Convert feature lists to lowercase sets for case-insensitive matching
        true_set = set(feature.lower() for feature in review['output'])
        initial_set = set(feature.lower() for feature in review['initial_features'])
        refined_set = set(feature.lower() for feature in review['refined_features'])
        
        true_features.append(true_set)
        initial_features.append(initial_set)
        refined_features.append(refined_set)
    
    return true_features, initial_features, refined_features

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



def evaluate_feedback_refinement_results(file_path: str) -> dict:
    """
    Evaluate the feedback-based refinement feature extraction results
    """
    # Load data
    data = load_multiagent_data(file_path)
    
    # Prepare data for evaluation
    true_features, initial_features, refined_features = prepare_evaluation_data(data)
    
    # Calculate metrics for all approaches
    initial_metrics = calculate_metrics(true_features, initial_features)
    refined_metrics = calculate_metrics(true_features, refined_features)
    
    # Combine results
    results = {
        'initial_metrics': initial_metrics,
        'refined_metrics': refined_metrics,
        'total_reviews': len(data),
        'total_true_features': sum(len(features) for features in true_features),
        'total_initial_features': sum(len(features) for features in initial_features),
        'total_refined_features': sum(len(features) for features in refined_features)
    }
    
    return results

def save_feedback_evaluation_results(file_path: str, results: dict):
    """Save feedback-based evaluation results to a text file"""
    
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
    output_lines.append("Feedback-Based Refinement Feature Extraction Evaluation Results")
    output_lines.append("=" * 70)
    output_lines.append(f"Total Reviews: {results['total_reviews']}")
    output_lines.append(f"Total True Features: {results['total_true_features']}")
    output_lines.append(f"Total Initial Features: {results['total_initial_features']}")
    output_lines.append(f"Total Refined Features: {results['total_refined_features']}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("INITIAL EXTRACTOR AGENT RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['initial_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['initial_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['initial_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("REFINED EXTRACTOR AGENT RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['refined_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['refined_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['refined_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("PERFORMANCE COMPARISON")
    output_lines.append("-" * 40)
    
    # Compare with initial extractor
    refinement_improvement = results['refined_metrics']['f1'] - results['initial_metrics']['f1']
    
    output_lines.append(f"Initial Extractor F1: {results['initial_metrics']['f1']:.4f}")
    output_lines.append(f"Refined F1: {results['refined_metrics']['f1']:.4f}")
    output_lines.append(f"Refinement Improvement: {refinement_improvement:+.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("DETAILED METRICS COMPARISON")
    output_lines.append("-" * 40)
    output_lines.append("Metric          | Initial   | Refined   | Improvement")
    output_lines.append("-" * 70)
    output_lines.append(f"Precision       | {results['initial_metrics']['precision']:.4f}     | {results['refined_metrics']['precision']:.4f}     | {results['refined_metrics']['precision'] - results['initial_metrics']['precision']:+.4f}")
    output_lines.append(f"Recall          | {results['initial_metrics']['recall']:.4f}     | {results['refined_metrics']['recall']:.4f}     | {results['refined_metrics']['recall'] - results['initial_metrics']['recall']:+.4f}")
    output_lines.append(f"F1 Score        | {results['initial_metrics']['f1']:.4f}     | {results['refined_metrics']['f1']:.4f}     | {refinement_improvement:+.4f}")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Feedback evaluation results saved to: {output_path}")
    return output_path

def main():
    # Check if file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        print("Please run the feedback-based refinement feature extraction first.")
        return
    
    # Evaluate results
    results = evaluate_feedback_refinement_results(INPUT_FILE)
    
    # Save results
    save_feedback_evaluation_results(INPUT_FILE, results)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("FEEDBACK-BASED REFINEMENT EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Initial Extractor F1: {results['initial_metrics']['f1']:.4f}")
    print(f"Refined Extractor F1: {results['refined_metrics']['f1']:.4f}")
    
    refinement_improvement = results['refined_metrics']['f1'] - results['initial_metrics']['f1']
    print(f"Refinement Improvement: {refinement_improvement:+.4f}")

if __name__ == '__main__':
    main() 