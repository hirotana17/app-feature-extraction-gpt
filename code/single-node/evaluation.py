import json
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
import os

# Configuration
INPUT_DIR = './data-gpt/T-FREX/in-domain/bin0/feature_extracted_data'
INPUT_FILE = 'test-set-extracted-promptft1-BYrc6PIY.json'
OUTPUT_DIR = './data-gpt/T-FREX/in-domain/bin0/evaluation_result'

def load_extracted_data(file_path: str) -> List[Dict]:
    """Load the extracted features data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_evaluation_data(data: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prepare data for evaluation by converting features lists to sets
    for exact matching evaluation (case-insensitive)
    """
    true_features = []
    predicted_features = []
    
    for review in data:
        # Convert feature lists to sets for exact matching
        # Make case-insensitive by converting to lowercase
        true_set = set([feature.lower().strip() for feature in review['output']])
        pred_set = set([feature.lower().strip() for feature in review['extracted_features']])
        
        true_features.append(true_set)
        predicted_features.append(pred_set)
    
    return true_features, predicted_features

def calculate_metrics(true_features: List[set], predicted_features: List[set]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for feature extraction.
    Note: Accuracy is not used because we don't have ground-truth data for non-feature entities (true negatives).
    We only have explicit annotations for features, but not for what is NOT a feature.
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
    # We use micro-averaging because we want to treat all features equally
    precision = precision_score(true_binary, pred_binary, average='micro', zero_division=0)
    recall = recall_score(true_binary, pred_binary, average='micro', zero_division=0)
    f1 = f1_score(true_binary, pred_binary, average='micro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def analyze_feature_distribution(data: List[Dict]) -> Dict:
    """
    Analyze the distribution of features in true and predicted sets (case-insensitive)
    """
    true_features_count = defaultdict(int)
    pred_features_count = defaultdict(int)
    
    for review in data:
        for feature in review['output']:
            # Make case-insensitive by converting to lowercase
            normalized_feature = feature.lower().strip()
            true_features_count[normalized_feature] += 1
        for feature in review['extracted_features']:
            # Make case-insensitive by converting to lowercase
            normalized_feature = feature.lower().strip()
            pred_features_count[normalized_feature] += 1
    
    return {
        'true_features_distribution': dict(true_features_count),
        'predicted_features_distribution': dict(pred_features_count)
    }

def evaluate_extraction(file_path: str) -> Dict:
    """
    Evaluate the feature extraction results
    """
    # Load data
    data = load_extracted_data(file_path)
    
    # Prepare data for evaluation
    true_features, predicted_features = prepare_evaluation_data(data)
    
    # Calculate metrics
    metrics = calculate_metrics(true_features, predicted_features)
    
    # Analyze feature distribution
    distribution = analyze_feature_distribution(data)
    
    # Combine results
    results = {
        'metrics': metrics,
        'distribution': distribution,
        'total_reviews': len(data),
        'total_true_features': sum(len(features) for features in true_features),
        'total_predicted_features': sum(len(features) for features in predicted_features)
    }
    
    return results

def main():
    # Generate file paths
    input_file = os.path.join(INPUT_DIR, INPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.basename(input_file).replace('.json', '')
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}-result-insensitive.txt")
    
    # Evaluate the results
    results = evaluate_extraction(input_file)
    
    # Prepare output text
    output_text = []
    output_text.append("\nEvaluation Results:")
    output_text.append("-" * 50)
    output_text.append(f"Total Reviews: {results['total_reviews']}")
    output_text.append(f"Total True Features: {results['total_true_features']}")
    output_text.append(f"Total Predicted Features: {results['total_predicted_features']}")
    output_text.append("\nMetrics:")
    output_text.append(f"Precision: {results['metrics']['precision']:.4f}")
    output_text.append(f"Recall: {results['metrics']['recall']:.4f}")
    output_text.append(f"F1 Score: {results['metrics']['f1']:.4f}")
    
    # Print top 10 most common features
    output_text.append("\nTop 10 Most Common True Features:")
    true_features = sorted(
        results['distribution']['true_features_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for feature, count in true_features:
        output_text.append(f"{feature}: {count}")
    
    output_text.append("\nTop 10 Most Common Predicted Features:")
    pred_features = sorted(
        results['distribution']['predicted_features_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for feature, count in pred_features:
        output_text.append(f"{feature}: {count}")
    
    # Save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_text))
    
    # Also print to console
    print('\n'.join(output_text))
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
