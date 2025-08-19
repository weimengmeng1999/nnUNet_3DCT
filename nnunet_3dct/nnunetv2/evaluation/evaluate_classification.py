#!/usr/bin/env python3
"""
Classification performance evaluation script
Calculates F1-score and other classification metrics from prediction JSONs and ground truth labels
"""

import os
import json
import argparse
from typing import Dict, List, Union, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    confusion_matrix, classification_report, roc_auc_score
)
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json


def load_classification_predictions(cls_folder: str) -> Dict[str, Dict]:
    """
    Load all classification prediction JSON files from a folder
    
    Args:
        cls_folder: Folder containing *_classification.json files
    
    Returns:
        Dictionary mapping case_id to prediction data
    """
    predictions = {}
    json_files = subfiles(cls_folder, suffix='_classification.json', join=True)
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        filename = os.path.basename(json_file)
        case_id = filename.replace('_classification.json', '')
        # case_id = data['case'] #new version use case id in the name of classification pre file
        predictions[case_id] = data
    
    print(f"Loaded {len(predictions)} classification predictions from {cls_folder}")
    return predictions


def load_ground_truth_labels(gt_file: str) -> Dict[str, int]:
    """
    Load ground truth labels from JSON file for single-label classification
    
    Expected format (supports both string and integer labels):
    {
        "case_001": "0",  # or 0
        "case_002": "1",  # or 1
        "case_003": "2",  # or 2
        ...
    }
    
    Args:
        gt_file: Path to ground truth JSON file
    
    Returns:
        Dictionary mapping case_id to ground truth label (integer)
    """
    with open(gt_file, 'r') as f:
        gt_labels_raw = json.load(f)
    
    # Convert string labels to integers and filter out non-case entries
    gt_labels = {}
    for case_id, label in gt_labels_raw.items():
        # Skip entries that start with underscore (comments/metadata)
        if case_id.startswith('_'):
            continue
            
        # Convert string labels to integers
        if isinstance(label, str):
            try:
                gt_labels[case_id] = int(label)
            except ValueError:
                print(f"Warning: Could not convert label '{label}' to integer for case {case_id}")
                continue
        elif isinstance(label, int):
            gt_labels[case_id] = label
        else:
            print(f"Warning: Unsupported label type {type(label)} for case {case_id}")
            continue
    
    print(f"Loaded ground truth labels for {len(gt_labels)} cases from {gt_file}")
    if len(gt_labels) < len(gt_labels_raw):
        print(f"Note: Filtered out {len(gt_labels_raw) - len(gt_labels)} non-case entries")
    
    return gt_labels


def create_gt_from_case_names(case_names: List[str], label_mapping: Dict[str, Union[int, List[int]]]) -> Dict[str, Union[int, List[int]]]:
    """
    Create ground truth labels based on case naming convention
    
    Args:
        case_names: List of case identifiers
        label_mapping: Mapping from substring/pattern to label
    
    Returns:
        Dictionary mapping case_id to ground truth labels
    """
    gt_labels = {}
    
    for case_id in case_names:
        # Example: if case names contain class info like "normal_001", "tumor_002", etc.
        assigned = False
        for pattern, label in label_mapping.items():
            if pattern in case_id.lower():
                gt_labels[case_id] = label
                assigned = True
                break
        
        if not assigned:
            print(f"Warning: Could not assign label to case {case_id}")
    
    return gt_labels


def calculate_classification_metrics(predictions: Dict[str, Dict], 
                                   gt_labels: Dict[str, int],
                                   class_names: List[str] = None) -> Dict:
    """
    Calculate comprehensive single-label classification metrics
    
    Args:
        predictions: Dictionary of predictions from load_classification_predictions
        gt_labels: Dictionary of ground truth labels (single integer per case)
        class_names: List of class names for reporting
    
    Returns:
        Dictionary containing all calculated metrics
    """
    # Align predictions and ground truth
    common_cases = set(predictions.keys()) & set(gt_labels.keys())
    if len(common_cases) == 0:
        raise ValueError("No common cases found between predictions and ground truth")
    
    print(f"Evaluating {len(common_cases)} common cases")
    
    if len(common_cases) < len(predictions):
        missing_gt = set(predictions.keys()) - common_cases
        print(f"Warning: Missing ground truth for {len(missing_gt)} cases: {list(missing_gt)[:5]}...")
    
    if len(common_cases) < len(gt_labels):
        missing_pred = set(gt_labels.keys()) - common_cases
        print(f"Warning: Missing predictions for {len(missing_pred)} cases: {list(missing_pred)[:5]}...")
    
    # Extract aligned predictions and ground truth
    y_true = []
    y_pred = []
    y_probs = []
    
    for case_id in sorted(common_cases):
        pred_data = predictions[case_id]
        true_label = gt_labels[case_id]
        
        y_true.append(true_label)
        y_pred.append(pred_data['pred'])
        y_probs.append(pred_data['probs'])
    
    # Convert to numpy arrays for single-label classification
    y_true = np.array(y_true)  # Shape: (n_samples,)
    y_pred = np.array(y_pred)  # Shape: (n_samples,)
    y_probs = np.array(y_probs)  # Shape: (n_samples, n_classes)
    
    # Calculate metrics
    results = {
        'n_samples': len(common_cases),
        'class_names': class_names
    }
    
    # Single-label classification metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    results['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    results['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    results['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # AUC calculation
    n_classes = y_probs.shape[1]
    if n_classes == 2:  # Binary classification
        try:
            results['auc'] = roc_auc_score(y_true, y_probs[:, 1])
        except ValueError as e:
            print(f"Could not calculate binary AUC: {e}")
            results['auc'] = None
    elif n_classes > 2:  # Multi-class
        try:
            results['auc_macro'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            results['auc_weighted'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except ValueError as e:
            print(f"Could not calculate multi-class AUC: {e}")
            results['auc_macro'] = None
            results['auc_weighted'] = None
    
    # Add detailed classification report
    if class_names:
        target_names = class_names
    else:
        target_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
    
    results['classification_report'] = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    
    return results


def print_results_summary(results: Dict):
    """Print a nice summary of the results"""
    print("\n" + "="*60)
    print("CLASSIFICATION PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Number of samples: {results['n_samples']}")
    print(f"Task type: Single-label classification")
    
    print(f"\nACCURACY: {results['accuracy']:.4f}")
    print(f"F1-SCORE (macro): {results['f1_macro']:.4f}")
    print(f"F1-SCORE (micro): {results['f1_micro']:.4f}")
    print(f"F1-SCORE (weighted): {results['f1_weighted']:.4f}")
    
    print(f"\nPRECISION (macro): {results['precision_macro']:.4f}")
    print(f"RECALL (macro): {results['recall_macro']:.4f}")
    
    if 'auc' in results and results['auc'] is not None:
        print(f"AUC: {results['auc']:.4f}")
    elif 'auc_macro' in results and results['auc_macro'] is not None:
        print(f"AUC (macro): {results['auc_macro']:.4f}")
    
    # Check if F1 meets requirement
    f1_requirement = 0.7
    if results['f1_macro'] >= f1_requirement:
        print(f"\nREQUIREMENT MET: F1-macro ({results['f1_macro']:.4f}) >= {f1_requirement}")
    else:
        print(f"\nREQUIREMENT NOT MET: F1-macro ({results['f1_macro']:.4f}) < {f1_requirement}")
    
    # Per-class performance
    if results['class_names']:
        print(f"\nPER-CLASS F1 SCORES:")
        for i, (name, f1) in enumerate(zip(results['class_names'], results['f1_per_class'])):
            print(f"  {name}: {f1:.4f}")
    else:
        print(f"\nPER-CLASS F1 SCORES:")
        for i, f1 in enumerate(results['f1_per_class']):
            print(f"  Class {i}: {f1:.4f}")
    
    # Show confusion matrix
    print(f"\nCONFUSION MATRIX:")
    cm = np.array(results['confusion_matrix'])
    if results['class_names'] and len(results['class_names']) == cm.shape[0]:
        print("Predicted ->")
        print("Actual ↓   ", end="")
        for name in results['class_names']:
            print(f"{name:>8}", end="")
        print()
        for i, name in enumerate(results['class_names']):
            print(f"{name:>10}", end="")
            for j in range(cm.shape[1]):
                print(f"{cm[i,j]:>8}", end="")
            print()
    else:
        print(f"Shape: {cm.shape}")
        for row in cm:
            print("  ", row.tolist())


def main():
    parser = argparse.ArgumentParser(description='Evaluate classification performance')
    parser.add_argument('cls_folder', type=str, 
                       help='Folder containing *_classification.json prediction files')
    parser.add_argument('gt_file', type=str,
                       help='JSON file with ground truth labels')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output JSON file for detailed results')
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                       help='Names of classes for reporting (e.g. --class-names "Normal" "Tumor" "Other")')
    
    args = parser.parse_args()
    
    # Load predictions and ground truth
    predictions = load_classification_predictions(args.cls_folder)
    gt_labels = load_ground_truth_labels(args.gt_file)
    
    # Calculate metrics
    results = calculate_classification_metrics(
        predictions, gt_labels, 
        class_names=args.class_names
    )
    
    # Print summary
    print_results_summary(results)
    
    # Save detailed results
    if args.output:
        save_json(results, args.output, sort_keys=True)
        print(f"\nDetailed results saved to: {args.output}")
    else:
        output_file = join(args.cls_folder, 'classification_metrics.json')
        save_json(results, output_file, sort_keys=True)
        print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()