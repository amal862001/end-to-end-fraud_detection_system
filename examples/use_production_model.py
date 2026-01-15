"""
Use Production Model

This script demonstrates how to use the final production model
with the optimal threshold for fraud detection.

Author: Your Name
Date: 2026-01-15
"""

import sys
sys.path.append('src')

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data.load_data import load_data


def load_production_model():
    """Load production model, scaler, and threshold."""
    print("="*60)
    print("LOADING PRODUCTION MODEL")
    print("="*60)
    
    try:
        # Load model
        model = joblib.load('models/production/production_model.pkl')
        print("✓ Loaded model")
        
        # Load scaler
        scaler = joblib.load('models/production/production_scaler.pkl')
        print("✓ Loaded scaler")
        
        # Load threshold configuration
        with open('models/production/production_threshold.json', 'r') as f:
            threshold_config = json.load(f)
        print("✓ Loaded threshold configuration")
        
        # Load metadata
        with open('models/production/production_metadata.json', 'r') as f:
            metadata = json.load(f)
        print("✓ Loaded metadata")
        
        print(f"\nModel: {metadata['model_name']}")
        print(f"Version: {metadata['version']}")
        print(f"Optimal Threshold: {threshold_config['optimal_threshold']:.2f}")
        
        return model, scaler, threshold_config, metadata
        
    except FileNotFoundError as e:
        print(f"\n❌ Production model not found!")
        print("Please run final model selection first:")
        print("   python src/models/final_selection.py")
        raise e


def make_predictions(model, scaler, X, threshold):
    """Make predictions using production model."""
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Apply optimal threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred, y_pred_proba


def evaluate_predictions(y_true, y_pred, y_pred_proba):
    """Evaluate predictions."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print("\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TN: {tn:4d}")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def show_example_predictions(X, y, y_pred, y_pred_proba, n_examples=10):
    """Show example predictions."""
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Get fraud predictions
    fraud_indices = np.where(y_pred == 1)[0]
    
    if len(fraud_indices) == 0:
        print("\nNo frauds predicted!")
        return
    
    # Show first n examples
    n_show = min(n_examples, len(fraud_indices))
    
    print(f"\nShowing {n_show} fraud predictions:")
    print("-" * 80)
    print(f"{'Index':<8} {'True Label':<12} {'Predicted':<12} {'Probability':<15} {'Correct?':<10}")
    print("-" * 80)
    
    for i in range(n_show):
        idx = fraud_indices[i]
        true_label = 'Fraud' if y.iloc[idx] == 1 else 'Legitimate'
        pred_label = 'Fraud'
        prob = y_pred_proba[idx]
        correct = '✓' if y.iloc[idx] == y_pred[idx] else '✗'
        
        print(f"{idx:<8} {true_label:<12} {pred_label:<12} {prob:<15.4f} {correct:<10}")
    
    print("-" * 80)


def main():
    """Main function to demonstrate production model usage."""
    print("\n" + "="*60)
    print("PRODUCTION MODEL DEMONSTRATION")
    print("="*60)
    
    # ========================================================================
    # 1. LOAD PRODUCTION MODEL
    # ========================================================================
    
    model, scaler, threshold_config, metadata = load_production_model()
    optimal_threshold = threshold_config['optimal_threshold']
    
    # ========================================================================
    # 2. LOAD TEST DATA
    # ========================================================================
    
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    X, y = load_data()
    
    # Use same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTest set: {len(X_test)} samples")
    print(f"Frauds: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # ========================================================================
    # 3. MAKE PREDICTIONS
    # ========================================================================
    
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    print(f"\nUsing optimal threshold: {optimal_threshold:.2f}")
    
    y_pred, y_pred_proba = make_predictions(model, scaler, X_test, optimal_threshold)
    
    print(f"\nPredicted frauds: {y_pred.sum()}")
    print(f"Actual frauds: {y_test.sum()}")
    
    # ========================================================================
    # 4. EVALUATE
    # ========================================================================
    
    metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)
    
    # ========================================================================
    # 5. SHOW EXAMPLES
    # ========================================================================
    
    show_example_predictions(X_test, y_test, y_pred, y_pred_proba, n_examples=10)
    
    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print("✓ PRODUCTION MODEL READY!")
    print("="*60)
    
    print(f"\nModel: {metadata['model_name']}")
    print(f"Threshold: {optimal_threshold:.2f}")
    print(f"Performance:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\n💡 Next Steps:")
    print("   1. Deploy model with FastAPI")
    print("   2. Set up monitoring")
    print("   3. Track production metrics")
    print("   4. Retrain periodically")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()

