"""
Test Artifacts - Verify Deployment Artifacts

This script demonstrates how to load and use the serialized model artifacts
for deployment.

Author: Your Name
Date: 2026-01-15
"""

import sys
sys.path.append('src')

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data.load_data import load_data


def load_artifacts():
    """Load model artifacts from artifacts/ directory."""
    print("="*60)
    print("LOADING ARTIFACTS")
    print("="*60)
    
    # Load model
    model = joblib.load('artifacts/fraud_model.pkl')
    print(f"✓ Loaded model: {type(model).__name__}")
    
    # Load scaler
    scaler = joblib.load('artifacts/scaler.pkl')
    print(f"✓ Loaded scaler: {type(scaler).__name__}")
    
    # Load threshold
    with open('artifacts/threshold.txt', 'r') as f:
        threshold = float(f.read().strip())
    print(f"✓ Loaded threshold: {threshold}")
    
    return model, scaler, threshold


def test_prediction(model, scaler, threshold):
    """Test prediction on sample data."""
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    X, y = load_data(verbose=False)
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Test set: {len(X_test):,} samples, {y_test.sum()} frauds")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Preserve feature names if needed
    if hasattr(X_test, 'columns'):
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Get probabilities
    print("\nMaking predictions...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Results
    print(f"\n✓ Predictions complete!")
    print(f"  Total predictions: {len(y_pred):,}")
    print(f"  Predicted frauds: {y_pred.sum()}")
    print(f"  Actual frauds: {y_test.sum()}")
    
    return y_test, y_pred, y_pred_proba


def show_performance(y_test, y_pred, y_pred_proba):
    """Show model performance."""
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Legit    Fraud")
    print(f"Actual Legit    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"       Fraud    {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Precision:         {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:            {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:          {f1:.4f}")
    print(f"False Positive Rate: {fpr:.6f} ({fpr*100:.4f}%)")
    print(f"\nTrue Positives:    {tp} (frauds caught)")
    print(f"False Negatives:   {fn} (frauds missed)")
    print(f"False Positives:   {fp} (false alarms)")
    print(f"True Negatives:    {tn} (correct legitimate)")


def show_examples(y_test, y_pred, y_pred_proba):
    """Show example predictions."""
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Convert to arrays if needed
    y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test
    
    # High confidence frauds
    fraud_mask = (y_pred == 1) & (y_test_arr == 1)
    fraud_probs = y_pred_proba[fraud_mask]
    
    if len(fraud_probs) > 0:
        print("\n✓ Correctly Detected Frauds (Top 5):")
        top_frauds = np.argsort(fraud_probs)[-5:][::-1]
        for i, idx in enumerate(top_frauds, 1):
            prob = fraud_probs[idx]
            print(f"  {i}. Fraud probability: {prob:.4f} ({prob*100:.2f}%)")
    
    # False positives
    fp_mask = (y_pred == 1) & (y_test_arr == 0)
    fp_probs = y_pred_proba[fp_mask]
    
    if len(fp_probs) > 0:
        print("\n⚠ False Positives (Legitimate flagged as fraud):")
        for i, prob in enumerate(fp_probs[:5], 1):
            print(f"  {i}. Fraud probability: {prob:.4f} ({prob*100:.2f}%)")
    
    # Missed frauds
    fn_mask = (y_pred == 0) & (y_test_arr == 1)
    fn_probs = y_pred_proba[fn_mask]
    
    if len(fn_probs) > 0:
        print("\n⚠ Missed Frauds (False Negatives):")
        for i, prob in enumerate(fn_probs[:5], 1):
            print(f"  {i}. Fraud probability: {prob:.4f} ({prob*100:.2f}%)")


def main():
    """Main test pipeline."""
    print("\n" + "="*60)
    print("TESTING DEPLOYMENT ARTIFACTS")
    print("="*60)
    print("\nThis script tests the serialized model artifacts")
    print("from the artifacts/ directory.\n")
    
    try:
        # 1. Load artifacts
        model, scaler, threshold = load_artifacts()
        
        # 2. Test predictions
        y_test, y_pred, y_pred_proba = test_prediction(model, scaler, threshold)
        
        # 3. Show performance
        show_performance(y_test, y_pred, y_pred_proba)
        
        # 4. Show examples
        show_examples(y_test, y_pred, y_pred_proba)
        
        # 5. Summary
        print("\n" + "="*60)
        print("✓ ARTIFACTS TEST COMPLETE!")
        print("="*60)
        print("\n✅ Model artifacts are working correctly!")
        print("\n📁 Deployment files:")
        print("  - artifacts/fraud_model.pkl")
        print("  - artifacts/scaler.pkl")
        print("  - artifacts/threshold.txt")
        print("\n🚀 Ready for deployment!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solution:")
        print("  Run model serialization first:")
        print("  python src/models/serialize_model.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

