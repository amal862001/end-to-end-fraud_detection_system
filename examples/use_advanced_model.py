"""
Example: How to use trained advanced models (XGBoost/LightGBM)
"""

import sys
import os
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

import joblib
import pandas as pd
import numpy as np
from data.load_data import load_data


def load_advanced_model(model_type='lightgbm'):
    """Load trained advanced model."""
    model_path = f'models/advanced/{model_type}.pkl'
    scaler_path = 'models/advanced/scaler.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("\n💡 First train the model:")
        print("   pip install xgboost lightgbm")
        print("   python src/models/advanced_models.py")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✓ Loaded {model_type} model")
    return model, scaler


def predict_fraud(model, scaler, X):
    """Make fraud predictions."""
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities


def main():
    """Example usage."""
    print("="*60)
    print("ADVANCED MODEL PREDICTION EXAMPLE")
    print("="*60)
    
    # Load model
    print("\n1. Loading LightGBM model...")
    model, scaler = load_advanced_model('lightgbm')
    
    if model is None:
        return
    
    # Load data
    print("\n2. Loading test data...")
    X, y = load_data()
    
    # Take a sample
    sample_size = 1000
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    print(f"   Sample size: {sample_size}")
    print(f"   Actual frauds: {y_sample.sum()}")
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions, probabilities = predict_fraud(model, scaler, X_sample)
    
    print(f"   Predicted frauds: {predictions.sum()}")
    print(f"   Average fraud probability: {probabilities.mean():.4f}")
    
    # Show results
    print("\n4. Sample predictions:")
    results = pd.DataFrame({
        'Actual': y_sample.values,
        'Predicted': predictions,
        'Probability': probabilities
    })
    
    # High-risk transactions
    print("\n   Top 10 highest fraud probabilities:")
    top_10 = results.nlargest(10, 'Probability')
    print(top_10.to_string(index=False))
    
    # Accuracy
    accuracy = (predictions == y_sample.values).mean()
    print(f"\n5. Accuracy: {accuracy*100:.2f}%")
    
    # Fraud detection rate
    if y_sample.sum() > 0:
        fraud_detected = ((predictions == 1) & (y_sample.values == 1)).sum()
        fraud_total = y_sample.sum()
        detection_rate = fraud_detected / fraud_total
        print(f"   Fraud detection rate: {detection_rate*100:.1f}% ({fraud_detected}/{fraud_total})")
    
    print("\n" + "="*60)
    print("✓ PREDICTION COMPLETE!")
    print("="*60)
    
    print("\n💡 To use in production:")
    print("   1. Load model: joblib.load('models/advanced/lightgbm.pkl')")
    print("   2. Scale data: scaler.transform(X)")
    print("   3. Predict: model.predict_proba(X_scaled)[:, 1]")
    print("   4. Apply threshold: predictions = (proba >= 0.3).astype(int)")


if __name__ == '__main__':
    main()

