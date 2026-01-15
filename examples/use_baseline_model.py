"""
Example: How to use the trained baseline models for predictions
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

import joblib
import pandas as pd
from data.load_data import load_data


def load_trained_model(model_name='random_forest'):
    """Load a trained model and scaler."""
    model_path = f'models/baseline/{model_name}.pkl'
    scaler_path = 'models/baseline/scaler.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✓ Loaded {model_name} model")
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
    print("BASELINE MODEL PREDICTION EXAMPLE")
    print("="*60)
    
    # Load data
    print("\n1. Loading test data...")
    X, y = load_data()
    
    # Take a small sample
    sample_size = 1000
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    print(f"   Sample size: {sample_size}")
    print(f"   Actual frauds in sample: {y_sample.sum()}")
    
    # Load model
    print("\n2. Loading Random Forest model...")
    model, scaler = load_trained_model('random_forest')
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions, probabilities = predict_fraud(model, scaler, X_sample)
    
    print(f"   Predicted frauds: {predictions.sum()}")
    print(f"   Average fraud probability: {probabilities.mean():.4f}")
    
    # Show some examples
    print("\n4. Sample predictions:")
    results = pd.DataFrame({
        'Actual': y_sample.values,
        'Predicted': predictions,
        'Probability': probabilities
    })
    
    # Show high-risk transactions
    print("\n   Top 10 highest fraud probabilities:")
    top_10 = results.nlargest(10, 'Probability')
    print(top_10.to_string(index=False))
    
    # Accuracy
    accuracy = (predictions == y_sample.values).mean()
    print(f"\n5. Accuracy on sample: {accuracy*100:.2f}%")
    
    print("\n" + "="*60)
    print("✓ PREDICTION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()

