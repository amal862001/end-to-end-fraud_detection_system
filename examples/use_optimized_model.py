"""
Example: How to use the optimized model
"""

import sys
import os
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

import joblib
import json
import pandas as pd
from data.load_data import load_data


def load_optimized_model():
    """Load the optimized model and parameters."""
    model_path = 'models/optimized/best_model.pkl'
    scaler_path = 'models/optimized/scaler.pkl'
    params_path = 'models/optimized/best_params.json'
    summary_path = 'models/optimized/optimization_summary.json'
    
    if not os.path.exists(model_path):
        print("❌ Optimized model not found!")
        print("\n💡 First run optimization:")
        print("   pip install optuna")
        print("   python src/models/optimize.py")
        return None, None, None, None
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("✓ Loaded optimized model")
    return model, scaler, params, summary


def show_optimization_results(params, summary):
    """Display optimization results."""
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nBest PR-AUC (CV): {summary['best_pr_auc_cv']:.4f}")
    print(f"Best PR-AUC (Test): {summary['best_pr_auc_test']:.4f}")
    print(f"Number of trials: {summary['n_trials']}")
    print(f"Optimized on: {summary['timestamp']}")
    
    print("\nBest hyperparameters:")
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")
        else:
            print(f"  {key:20s}: {value}")


def compare_models():
    """Compare all models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    models = {
        'Logistic Regression': 0.7612,
        'Random Forest': 0.8171,
        'LightGBM (Default)': 0.8689,
        'LightGBM (Optimized)': 0.8734
    }
    
    df = pd.DataFrame({
        'Model': models.keys(),
        'PR-AUC': models.values()
    })
    
    # Calculate improvement
    baseline = models['Random Forest']
    df['Improvement vs Baseline'] = ((df['PR-AUC'] - baseline) / baseline * 100).round(2)
    
    print("\n", df.to_string(index=False))
    
    print("\n🏆 Best Model: LightGBM (Optimized)")
    print(f"   PR-AUC: {models['LightGBM (Optimized)']:.4f}")
    print(f"   Improvement: +{((models['LightGBM (Optimized)'] - baseline) / baseline * 100):.2f}%")


def make_predictions(model, scaler):
    """Make predictions with optimized model."""
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    # Load data
    print("\nLoading test data...")
    X, y = load_data()
    
    # Take a sample
    sample_size = 1000
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    print(f"Sample size: {sample_size}")
    print(f"Actual frauds: {y_sample.sum()}")
    
    # Scale and predict
    X_scaled = scaler.transform(X_sample)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    print(f"Predicted frauds: {predictions.sum()}")
    
    # Show high-risk transactions
    print("\nTop 10 highest fraud probabilities:")
    results = pd.DataFrame({
        'Actual': y_sample.values,
        'Predicted': predictions,
        'Probability': probabilities
    })
    
    top_10 = results.nlargest(10, 'Probability')
    print(top_10.to_string(index=False))
    
    # Calculate metrics
    accuracy = (predictions == y_sample.values).mean()
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    
    if y_sample.sum() > 0:
        fraud_detected = ((predictions == 1) & (y_sample.values == 1)).sum()
        fraud_total = y_sample.sum()
        detection_rate = fraud_detected / fraud_total
        print(f"Fraud detection rate: {detection_rate*100:.1f}% ({fraud_detected}/{fraud_total})")


def main():
    """Main example."""
    print("="*60)
    print("OPTIMIZED MODEL USAGE EXAMPLE")
    print("="*60)
    
    # Load model
    model, scaler, params, summary = load_optimized_model()
    
    if model is None:
        return
    
    # Show optimization results
    show_optimization_results(params, summary)
    
    # Compare models
    compare_models()
    
    # Make predictions
    make_predictions(model, scaler)
    
    print("\n" + "="*60)
    print("✓ EXAMPLE COMPLETE!")
    print("="*60)
    
    print("\n💡 To use in production:")
    print("   1. Load model: joblib.load('models/optimized/best_model.pkl')")
    print("   2. Load scaler: joblib.load('models/optimized/scaler.pkl')")
    print("   3. Scale data: X_scaled = scaler.transform(X)")
    print("   4. Predict: proba = model.predict_proba(X_scaled)[:, 1]")
    print("   5. Apply threshold: predictions = (proba >= 0.3).astype(int)")


if __name__ == '__main__':
    main()

