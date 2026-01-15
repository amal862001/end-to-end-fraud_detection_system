"""
MLflow Tracking Example

This script demonstrates how to use MLflow tracking for the fraud detection project.

Run this to see MLflow in action!

Author: Your Name
Date: 2026-01-15
"""

import sys
sys.path.append('src')

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.load_data import load_data
from tracking.mlflow_utils import MLflowTracker


def main():
    """Demonstrate MLflow tracking with a simple example."""
    
    print("="*60)
    print("MLFLOW TRACKING EXAMPLE")
    print("="*60)
    print("\nThis example shows how to track experiments with MLflow.")
    print("We'll track the LightGBM model (best model).\n")
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    
    print("Step 1: Loading data...")
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X.columns.tolist()
    
    print(f"✓ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # ========================================================================
    # 2. LOAD MODEL
    # ========================================================================
    
    print("\nStep 2: Loading trained model...")
    try:
        model = joblib.load('models/advanced/lightgbm.pkl')
        print("✓ Model loaded: LightGBM (default parameters)")
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first:")
        print("   python src/models/advanced_models.py")
        return
    
    # ========================================================================
    # 3. INITIALIZE MLFLOW TRACKER
    # ========================================================================
    
    print("\nStep 3: Initializing MLflow tracker...")
    tracker = MLflowTracker(
        experiment_name="fraud-detection-demo",
        # Uncomment to use DagsHub:
        # dagshub_repo="fraud-detection",
        # dagshub_user="your-username"
    )
    
    # ========================================================================
    # 4. START RUN AND LOG EVERYTHING
    # ========================================================================
    
    print("\nStep 4: Starting MLflow run...")
    tracker.start_run(
        run_name="lightgbm-demo",
        tags={
            "model_type": "LightGBM",
            "version": "v1",
            "best": "true"
        },
        description="Demo: LightGBM with default parameters - Best model"
    )
    
    # Log parameters
    print("\nStep 5: Logging parameters...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        'model': 'LGBMClassifier',
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': scale_pos_weight,
        'class_weight': 'balanced',
        'random_state': 42,
        # Data info
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_frauds': int(y_train.sum()),
        'test_frauds': int(y_test.sum()),
        'n_features': X_train.shape[1]
    }
    tracker.log_params(params)
    
    # Log complete evaluation
    print("\nStep 6: Logging complete evaluation...")
    print("  - Calculating metrics...")
    print("  - Creating confusion matrix...")
    print("  - Creating PR curve...")
    print("  - Creating feature importance plot...")
    print("  - Logging model...")
    
    metrics = tracker.log_complete_evaluation(
        model=model,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_names=feature_names,
        model_name="lightgbm_demo"
    )
    
    # End run
    tracker.end_run()
    
    # ========================================================================
    # 5. DISPLAY RESULTS
    # ========================================================================
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nKey Metrics:")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f} ⭐")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # ========================================================================
    # 6. INSTRUCTIONS
    # ========================================================================
    
    print("\n" + "="*60)
    print("✓ EXPERIMENT TRACKED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📊 To view your experiment:")
    print("\n1. Start MLflow UI:")
    print("   mlflow ui")
    print("\n2. Open in browser:")
    print("   http://localhost:5000")
    print("\n3. You'll see:")
    print("   - Experiment: fraud-detection-demo")
    print("   - Run: lightgbm-demo")
    print("   - All parameters, metrics, and artifacts")
    
    print("\n📁 Artifacts saved:")
    print("   - Model: lightgbm_demo")
    print("   - Confusion matrix plot")
    print("   - Precision-Recall curve")
    print("   - Feature importance plot + CSV")
    
    print("\n💡 Next steps:")
    print("   1. Track more models: python src/tracking/track_experiments.py")
    print("   2. Compare experiments in MLflow UI")
    print("   3. Register best model for deployment")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()

