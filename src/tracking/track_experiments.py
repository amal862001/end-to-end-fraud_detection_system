"""
Track All Fraud Detection Experiments with MLflow

This script demonstrates how to track all your models with MLflow:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM (default)
- LightGBM (optimized)

It logs:
- Parameters (hyperparameters, data splits)
- Metrics (PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix)
- Artifacts (models, plots, feature importance)

Author: Your Name
Date: 2026-01-15
"""

import sys
import os
sys.path.append('src')

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.load_data import load_data
from tracking.mlflow_utils import MLflowTracker, track_experiment


def track_all_models():
    """Track all trained models with MLflow."""
    
    print("="*60)
    print("TRACKING ALL EXPERIMENTS WITH MLFLOW")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
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
    
    print(f"Train: {X_train.shape[0]} samples, {y_train.sum()} frauds")
    print(f"Test: {X_test.shape[0]} samples, {y_test.sum()} frauds")
    
    # Initialize tracker
    tracker = MLflowTracker(
        experiment_name="fraud-detection-all-models",
        # Uncomment to use DagsHub:
        # dagshub_repo="fraud-detection",
        # dagshub_user="your-username"
    )
    
    # ========================================================================
    # 1. TRACK RANDOM FOREST
    # ========================================================================
    
    print("\n" + "="*60)
    print("TRACKING: Random Forest")
    print("="*60)
    
    try:
        rf_model = joblib.load('models/random_forest.pkl')
        
        tracker.start_run(
            run_name="random-forest-baseline",
            tags={"model_type": "RandomForest", "version": "v1"},
            description="Random Forest with class_weight='balanced'"
        )
        
        # Log parameters
        rf_params = {
            'model': 'RandomForestClassifier',
            'n_estimators': 100,
            'max_depth': None,
            'class_weight': 'balanced',
            'random_state': 42
        }
        tracker.log_params(rf_params)
        
        # Log complete evaluation
        tracker.log_complete_evaluation(
            rf_model, X_test_scaled, y_test, feature_names, "random_forest"
        )
        
        tracker.end_run()
        
    except FileNotFoundError:
        print("⚠ Random Forest model not found. Skipping...")
    
    # ========================================================================
    # 2. TRACK XGBOOST
    # ========================================================================
    
    print("\n" + "="*60)
    print("TRACKING: XGBoost")
    print("="*60)
    
    try:
        xgb_model = joblib.load('models/advanced/xgboost.pkl')
        
        tracker.start_run(
            run_name="xgboost-advanced",
            tags={"model_type": "XGBoost", "version": "v1"},
            description="XGBoost with scale_pos_weight and early stopping"
        )
        
        # Log parameters
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_params = {
            'model': 'XGBClassifier',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'early_stopping_rounds': 10
        }
        tracker.log_params(xgb_params)
        
        # Log complete evaluation
        tracker.log_complete_evaluation(
            xgb_model, X_test_scaled, y_test, feature_names, "xgboost"
        )
        
        tracker.end_run()
        
    except FileNotFoundError:
        print("⚠ XGBoost model not found. Skipping...")
    
    # ========================================================================
    # 3. TRACK LIGHTGBM (DEFAULT)
    # ========================================================================
    
    print("\n" + "="*60)
    print("TRACKING: LightGBM (Default)")
    print("="*60)
    
    try:
        lgbm_model = joblib.load('models/advanced/lightgbm.pkl')
        
        tracker.start_run(
            run_name="lightgbm-default",
            tags={"model_type": "LightGBM", "version": "default", "best": "true"},
            description="LightGBM with default parameters - BEST MODEL"
        )
        
        # Log parameters
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        lgbm_params = {
            'model': 'LGBMClassifier',
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        tracker.log_params(lgbm_params)

        # Log complete evaluation
        tracker.log_complete_evaluation(
            lgbm_model, X_test_scaled, y_test, feature_names, "lightgbm_default"
        )

        tracker.end_run()

    except FileNotFoundError:
        print("⚠ LightGBM (default) model not found. Skipping...")

    # ========================================================================
    # 4. TRACK LIGHTGBM (OPTIMIZED)
    # ========================================================================

    print("\n" + "="*60)
    print("TRACKING: LightGBM (Optimized)")
    print("="*60)

    try:
        lgbm_opt_model = joblib.load('models/optimized/best_model.pkl')

        # Load optimization parameters
        import json
        with open('models/optimized/best_params.json', 'r') as f:
            opt_params = json.load(f)

        tracker.start_run(
            run_name="lightgbm-optimized",
            tags={"model_type": "LightGBM", "version": "optimized", "optimizer": "Optuna"},
            description="LightGBM with Optuna-optimized hyperparameters (50 trials)"
        )

        # Log parameters
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        opt_params['scale_pos_weight'] = scale_pos_weight
        opt_params['model'] = 'LGBMClassifier'
        opt_params['optimizer'] = 'Optuna'
        opt_params['n_trials'] = 50
        tracker.log_params(opt_params)

        # Log complete evaluation
        tracker.log_complete_evaluation(
            lgbm_opt_model, X_test_scaled, y_test, feature_names, "lightgbm_optimized"
        )

        tracker.end_run()

    except FileNotFoundError:
        print("⚠ LightGBM (optimized) model not found. Skipping...")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*60)
    print("✓ ALL EXPERIMENTS TRACKED!")
    print("="*60)
    print("\nTo view results:")
    print("  1. Run: mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("  3. Compare experiments and metrics")
    print("\nOr view on DagsHub if configured:")
    print("  https://dagshub.com/your-username/fraud-detection/experiments")
    print("="*60 + "\n")


if __name__ == '__main__':
    track_all_models()


