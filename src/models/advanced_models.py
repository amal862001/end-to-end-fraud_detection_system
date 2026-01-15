"""
Advanced Modeling - XGBoost & LightGBM
Simple implementation for fraud detection.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    recall_score,
    precision_score
)
import xgboost as xgb
import lightgbm as lgb
import joblib

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import load_data


def prepare_data(test_size=0.2, random_state=42):
    """Load and prepare data for training."""
    print("Loading data...")
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train.shape[0]} samples ({y_train.sum()} frauds)")
    print(f"Test set: {X_test.shape[0]} samples ({y_test.sum()} frauds)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    # PR-AUC (most important for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }
    
    return metrics


def print_results(name, metrics):
    """Print formatted results."""
    print(f"\n{name} Results:")
    print(f"  PR-AUC:    {metrics['PR-AUC']:.4f} ⭐")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"  Confusion: TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with early stopping."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Simple XGBoost parameters (easy to understand)
    params = {
        'max_depth': 6,              # Tree depth
        'learning_rate': 0.1,        # Step size
        'n_estimators': 200,         # Number of trees
        'scale_pos_weight': scale_pos_weight,  # Handle imbalance
        'eval_metric': 'aucpr',      # PR-AUC metric
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("\nTraining with early stopping...")
    
    # Train with early stopping
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False  # Set to True to see progress
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_results("XGBoost", metrics)

    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with early stopping."""
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)

    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Simple LightGBM parameters (easy to understand)
    params = {
        'max_depth': 6,              # Tree depth
        'learning_rate': 0.1,        # Step size
        'n_estimators': 200,         # Number of trees
        'scale_pos_weight': scale_pos_weight,  # Handle imbalance
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1                # Suppress warnings
    }

    print("\nTraining with early stopping...")

    # Train with early stopping
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_results("LightGBM", metrics)

    return model, metrics


def compare_models(baseline_metrics, xgb_metrics, lgb_metrics):
    """Compare all models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Random Forest (Baseline)': baseline_metrics,
        'XGBoost': xgb_metrics,
        'LightGBM': lgb_metrics
    }).T

    # Select key metrics
    comparison = comparison[['PR-AUC', 'ROC-AUC', 'Recall', 'Precision', 'F1-Score']]

    print("\n", comparison.to_string())

    # Find best model
    best_model = comparison['PR-AUC'].idxmax()
    best_pr_auc = comparison['PR-AUC'].max()

    print(f"\n🏆 Best Model: {best_model}")
    print(f"   PR-AUC: {best_pr_auc:.4f}")

    return comparison


def save_best_model(model, scaler, model_name, output_dir='models/advanced'):
    """Save the best model."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("SAVING BEST MODEL")
    print("="*60)

    # Save model
    model_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump(model, model_path)
    print(f"✓ Saved {model_name} to {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to {scaler_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("ADVANCED MODELING - XGBOOST & LIGHTGBM")
    print("="*60)
    print("\nObjective: Maximize fraud detection performance")
    print("Key Metric: PR-AUC (Precision-Recall AUC)")

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # Load baseline results for comparison
    print("\n" + "="*60)
    print("BASELINE (Random Forest from previous training)")
    print("="*60)
    baseline_metrics = {
        'PR-AUC': 0.8171,
        'ROC-AUC': 0.9766,
        'Recall': 0.8265,
        'Precision': 0.8182,
        'F1-Score': 0.8223,
        'TP': 81,
        'FP': 18,
        'TN': 56846,
        'FN': 17
    }
    print_results("Random Forest (Baseline)", baseline_metrics)

    # Train advanced models
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)

    # Compare models
    comparison = compare_models(baseline_metrics, xgb_metrics, lgb_metrics)

    # Save both models (not just the best one)
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)

    # Save XGBoost
    save_best_model(xgb_model, scaler, 'XGBoost')

    # Save LightGBM
    save_best_model(lgb_model, scaler, 'LightGBM')

    # Determine best model
    if xgb_metrics['PR-AUC'] > lgb_metrics['PR-AUC']:
        best_model = xgb_model
        best_name = 'XGBoost'
    else:
        best_model = lgb_model
        best_name = 'LightGBM'

    print(f"\n🏆 Best Model: {best_name} (PR-AUC: {max(xgb_metrics['PR-AUC'], lgb_metrics['PR-AUC']):.4f})")

    print("\n" + "="*60)
    print("✓ ADVANCED MODELING COMPLETE!")
    print("="*60)

    return xgb_model, lgb_model, comparison, best_model, best_name


if __name__ == '__main__':
    results = main()


