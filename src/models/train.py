"""
Baseline Model Training
Train and evaluate baseline models for fraud detection.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score
)
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

    # Scale features (Time and Amount need scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud rate - Train: {y_train.mean()*100:.2f}%, Test: {y_test.mean()*100:.2f}%")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn
    }

    return metrics


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression baseline."""
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)

    # Train model with class weights to handle imbalance
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    print("Training...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Print results
    print("\nResults:")
    print(f"  PR-AUC:    {metrics['PR-AUC']:.4f}")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['True Positives']:4d}  |  FP: {metrics['False Positives']:4d}")
    print(f"  FN: {metrics['False Negatives']:4d}  |  TN: {metrics['True Negatives']:4d}")

    return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest baseline."""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)

    # Train model with class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("Training...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Print results
    print("\nResults:")
    print(f"  PR-AUC:    {metrics['PR-AUC']:.4f}")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['True Positives']:4d}  |  FP: {metrics['False Positives']:4d}")
    print(f"  FN: {metrics['False Negatives']:4d}  |  TN: {metrics['True Negatives']:4d}")

    return model, metrics


def compare_models(results):
    """Compare model performance."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    # Create comparison dataframe
    comparison = pd.DataFrame(results).T
    comparison = comparison[['PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']]

    print("\n", comparison.to_string())

    # Find best model
    best_model = comparison['PR-AUC'].idxmax()
    print(f"\n✓ Best Model (by PR-AUC): {best_model}")

    return comparison


def save_models(models, scaler, output_dir='models/baseline'):
    """Save trained models."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)

    for name, model in models.items():
        filepath = os.path.join(output_dir, f'{name.lower().replace(" ", "_")}.pkl')
        joblib.dump(model, filepath)
        print(f"✓ Saved {name} to {filepath}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to {scaler_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # Train models
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)

    # Compare models
    results = {
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics
    }
    comparison = compare_models(results)

    # Save models
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    save_models(models, scaler)

    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)

    return models, results, comparison


if __name__ == '__main__':
    models, results, comparison = main()
