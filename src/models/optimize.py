"""
Hyperparameter Optimization using Optuna
Simple implementation for fraud detection.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
import lightgbm as lgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import load_data


def pr_auc_score(y_true, y_pred_proba, **kwargs):
    """Calculate PR-AUC score."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def prepare_data(test_size=0.2, val_size=0.1, random_state=42):
    """Load and prepare data with train/val/test split."""
    print("Loading data...")
    X, y = load_data()

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate validation set from training
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to preserve feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print(f"Train set: {X_train.shape[0]} samples ({y_train.sum()} frauds)")
    print(f"Val set: {X_val.shape[0]} samples ({y_val.sum()} frauds)")
    print(f"Test set: {X_test.shape[0]} samples ({y_test.sum()} frauds)")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function.

    This function defines the hyperparameter search space
    and returns the metric to optimize (PR-AUC).

    Note: Uses a validation set instead of CV for faster, more reliable optimization.
    """

    # Define hyperparameter search space (narrower ranges based on LightGBM defaults)
    params = {
        # Tree structure
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),

        # Regularization
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),

        # Sampling
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),

        # Fixed parameters
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params['scale_pos_weight'] = scale_pos_weight

    # Train model on training set
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    score = pr_auc_score(y_val, y_pred_proba)

    return score


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Run Optuna optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials (default: 50)

    Returns:
        study: Optuna study object with results
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*60)
    print(f"\nOptimizing PR-AUC with {n_trials} trials...")
    print("Using validation set for evaluation (more reliable than CV)")
    print("This may take 5-10 minutes...\n")

    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize PR-AUC
        study_name='fraud_detection_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )

    return study


def print_results(study):
    """Print optimization results."""
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    # Check if any trials completed successfully
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(completed_trials) == 0:
        print("\n❌ No trials completed successfully!")
        print("This might be due to:")
        print("  1. All trials failed during cross-validation")
        print("  2. Issues with the scoring function")
        print("  3. Data or parameter issues")
        return False

    print(f"\nBest PR-AUC: {study.best_value:.4f}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Completed trials: {len(completed_trials)}")
    print(f"Failed trials: {len(study.trials) - len(completed_trials)}")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")
        else:
            print(f"  {key:20s}: {value}")

    return True


def train_best_model(X_train, y_train, X_test, y_test, best_params):
    """Train final model with best parameters."""
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*60)

    # Add fixed parameters
    params = best_params.copy()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params['scale_pos_weight'] = scale_pos_weight
    params['random_state'] = 42
    params['n_jobs'] = -1
    params['verbose'] = -1

    # Train model
    print("\nTraining...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_pr_auc = pr_auc_score(y_test, y_pred_proba)

    print(f"\nTest PR-AUC: {test_pr_auc:.4f}")

    return model, test_pr_auc


def save_results(study, model, scaler, test_pr_auc, output_dir='models/optimized'):
    """Save optimization results and best model."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save model
    model_path = os.path.join(output_dir, 'best_model.pkl')
    joblib.dump(model, model_path)
    print(f"✓ Saved model to {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to {scaler_path}")

    # Save best parameters
    params_path = os.path.join(output_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"✓ Saved parameters to {params_path}")

    # Save optimization summary
    summary = {
        'best_pr_auc_cv': study.best_value,
        'best_pr_auc_test': test_pr_auc,
        'n_trials': len(study.trials),
        'best_params': study.best_params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    summary_path = os.path.join(output_dir, 'optimization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(output_dir, 'trials_history.csv')
    trials_df.to_csv(trials_path, index=False)
    print(f"✓ Saved trial history to {trials_path}")


def compare_with_baseline(test_pr_auc):
    """Compare optimized model with baseline."""
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)

    # Baseline results (from previous training)
    baseline_pr_auc = 0.8689  # LightGBM from advanced_models.py

    comparison = pd.DataFrame({
        'Model': ['LightGBM (Default)', 'LightGBM (Optimized)'],
        'PR-AUC': [baseline_pr_auc, test_pr_auc],
        'Improvement': [0, test_pr_auc - baseline_pr_auc]
    })

    print("\n", comparison.to_string(index=False))

    if test_pr_auc > baseline_pr_auc:
        improvement = ((test_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100
        print(f"\n✓ Optimization improved PR-AUC by {improvement:.2f}%")
    else:
        print("\n⚠ Default parameters were already good!")
        print("  This is common - LightGBM has excellent defaults.")


def main():
    """Main optimization pipeline."""
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*60)
    print("\nObjective: Find best hyperparameters for fraud detection")
    print("Metric: PR-AUC (Precision-Recall AUC)")
    print("Algorithm: LightGBM")
    print("Optimizer: Optuna (TPE Sampler)")

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data()

    # Run optimization
    n_trials = 50  # Adjust based on time available
    study = optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=n_trials)

    # Print results
    success = print_results(study)

    if not success:
        print("\n❌ Optimization failed. Please check the errors above.")
        return None, None

    # Train final model with best parameters
    model, test_pr_auc = train_best_model(
        X_train, y_train, X_test, y_test, study.best_params
    )

    # Compare with baseline
    compare_with_baseline(test_pr_auc)

    # Save results
    save_results(study, model, scaler, test_pr_auc)

    print("\n" + "="*60)
    print("✓ OPTIMIZATION COMPLETE!")
    print("="*60)

    print("\n💡 Next steps:")
    print("   1. Review best parameters in models/optimized/best_params.json")
    print("   2. Check trial history in models/optimized/trials_history.csv")
    print("   3. Use optimized model: joblib.load('models/optimized/best_model.pkl')")

    return study, model


if __name__ == '__main__':
    results = main()


