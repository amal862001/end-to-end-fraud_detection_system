"""
Experiment Tracking with MLflow and DagsHub

This module provides comprehensive experiment tracking for the fraud detection project:
- Parameters (hyperparameters, data splits, preprocessing)
- Metrics (PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix)
- Artifacts (models, plots, feature importance, data)
- Model registry for versioning and deployment

Features:
- Class-based API for easy tracking
- DagsHub integration for remote tracking
- Automatic plot generation and logging
- Feature importance tracking
- Model versioning and registry
- Experiment comparison utilities

Author: Your Name
Date: 2026-01-15
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import dagshub
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
from datetime import datetime
import joblib


def setup_mlflow(experiment_name="fraud-detection"):
    """
    Setup MLflow with DagsHub integration.
    
    Args:
        experiment_name: Name of the experiment
    """
    print("="*60)
    print("SETTING UP MLFLOW + DAGSHUB")
    print("="*60)
    
    # Initialize DagsHub (optional - only if you have DagsHub account)
    # Uncomment these lines if you want to use DagsHub:
    # dagshub.init(repo_owner='your-username', 
    #              repo_name='fraud-detection', 
    #              mlflow=True)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    print(f"\n✓ MLflow experiment: {experiment_name}")
    print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")
    
    return experiment_name


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    # PR-AUC (most important for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Other metrics
    metrics = {
        # Main metrics
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        
        # Confusion matrix
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        
        # Additional metrics
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    
    return metrics


def create_confusion_matrix_plot(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Create and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_pr_curve_plot(y_true, y_pred_proba, save_path='pr_curve.png'):
    """
    Create and save Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_feature_importance_plot(model, feature_names, top_n=20, save_path='feature_importance.png'):
    """
    Create and save feature importance plot.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("⚠ Model doesn't have feature_importances_")
        return None

    # Create DataFrame
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_imp)), feature_imp['importance'])
    plt.yticks(range(len(feature_imp)), feature_imp['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# CLASS-BASED MLFLOW TRACKER
# ============================================================================

class MLflowTracker:
    """
    Comprehensive MLflow experiment tracker with DagsHub integration.

    This class provides a high-level API for tracking ML experiments including:
    - Parameters (hyperparameters, data configuration, preprocessing)
    - Metrics (PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix)
    - Artifacts (models, plots, feature importance, datasets)
    - Model registry for versioning

    Example:
        >>> tracker = MLflowTracker(experiment_name="fraud-detection")
        >>> tracker.start_run(run_name="lightgbm-v1")
        >>> tracker.log_params({"max_depth": 9, "learning_rate": 0.05})
        >>> tracker.log_metrics({"pr_auc": 0.8689, "roc_auc": 0.9684})
        >>> tracker.log_model(model, "lightgbm")
        >>> tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str = "fraud-detection",
        tracking_uri: Optional[str] = None,
        dagshub_repo: Optional[str] = None,
        dagshub_user: Optional[str] = None,
        dagshub_token: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (default: local ./mlruns)
            dagshub_repo: DagsHub repository name (e.g., "fraud-detection")
            dagshub_user: DagsHub username (or set DAGSHUB_USER env var)
            dagshub_token: DagsHub token (or set DAGSHUB_TOKEN env var)
        """
        self.experiment_name = experiment_name
        self.run = None
        self.artifacts_dir = Path("mlflow_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)

        print("="*60)
        print("INITIALIZING MLFLOW TRACKER")
        print("="*60)

        # Setup tracking URI
        if dagshub_repo:
            self._setup_dagshub(dagshub_repo, dagshub_user, dagshub_token)
        elif tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"✓ Custom tracking URI: {tracking_uri}")
        else:
            # Use local tracking
            mlflow.set_tracking_uri("file:./mlruns")
            print(f"✓ Local tracking: ./mlruns")

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created new experiment: {experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            print(f"✓ Using existing experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)
        print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")
        print("="*60 + "\n")

    def _setup_dagshub(
        self,
        repo: str,
        user: Optional[str] = None,
        token: Optional[str] = None
    ):
        """Setup DagsHub integration."""
        # Get credentials from args or environment
        dagshub_user = user or os.getenv('DAGSHUB_USER')
        dagshub_token = token or os.getenv('DAGSHUB_TOKEN')

        if dagshub_user and dagshub_token:
            # Initialize DagsHub
            try:
                dagshub.init(repo_owner=dagshub_user, repo_name=repo, mlflow=True)
                print(f"✓ DagsHub integration enabled")
                print(f"  Repository: {dagshub_user}/{repo}")
                print(f"  MLflow UI: https://dagshub.com/{dagshub_user}/{repo}/experiments")
            except Exception as e:
                print(f"⚠ DagsHub initialization failed: {e}")
                print("  Falling back to local tracking")
                mlflow.set_tracking_uri("file:./mlruns")
        else:
            print("⚠ DagsHub credentials not found")
            print("  Set DAGSHUB_USER and DAGSHUB_TOKEN environment variables")
            print("  Or pass them as arguments")
            print("  Falling back to local tracking")
            mlflow.set_tracking_uri("file:./mlruns")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Dictionary of tags to add to the run
            description: Description of the run

        Returns:
            MLflow run object
        """
        self.run = mlflow.start_run(run_name=run_name)

        # Add default tags
        default_tags = {
            "project": "fraud-detection",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mlflow.note.content": description or "Fraud detection experiment"
        }

        if tags:
            default_tags.update(tags)

        for key, value in default_tags.items():
            mlflow.set_tag(key, value)

        print(f"\n{'='*60}")
        print(f"MLFLOW RUN STARTED")
        print(f"{'='*60}")
        print(f"Run ID: {self.run.info.run_id}")
        if run_name:
            print(f"Run Name: {run_name}")
        if description:
            print(f"Description: {description}")
        print(f"{'='*60}\n")

        return self.run

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")

        for key, value in params.items():
            # MLflow has limits on param value length (250 chars)
            str_value = str(value)
            if len(str_value) > 250:
                str_value = str_value[:247] + "..."
            mlflow.log_param(key, str_value)

        print(f"✓ Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for tracking metrics over time
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")

        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(key, float(value), step=step)

        print(f"✓ Logged {len(metrics)} metrics")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None
    ):
        """
        Log a model to MLflow.

        Args:
            model: Trained model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: Name for model registry (optional)
            signature: MLflow model signature (optional)
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")

        # Determine model type and log accordingly
        model_type = type(model).__name__

        if 'LGBMClassifier' in model_type or 'LGBMRegressor' in model_type:
            mlflow.lightgbm.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                signature=signature
            )
        elif 'XGBClassifier' in model_type or 'XGBRegressor' in model_type:
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                signature=signature
            )
        else:
            # Default to sklearn
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                signature=signature
            )

        print(f"✓ Logged model: {artifact_path}")
        if registered_model_name:
            print(f"  Registered as: {registered_model_name}")

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        """
        Log an artifact (file) to MLflow.

        Args:
            local_path: Path to the local file
            artifact_path: Path within the run's artifact directory
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")

        mlflow.log_artifact(str(local_path), artifact_path)
        print(f"✓ Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Name of the JSON file
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")

        # Save to temp file
        temp_path = self.artifacts_dir / filename
        with open(temp_path, 'w') as f:
            json.dump(dictionary, f, indent=2, default=str)

        # Log to MLflow
        mlflow.log_artifact(str(temp_path))
        print(f"✓ Logged dictionary: {filename}")

    def log_figure(self, figure: plt.Figure, filename: str):
        """
        Log a matplotlib figure as an artifact.

        Args:
            figure: Matplotlib figure
            filename: Name of the image file
        """
        if not self.run:
            raise RuntimeError("No active run. Call start_run() first.")

        # Save figure
        temp_path = self.artifacts_dir / filename
        figure.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close(figure)

        # Log to MLflow
        mlflow.log_artifact(str(temp_path))
        print(f"✓ Logged figure: {filename}")

    def log_confusion_matrix(self, y_true, y_pred):
        """
        Create and log confusion matrix plot.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm_path = create_confusion_matrix_plot(
            y_true, y_pred,
            save_path=str(self.artifacts_dir / 'confusion_matrix.png')
        )
        mlflow.log_artifact(cm_path)
        print(f"✓ Logged confusion matrix")

    def log_pr_curve(self, y_true, y_pred_proba):
        """
        Create and log Precision-Recall curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        pr_path = create_pr_curve_plot(
            y_true, y_pred_proba,
            save_path=str(self.artifacts_dir / 'pr_curve.png')
        )
        mlflow.log_artifact(pr_path)
        print(f"✓ Logged PR curve")

    def log_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20
    ):
        """
        Create and log feature importance plot.

        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            top_n: Number of top features to show
        """
        fi_path = create_feature_importance_plot(
            model, feature_names, top_n,
            save_path=str(self.artifacts_dir / 'feature_importance.png')
        )

        if fi_path:
            mlflow.log_artifact(fi_path)

            # Also log as CSV
            if hasattr(model, 'feature_importances_'):
                feature_imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                csv_path = self.artifacts_dir / 'feature_importance.csv'
                feature_imp_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(str(csv_path))

                print(f"✓ Logged feature importance (plot + CSV)")

    def log_complete_evaluation(
        self,
        model: Any,
        X_test,
        y_test,
        feature_names: List[str],
        model_name: str = "model"
    ):
        """
        Log complete model evaluation including metrics, plots, and feature importance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            model_name: Name for the model artifact
        """
        print(f"\n{'='*60}")
        print(f"LOGGING COMPLETE EVALUATION")
        print(f"{'='*60}\n")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        # Log metrics
        self.log_metrics(metrics)

        # Log plots
        self.log_confusion_matrix(y_test, y_pred)
        self.log_pr_curve(y_test, y_pred_proba)

        # Log feature importance
        self.log_feature_importance(model, feature_names)

        # Log model
        self.log_model(model, model_name)

        print(f"\n{'='*60}")
        print(f"✓ COMPLETE EVALUATION LOGGED")
        print(f"{'='*60}\n")

        return metrics

    def end_run(self):
        """End the current MLflow run."""
        if self.run:
            mlflow.end_run()
            print(f"\n{'='*60}")
            print(f"MLFLOW RUN ENDED")
            print(f"{'='*60}")
            print(f"Run ID: {self.run.info.run_id}")
            print(f"{'='*60}\n")
            self.run = None
        else:
            print("⚠ No active run to end")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def track_experiment(
    experiment_name: str,
    run_name: str,
    model: Any,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: List[str],
    params: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
    dagshub_repo: Optional[str] = None
):
    """
    Convenience function to track a complete experiment.

    Args:
        experiment_name: Name of the experiment
        run_name: Name for this run
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        params: Dictionary of parameters
        tags: Optional tags
        dagshub_repo: Optional DagsHub repository

    Returns:
        Dictionary of metrics
    """
    tracker = MLflowTracker(
        experiment_name=experiment_name,
        dagshub_repo=dagshub_repo
    )

    tracker.start_run(run_name=run_name, tags=tags)

    # Log parameters
    tracker.log_params(params)

    # Log data info
    data_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_frauds': int(y_train.sum()),
        'test_frauds': int(y_test.sum()),
        'n_features': X_train.shape[1]
    }
    tracker.log_params(data_info)

    # Log complete evaluation
    metrics = tracker.log_complete_evaluation(
        model, X_test, y_test, feature_names, model_name=run_name
    )

    tracker.end_run()

    return metrics

