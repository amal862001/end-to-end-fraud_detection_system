"""
Final Model Selection & Threshold Tuning

This script:
1. Loads and evaluates all trained models
2. Compares performance across all models
3. Performs precision-recall tradeoff analysis
4. Tunes optimal threshold for production
5. Locks final production model with threshold

Author: Your Name
Date: 2026-01-15
"""

import sys
sys.path.append('src')

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.load_data import load_data

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_all_models():
    """Load all trained models."""
    print("="*60)
    print("LOADING ALL TRAINED MODELS")
    print("="*60)
    
    models = {}
    
    # 1. Logistic Regression (Baseline)
    try:
        models['Logistic Regression'] = {
            'model': joblib.load('models/baseline/logistic_regression.pkl'),
            'scaler': joblib.load('models/baseline/scaler.pkl'),
            'type': 'baseline'
        }
        print("✓ Loaded: Logistic Regression (Baseline)")
    except FileNotFoundError:
        print("⚠ Logistic Regression not found")
    
    # 2. Random Forest (Baseline)
    try:
        models['Random Forest'] = {
            'model': joblib.load('models/baseline/random_forest.pkl'),
            'scaler': joblib.load('models/baseline/scaler.pkl'),
            'type': 'baseline'
        }
        print("✓ Loaded: Random Forest (Baseline)")
    except FileNotFoundError:
        print("⚠ Random Forest not found")
    
    # 3. XGBoost (Advanced)
    try:
        models['XGBoost'] = {
            'model': joblib.load('models/advanced/xgboost.pkl'),
            'scaler': joblib.load('models/advanced/scaler.pkl'),
            'type': 'advanced'
        }
        print("✓ Loaded: XGBoost (Advanced)")
    except FileNotFoundError:
        print("⚠ XGBoost not found")
    
    # 4. LightGBM Default (Advanced)
    try:
        # Try direct path first
        lgbm_path = 'models/advanced/lightgbm.pkl'
        if os.path.exists(lgbm_path):
            models['LightGBM (Default)'] = {
                'model': joblib.load(lgbm_path),
                'scaler': joblib.load('models/advanced/scaler.pkl'),
                'type': 'advanced'
            }
            print("✓ Loaded: LightGBM (Default)")
        else:
            # Try to find any LightGBM file in advanced folder
            if os.path.exists('models/advanced'):
                advanced_files = os.listdir('models/advanced')
                lgbm_files = [f for f in advanced_files if 'lightgbm' in f.lower() or 'lgbm' in f.lower()]

                if lgbm_files:
                    models['LightGBM (Default)'] = {
                        'model': joblib.load(f'models/advanced/{lgbm_files[0]}'),
                        'scaler': joblib.load('models/advanced/scaler.pkl'),
                        'type': 'advanced'
                    }
                    print(f"✓ Loaded: LightGBM (Default) from {lgbm_files[0]}")
                else:
                    print("⚠ LightGBM (Default) not found")
            else:
                print("⚠ LightGBM (Default) not found")
    except (FileNotFoundError, IndexError) as e:
        print(f"⚠ LightGBM (Default) not found: {e}")
    
    # 5. LightGBM Optimized (Optuna)
    try:
        models['LightGBM (Optimized)'] = {
            'model': joblib.load('models/optimized/best_model.pkl'),
            'scaler': joblib.load('models/optimized/scaler.pkl'),
            'type': 'optimized',
            'params': json.load(open('models/optimized/best_params.json'))
        }
        print("✓ Loaded: LightGBM (Optimized)")
    except FileNotFoundError:
        print("⚠ LightGBM (Optimized) not found")
    
    print(f"\n✓ Total models loaded: {len(models)}")
    return models


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive evaluation metrics."""
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    metrics = {
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    
    return metrics


def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models on test set."""
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)

    results = {}
    predictions = {}

    for name, model_info in models.items():
        print(f"\nEvaluating: {name}...")

        # Get model and scaler
        model = model_info['model']
        scaler = model_info['scaler']

        # Scale features
        X_test_scaled = scaler.transform(X_test)

        # Preserve feature names for models that need them (e.g., LightGBM)
        if hasattr(X_test, 'columns'):
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        results[name] = metrics
        predictions[name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # Print results
        print(f"  PR-AUC:    {metrics['PR-AUC']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1-Score']:.4f}")

    return results, predictions


def compare_models(results):
    """Create comprehensive model comparison."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    # Create comparison dataframe
    df = pd.DataFrame(results).T

    # Select key metrics
    key_metrics = ['PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'FPR', 'FNR']
    comparison = df[key_metrics].copy()

    # Sort by PR-AUC (most important for imbalanced data)
    comparison = comparison.sort_values('PR-AUC', ascending=False)

    print("\n" + comparison.to_string())

    # Find best model
    best_model = comparison.index[0]
    best_pr_auc = comparison.loc[best_model, 'PR-AUC']

    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"{'='*60}")
    print(f"  PR-AUC:    {best_pr_auc:.4f} ⭐")
    print(f"  ROC-AUC:   {comparison.loc[best_model, 'ROC-AUC']:.4f}")
    print(f"  Precision: {comparison.loc[best_model, 'Precision']:.4f}")
    print(f"  Recall:    {comparison.loc[best_model, 'Recall']:.4f}")
    print(f"  F1-Score:  {comparison.loc[best_model, 'F1-Score']:.4f}")

    return comparison, best_model


def plot_model_comparison(results, save_path='models/production'):
    """Create visualization comparing all models."""
    print("\n" + "="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)

    os.makedirs(save_path, exist_ok=True)

    # Prepare data
    df = pd.DataFrame(results).T
    df = df.sort_values('PR-AUC', ascending=False)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold')

    # 1. PR-AUC Comparison
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df))]
    bars = ax1.barh(df.index, df['PR-AUC'], color=colors)
    ax1.set_xlabel('PR-AUC Score', fontweight='bold')
    ax1.set_title('PR-AUC (Primary Metric)', fontweight='bold')
    ax1.set_xlim(0, 1)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center', fontweight='bold')

    # 2. ROC-AUC Comparison
    ax2 = axes[0, 1]
    bars = ax2.barh(df.index, df['ROC-AUC'], color=colors)
    ax2.set_xlabel('ROC-AUC Score', fontweight='bold')
    ax2.set_title('ROC-AUC', fontweight='bold')
    ax2.set_xlim(0, 1)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center')

    # 3. Precision vs Recall
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['Recall'], df['Precision'],
                         s=300, c=range(len(df)), cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=2)
    for idx, name in enumerate(df.index):
        ax3.annotate(name, (df.loc[name, 'Recall'], df.loc[name, 'Precision']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Recall', fontweight='bold')
    ax3.set_ylabel('Precision', fontweight='bold')
    ax3.set_title('Precision vs Recall Tradeoff', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. F1-Score Comparison
    ax4 = axes[1, 1]
    bars = ax4.barh(df.index, df['F1-Score'], color=colors)
    ax4.set_xlabel('F1-Score', fontweight='bold')
    ax4.set_title('F1-Score', fontweight='bold')
    ax4.set_xlim(0, 1)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_path, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {plot_path}")
    plt.close()


def analyze_precision_recall_tradeoff(y_test, y_pred_proba, model_name, save_path='models/production'):
    """Analyze precision-recall tradeoff at different thresholds."""
    print("\n" + "="*60)
    print(f"PRECISION-RECALL TRADEOFF ANALYSIS: {model_name}")
    print("="*60)

    os.makedirs(save_path, exist_ok=True)

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Add threshold 1.0 for completeness
    thresholds = np.append(thresholds, 1.0)

    # Create detailed analysis at key thresholds
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\nThreshold Analysis:")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 80)

    threshold_results = {}

    for thresh in test_thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        threshold_results[thresh] = {
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn)
        }

        print(f"{thresh:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} "
              f"{tp:<8d} {fp:<8d} {fn:<8d}")

    print("-" * 80)

    # Plot precision-recall tradeoff
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Precision-Recall Tradeoff Analysis: {model_name}',
                 fontsize=14, fontweight='bold')

    # 1. Precision-Recall Curve
    ax1 = axes[0]
    ax1.plot(recall, precision, linewidth=2, label=f'{model_name}')
    ax1.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add PR-AUC to plot
    pr_auc = auc(recall, precision)
    ax1.text(0.6, 0.95, f'PR-AUC = {pr_auc:.4f}',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=12, fontweight='bold')

    # 2. Metrics vs Threshold
    ax2 = axes[1]
    df_thresh = pd.DataFrame(threshold_results).T
    ax2.plot(df_thresh.index, df_thresh['Precision'], 'o-', label='Precision', linewidth=2)
    ax2.plot(df_thresh.index, df_thresh['Recall'], 's-', label='Recall', linewidth=2)
    ax2.plot(df_thresh.index, df_thresh['F1-Score'], '^-', label='F1-Score', linewidth=2)
    ax2.set_xlabel('Threshold', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax2.set_title('Metrics vs Threshold', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_path, f'{model_name.replace(" ", "_")}_threshold_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved threshold analysis plot: {plot_path}")
    plt.close()

    return threshold_results


def select_optimal_threshold(threshold_results, strategy='balanced'):
    """
    Select optimal threshold based on strategy.

    Strategies:
    - 'balanced': Maximize F1-Score
    - 'high_recall': Prioritize recall (catch more frauds)
    - 'high_precision': Prioritize precision (fewer false alarms)
    - 'business': Custom business logic (e.g., recall >= 0.85)
    """
    print("\n" + "="*60)
    print(f"OPTIMAL THRESHOLD SELECTION: {strategy.upper()} STRATEGY")
    print("="*60)

    df = pd.DataFrame(threshold_results).T

    if strategy == 'balanced':
        # Maximize F1-Score
        optimal_threshold = df['F1-Score'].idxmax()
        print("\nStrategy: Maximize F1-Score (balanced precision-recall)")

    elif strategy == 'high_recall':
        # Maximize recall while maintaining reasonable precision (>= 0.5)
        df_filtered = df[df['Precision'] >= 0.5]
        if len(df_filtered) > 0:
            optimal_threshold = df_filtered['Recall'].idxmax()
        else:
            optimal_threshold = df['Recall'].idxmax()
        print("\nStrategy: Maximize Recall (catch more frauds)")
        print("Constraint: Precision >= 0.5")

    elif strategy == 'high_precision':
        # Maximize precision while maintaining reasonable recall (>= 0.7)
        df_filtered = df[df['Recall'] >= 0.7]
        if len(df_filtered) > 0:
            optimal_threshold = df_filtered['Precision'].idxmax()
        else:
            optimal_threshold = df['Precision'].idxmax()
        print("\nStrategy: Maximize Precision (fewer false alarms)")
        print("Constraint: Recall >= 0.7")

    elif strategy == 'business':
        # Business requirement: Recall >= 0.85, maximize precision
        df_filtered = df[df['Recall'] >= 0.85]
        if len(df_filtered) > 0:
            optimal_threshold = df_filtered['Precision'].idxmax()
        else:
            # Fallback to highest recall
            optimal_threshold = df['Recall'].idxmax()
        print("\nStrategy: Business Requirements")
        print("Requirement: Recall >= 0.85 (catch 85% of frauds)")
        print("Objective: Maximize Precision")

    else:
        # Default to balanced
        optimal_threshold = df['F1-Score'].idxmax()
        print("\nStrategy: Default (Balanced)")

    # Get metrics at optimal threshold
    optimal_metrics = threshold_results[optimal_threshold]

    print(f"\n{'='*60}")
    print(f"✓ OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
    print(f"{'='*60}")
    print(f"  Precision: {optimal_metrics['Precision']:.4f}")
    print(f"  Recall:    {optimal_metrics['Recall']:.4f}")
    print(f"  F1-Score:  {optimal_metrics['F1-Score']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {optimal_metrics['TP']:4d}  |  FP: {optimal_metrics['FP']:4d}")
    print(f"    FN: {optimal_metrics['FN']:4d}  |  TN: {optimal_metrics['TN']:4d}")

    return optimal_threshold, optimal_metrics


def save_production_model(model_name, model_info, optimal_threshold, optimal_metrics,
                          all_results, save_path='models/production'):
    """Save final production model with all metadata."""
    print("\n" + "="*60)
    print("SAVING PRODUCTION MODEL")
    print("="*60)

    os.makedirs(save_path, exist_ok=True)

    # 1. Save model
    model_path = os.path.join(save_path, 'production_model.pkl')
    joblib.dump(model_info['model'], model_path)
    print(f"✓ Saved model: {model_path}")

    # 2. Save scaler
    scaler_path = os.path.join(save_path, 'production_scaler.pkl')
    joblib.dump(model_info['scaler'], scaler_path)
    print(f"✓ Saved scaler: {scaler_path}")

    # 3. Save threshold
    threshold_path = os.path.join(save_path, 'production_threshold.json')
    threshold_config = {
        'optimal_threshold': float(optimal_threshold),
        'default_threshold': 0.5,
        'metrics_at_optimal': {
            'Precision': float(optimal_metrics['Precision']),
            'Recall': float(optimal_metrics['Recall']),
            'F1-Score': float(optimal_metrics['F1-Score']),
            'TP': int(optimal_metrics['TP']),
            'FP': int(optimal_metrics['FP']),
            'FN': int(optimal_metrics['FN']),
            'TN': int(optimal_metrics['TN'])
        }
    }
    with open(threshold_path, 'w', encoding='utf-8') as f:
        json.dump(threshold_config, f, indent=2)
    print(f"✓ Saved threshold config: {threshold_path}")

    # 4. Save model metadata
    metadata_path = os.path.join(save_path, 'production_metadata.json')
    metadata = {
        'model_name': model_name,
        'model_type': model_info['type'],
        'selection_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'performance_metrics': {
            'PR-AUC': float(all_results[model_name]['PR-AUC']),
            'ROC-AUC': float(all_results[model_name]['ROC-AUC']),
            'Precision': float(all_results[model_name]['Precision']),
            'Recall': float(all_results[model_name]['Recall']),
            'F1-Score': float(all_results[model_name]['F1-Score']),
            'Accuracy': float(all_results[model_name]['Accuracy']),
            'FPR': float(all_results[model_name]['FPR']),
            'FNR': float(all_results[model_name]['FNR'])
        },
        'optimal_threshold': float(optimal_threshold),
        'version': '1.0.0'
    }

    # Add parameters if available
    if 'params' in model_info:
        metadata['hyperparameters'] = model_info['params']

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    # 5. Save model comparison
    comparison_path = os.path.join(save_path, 'model_comparison.csv')
    df_comparison = pd.DataFrame(all_results).T
    df_comparison.to_csv(comparison_path)
    print(f"✓ Saved comparison: {comparison_path}")

    # 6. Create README
    readme_path = os.path.join(save_path, 'README.md')
    readme_content = f"""# Production Model

## Model Information

**Model:** {model_name}
**Type:** {model_info['type']}
**Version:** 1.0.0
**Selection Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Performance Metrics

| Metric | Score |
|--------|-------|
| **PR-AUC** | {all_results[model_name]['PR-AUC']:.4f} ⭐ |
| **ROC-AUC** | {all_results[model_name]['ROC-AUC']:.4f} |
| **Precision** | {all_results[model_name]['Precision']:.4f} |
| **Recall** | {all_results[model_name]['Recall']:.4f} |
| **F1-Score** | {all_results[model_name]['F1-Score']:.4f} |
| **Accuracy** | {all_results[model_name]['Accuracy']:.4f} |

---

## Optimal Threshold

**Threshold:** {optimal_threshold:.2f}

At this threshold:
- **Precision:** {optimal_metrics['Precision']:.4f}
- **Recall:** {optimal_metrics['Recall']:.4f}
- **F1-Score:** {optimal_metrics['F1-Score']:.4f}

**Confusion Matrix:**
```
TP: {optimal_metrics['TP']:4d}  |  FP: {optimal_metrics['FP']:4d}
FN: {optimal_metrics['FN']:4d}  |  TN: {optimal_metrics['TN']:4d}
```

---

## Usage

### Load Model

```python
import joblib
import json

# Load model and scaler
model = joblib.load('models/production/production_model.pkl')
scaler = joblib.load('models/production/production_scaler.pkl')

# Load threshold
with open('models/production/production_threshold.json', 'r') as f:
    threshold_config = json.load(f)
    optimal_threshold = threshold_config['optimal_threshold']
```

### Make Predictions

```python
# Scale features
X_scaled = scaler.transform(X)

# Get probabilities
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Apply optimal threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)
```

---

## Files

- `production_model.pkl` - Trained model
- `production_scaler.pkl` - Feature scaler
- `production_threshold.json` - Optimal threshold configuration
- `production_metadata.json` - Model metadata and metrics
- `model_comparison.csv` - Comparison with other models
- `model_comparison.png` - Visual comparison
- `README.md` - This file

---

**Model is locked and ready for production deployment!** 🚀
"""

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Saved README: {readme_path}")

    print(f"\n{'='*60}")
    print("✓ PRODUCTION MODEL SAVED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nLocation: {save_path}/")
    print("\nFiles created:")
    print("  - production_model.pkl")
    print("  - production_scaler.pkl")
    print("  - production_threshold.json")
    print("  - production_metadata.json")
    print("  - model_comparison.csv")
    print("  - README.md")


def main(threshold_strategy='balanced'):
    """
    Main function for final model selection and threshold tuning.

    Args:
        threshold_strategy: Strategy for threshold selection
            - 'balanced': Maximize F1-Score (default)
            - 'high_recall': Prioritize recall
            - 'high_precision': Prioritize precision
            - 'business': Business requirements (recall >= 0.85)
    """
    print("\n" + "="*60)
    print("FINAL MODEL SELECTION & THRESHOLD TUNING")
    print("="*60)
    print("\nObjective: Choose production model with optimal threshold")
    print(f"Strategy: {threshold_strategy}")

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 1: LOAD DATA")
    print("="*60)

    X, y = load_data()

    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTest set: {len(X_test)} samples, {y_test.sum()} frauds ({y_test.mean()*100:.2f}%)")

    # ========================================================================
    # 2. LOAD ALL MODELS
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 2: LOAD ALL TRAINED MODELS")
    print("="*60)

    models = load_all_models()

    if len(models) == 0:
        print("\n❌ No models found! Please train models first:")
        print("   python src/models/train.py")
        print("   python src/models/advanced_models.py")
        return

    # ========================================================================
    # 3. EVALUATE ALL MODELS
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 3: EVALUATE ALL MODELS")
    print("="*60)

    results, predictions = evaluate_all_models(models, X_test, y_test)

    # ========================================================================
    # 4. COMPARE MODELS
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 4: COMPARE MODELS")
    print("="*60)

    comparison, best_model_name = compare_models(results)

    # Create comparison plots
    plot_model_comparison(results)

    # ========================================================================
    # 5. THRESHOLD TUNING FOR BEST MODEL
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 5: THRESHOLD TUNING")
    print("="*60)

    # Get predictions for best model
    best_pred_proba = predictions[best_model_name]['y_pred_proba']

    # Analyze precision-recall tradeoff
    threshold_results = analyze_precision_recall_tradeoff(
        y_test, best_pred_proba, best_model_name
    )

    # Select optimal threshold
    optimal_threshold, optimal_metrics = select_optimal_threshold(
        threshold_results, strategy=threshold_strategy
    )

    # ========================================================================
    # 6. SAVE PRODUCTION MODEL
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 6: LOCK PRODUCTION MODEL")
    print("="*60)

    save_production_model(
        best_model_name,
        models[best_model_name],
        optimal_threshold,
        optimal_metrics,
        results
    )

    # ========================================================================
    # 7. FINAL SUMMARY
    # ========================================================================

    print("\n" + "="*60)
    print("✓ FINAL MODEL SELECTION COMPLETE!")
    print("="*60)

    print(f"\n🏆 PRODUCTION MODEL: {best_model_name}")
    print(f"   PR-AUC:    {results[best_model_name]['PR-AUC']:.4f} ⭐")
    print(f"   ROC-AUC:   {results[best_model_name]['ROC-AUC']:.4f}")
    print(f"   Threshold: {optimal_threshold:.2f}")

    print(f"\n📊 PERFORMANCE AT OPTIMAL THRESHOLD:")
    print(f"   Precision: {optimal_metrics['Precision']:.4f}")
    print(f"   Recall:    {optimal_metrics['Recall']:.4f}")
    print(f"   F1-Score:  {optimal_metrics['F1-Score']:.4f}")

    print(f"\n📁 SAVED TO: models/production/")

    print("\n💡 NEXT STEPS:")
    print("   1. Review production model: models/production/README.md")
    print("   2. Test model: python examples/use_production_model.py")
    print("   3. Deploy model: Use FastAPI or cloud deployment")
    print("   4. Monitor performance: Track metrics in production")

    print("\n" + "="*60 + "\n")

    return {
        'best_model': best_model_name,
        'optimal_threshold': optimal_threshold,
        'metrics': optimal_metrics,
        'all_results': results,
        'comparison': comparison
    }


if __name__ == '__main__':
    # Run with different strategies:
    # - 'balanced': Maximize F1-Score (default)
    # - 'high_recall': Catch more frauds
    # - 'high_precision': Fewer false alarms
    # - 'business': Business requirements

    results = main(threshold_strategy='balanced')




