"""
Handle Class Imbalance for Fraud Detection
Improve fraud recall through various techniques.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    recall_score,
    precision_score
)
from imblearn.over_sampling import SMOTE
import joblib

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import load_data


def prepare_data(test_size=0.2, random_state=42):
    """Load and prepare data."""
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
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud rate - Train: {y_train.mean()*100:.2f}%, Test: {y_test.mean()*100:.2f}%")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }
    
    return metrics


def print_results(name, metrics):
    """Print formatted results."""
    print(f"\n{name}:")
    print(f"  Recall:    {metrics['Recall']:.4f} ⭐")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"  PR-AUC:    {metrics['PR-AUC']:.4f}")
    print(f"  Confusion Matrix: TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")


def test_class_weights(X_train, y_train, X_test, y_test):
    """Test different class weight strategies."""
    print("\n" + "="*60)
    print("1. TESTING CLASS WEIGHTS")
    print("="*60)
    
    results = {}
    
    # Test different weight strategies
    weight_strategies = {
        'None': None,
        'Balanced': 'balanced',
        'Custom 1:10': {0: 1, 1: 10},
        'Custom 1:50': {0: 1, 1: 50},
        'Custom 1:100': {0: 1, 1: 100}
    }
    
    for name, weights in weight_strategies.items():
        print(f"\nTraining with {name}...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[name] = metrics
        print_results(name, metrics)

    return results


def test_smote(X_train, y_train, X_test, y_test):
    """Test SMOTE oversampling (carefully)."""
    print("\n" + "="*60)
    print("2. TESTING SMOTE OVERSAMPLING")
    print("="*60)

    results = {}

    # Test different SMOTE sampling strategies
    sampling_strategies = {
        'No SMOTE': None,
        'SMOTE 0.1': 0.1,  # Minority = 10% of majority
        'SMOTE 0.3': 0.3,  # Minority = 30% of majority
        'SMOTE 0.5': 0.5,  # Minority = 50% of majority
        'SMOTE 1.0': 1.0   # Full balance
    }

    for name, strategy in sampling_strategies.items():
        print(f"\nTraining with {name}...")

        if strategy is None:
            # No SMOTE - use original data
            X_train_resampled = X_train
            y_train_resampled = y_train
            print(f"  Original distribution: {y_train.value_counts().to_dict()}")
        else:
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=strategy, random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"  After SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[name] = metrics
        print_results(name, metrics)

    return results


def tune_threshold(X_train, y_train, X_test, y_test, target_recall=0.90):
    """Tune decision threshold for business requirements."""
    print("\n" + "="*60)
    print("3. TUNING DECISION THRESHOLD")
    print("="*60)
    print(f"Target: Recall >= {target_recall:.0%}")

    # Train model with balanced weights
    print("\nTraining model with balanced weights...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    print("\nTesting thresholds:")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1-Score':<12} {'FP':<8} {'FN':<8}")
    print("-" * 80)

    best_threshold = 0.5
    best_recall = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[threshold] = metrics

        print(f"{threshold:<12.2f} {metrics['Recall']:<10.4f} {metrics['Precision']:<12.4f} "
              f"{metrics['F1-Score']:<12.4f} {metrics['FP']:<8d} {metrics['FN']:<8d}")

        # Find best threshold that meets recall target
        if metrics['Recall'] >= target_recall and metrics['Recall'] > best_recall:
            best_recall = metrics['Recall']
            best_threshold = threshold

    print("-" * 80)
    print(f"\n✓ Best threshold: {best_threshold:.2f} (Recall: {best_recall:.4f})")

    return results, best_threshold, model


def visualize_results(class_weight_results, smote_results, threshold_results):
    """Visualize comparison of techniques."""
    print("\n" + "="*60)
    print("4. VISUALIZING RESULTS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Class Weights Comparison
    ax = axes[0, 0]
    names = list(class_weight_results.keys())
    recalls = [class_weight_results[n]['Recall'] for n in names]
    precisions = [class_weight_results[n]['Precision'] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, recalls, width, label='Recall', color='green', alpha=0.7)
    ax.bar(x + width/2, precisions, width, label='Precision', color='blue', alpha=0.7)
    ax.set_xlabel('Class Weight Strategy')
    ax.set_ylabel('Score')
    ax.set_title('Class Weights: Recall vs Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. SMOTE Comparison
    ax = axes[0, 1]
    names = list(smote_results.keys())
    recalls = [smote_results[n]['Recall'] for n in names]
    precisions = [smote_results[n]['Precision'] for n in names]

    x = np.arange(len(names))
    ax.bar(x - width/2, recalls, width, label='Recall', color='green', alpha=0.7)
    ax.bar(x + width/2, precisions, width, label='Precision', color='blue', alpha=0.7)
    ax.set_xlabel('SMOTE Strategy')
    ax.set_ylabel('Score')
    ax.set_title('SMOTE: Recall vs Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Threshold Tuning
    ax = axes[1, 0]
    thresholds = list(threshold_results.keys())
    recalls = [threshold_results[t]['Recall'] for t in thresholds]
    precisions = [threshold_results[t]['Precision'] for t in thresholds]

    ax.plot(thresholds, recalls, 'o-', label='Recall', color='green', linewidth=2)
    ax.plot(thresholds, precisions, 's-', label='Precision', color='blue', linewidth=2)
    ax.axhline(y=0.9, color='red', linestyle='--', label='Target Recall (90%)')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Tuning: Recall vs Precision')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. False Positives vs False Negatives
    ax = axes[1, 1]
    fps = [threshold_results[t]['FP'] for t in thresholds]
    fns = [threshold_results[t]['FN'] for t in thresholds]

    ax.plot(thresholds, fps, 'o-', label='False Positives', color='orange', linewidth=2)
    ax.plot(thresholds, fns, 's-', label='False Negatives', color='red', linewidth=2)
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Count')
    ax.set_title('Threshold Impact on Errors')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/imbalance_handling.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to reports/figures/imbalance_handling.png")

    plt.show()


def create_summary(class_weight_results, smote_results, threshold_results, best_threshold):
    """Create summary of findings."""
    print("\n" + "="*60)
    print("5. SUMMARY & RECOMMENDATIONS")
    print("="*60)

    # Find best from each technique
    best_cw = max(class_weight_results.items(), key=lambda x: x[1]['Recall'])
    best_smote = max(smote_results.items(), key=lambda x: x[1]['Recall'])
    best_thresh = threshold_results[best_threshold]

    print("\n📊 BEST RESULTS BY TECHNIQUE:")
    print("-" * 60)

    print(f"\n1. CLASS WEIGHTS - Best: {best_cw[0]}")
    print(f"   Recall:    {best_cw[1]['Recall']:.4f}")
    print(f"   Precision: {best_cw[1]['Precision']:.4f}")
    print(f"   F1-Score:  {best_cw[1]['F1-Score']:.4f}")
    print(f"   Errors:    FP={best_cw[1]['FP']}, FN={best_cw[1]['FN']}")

    print(f"\n2. SMOTE - Best: {best_smote[0]}")
    print(f"   Recall:    {best_smote[1]['Recall']:.4f}")
    print(f"   Precision: {best_smote[1]['Precision']:.4f}")
    print(f"   F1-Score:  {best_smote[1]['F1-Score']:.4f}")
    print(f"   Errors:    FP={best_smote[1]['FP']}, FN={best_smote[1]['FN']}")

    print(f"\n3. THRESHOLD TUNING - Best: {best_threshold:.2f}")
    print(f"   Recall:    {best_thresh['Recall']:.4f}")
    print(f"   Precision: {best_thresh['Precision']:.4f}")
    print(f"   F1-Score:  {best_thresh['F1-Score']:.4f}")
    print(f"   Errors:    FP={best_thresh['FP']}, FN={best_thresh['FN']}")

    # Recommendations
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)

    print("\n1. FOR MAXIMUM RECALL (Catch all frauds):")
    print(f"   → Use threshold tuning with threshold = {best_threshold:.2f}")
    print(f"   → Achieves {best_thresh['Recall']:.1%} recall")
    print(f"   → Trade-off: {best_thresh['FP']} false positives")

    print("\n2. FOR BALANCED PERFORMANCE:")
    print(f"   → Use class weights: {best_cw[0]}")
    print(f"   → Good balance of recall and precision")
    print(f"   → Fewer false alarms")

    print("\n3. SMOTE CAUTION:")
    print("   ⚠ SMOTE can cause overfitting on synthetic data")
    print("   ⚠ Test carefully on real data")
    print("   → Class weights often work better")

    print("\n4. BUSINESS DECISION:")
    print("   → If missing fraud is costly: Use low threshold")
    print("   → If false alarms are costly: Use balanced weights")
    print("   → Recommended: Threshold tuning for flexibility")

    # Save summary
    summary = {
        'best_class_weight': best_cw[0],
        'best_smote': best_smote[0],
        'best_threshold': best_threshold,
        'class_weight_recall': best_cw[1]['Recall'],
        'smote_recall': best_smote[1]['Recall'],
        'threshold_recall': best_thresh['Recall'],
        'class_weight_precision': best_cw[1]['Precision'],
        'smote_precision': best_smote[1]['Precision'],
        'threshold_precision': best_thresh['Precision']
    }

    return summary


def save_best_model(model, scaler, threshold, output_dir='models/imbalance'):
    """Save the best model with tuned threshold."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("SAVING BEST MODEL")
    print("="*60)

    # Save model
    model_path = os.path.join(output_dir, 'random_forest_tuned.pkl')
    joblib.dump(model, model_path)
    print(f"✓ Saved model to {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to {scaler_path}")

    # Save threshold
    threshold_path = os.path.join(output_dir, 'best_threshold.txt')
    with open(threshold_path, 'w') as f:
        f.write(f"{threshold:.4f}")
    print(f"✓ Saved threshold ({threshold:.4f}) to {threshold_path}")


def main():
    """Main pipeline for handling class imbalance."""
    print("="*60)
    print("HANDLING CLASS IMBALANCE FOR FRAUD DETECTION")
    print("="*60)
    print("\nObjective: Improve fraud recall (catch more frauds)")
    print("Key Metric: Recall (Fraud)")

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # Test different techniques
    class_weight_results = test_class_weights(X_train, y_train, X_test, y_test)
    smote_results = test_smote(X_train, y_train, X_test, y_test)
    threshold_results, best_threshold, model = tune_threshold(X_train, y_train, X_test, y_test, target_recall=0.90)

    # Visualize
    visualize_results(class_weight_results, smote_results, threshold_results)

    # Summary
    summary = create_summary(class_weight_results, smote_results, threshold_results, best_threshold)

    # Save best model
    save_best_model(model, scaler, best_threshold)

    print("\n" + "="*60)
    print("✓ IMBALANCE HANDLING COMPLETE!")
    print("="*60)

    return class_weight_results, smote_results, threshold_results, best_threshold, model, summary


if __name__ == '__main__':
    results = main()




