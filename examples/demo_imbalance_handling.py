"""
Quick Demo: Class Imbalance Handling
Shows the impact of different techniques on fraud recall.
"""

import sys
import os
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from data.load_data import load_data


def demo_class_weights():
    """Demo: Impact of class weights on recall."""
    print("="*60)
    print("DEMO: CLASS WEIGHTS IMPACT ON FRAUD RECALL")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Test set: {len(y_test)} samples, {y_test.sum()} frauds")
    
    # Test different class weights
    strategies = {
        'No Weights': None,
        'Balanced': 'balanced',
        'Heavy (1:50)': {0: 1, 1: 50}
    }
    
    print("\n" + "-"*60)
    print(f"{'Strategy':<15} {'Recall':<10} {'Precision':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-"*60)
    
    for name, weights in strategies.items():
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,  # Fewer trees for speed
            max_depth=10,
            class_weight=weights,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"{name:<15} {recall:<10.4f} {precision:<12.4f} {tp:<6d} {fp:<6d} {fn:<6d}")
    
    print("-"*60)
    print("\n💡 Key Insight:")
    print("   → Heavier class weights increase recall (catch more frauds)")
    print("   → But decrease precision (more false alarms)")
    print("   → Choose based on business cost of missing fraud vs false alarm")


def demo_threshold_tuning():
    """Demo: Impact of threshold on recall."""
    print("\n\n" + "="*60)
    print("DEMO: THRESHOLD TUNING FOR FRAUD RECALL")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model with balanced weights...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Test thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n" + "-"*70)
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-"*70)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        marker = " ⭐" if recall >= 0.90 else ""
        print(f"{threshold:<12.2f} {recall:<10.4f} {precision:<12.4f} {tp:<6d} {fp:<6d} {fn:<6d}{marker}")
    
    print("-"*70)
    print("\n💡 Key Insight:")
    print("   → Lower threshold = Higher recall (catch more frauds)")
    print("   → But more false positives (false alarms)")
    print("   → Threshold 0.2-0.3 often gives 90%+ recall with acceptable FP")
    print("   → ⭐ marks thresholds achieving 90%+ recall")


def main():
    """Run both demos."""
    print("\n🎯 QUICK DEMO: HANDLING CLASS IMBALANCE\n")
    print("This demo shows how to improve fraud recall using:")
    print("1. Class Weights")
    print("2. Threshold Tuning")
    print("\nRuntime: ~2-3 minutes\n")
    
    # Run demos
    demo_class_weights()
    demo_threshold_tuning()
    
    print("\n\n" + "="*60)
    print("✓ DEMO COMPLETE!")
    print("="*60)
    print("\n📚 For full analysis, run:")
    print("   python src/models/handle_imbalance.py")
    print("\n📖 For detailed guide, see:")
    print("   IMBALANCE_HANDLING_GUIDE.md")


if __name__ == '__main__':
    main()

