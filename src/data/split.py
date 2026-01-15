"""
Data splitting module for fraud detection system.

This module provides functions to split data into train/validation/test sets
with proper stratification to prevent data leakage and ensure fair evaluation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
from sklearn.model_selection import train_test_split


def get_processed_data_path() -> Path:
    """
    Get the path to the processed data directory.
    
    Returns:
        Path object pointing to the processed data directory
    """
    project_root = Path(__file__).parent.parent.parent
    processed_path = project_root / "data" / "processed"
    
    # Create directory if it doesn't exist
    processed_path.mkdir(parents=True, exist_ok=True)
    
    return processed_path


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets with stratification.
    
    This ensures that the class distribution (fraud rate) is maintained
    across all three splits, which is critical for imbalanced datasets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test set (default: 0.2 = 20%)
        val_size: Proportion of validation set from remaining data (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*70)
    print("STRATIFIED TRAIN/VALIDATION/TEST SPLIT")
    print("="*70)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Stratify by target to maintain fraud rate
    )
    
    # Second split: separate validation from training
    # Adjust val_size to be relative to the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp  # Stratify again
    )
    
    # Calculate actual proportions
    total_samples = len(X)
    train_pct = len(X_train) / total_samples * 100
    val_pct = len(X_val) / total_samples * 100
    test_pct = len(X_test) / total_samples * 100
    
    # Print split information
    print(f"\nTotal samples: {total_samples:,}")
    print(f"\nSplit proportions:")
    print(f"  Train:      {len(X_train):,} ({train_pct:.1f}%)")
    print(f"  Validation: {len(X_val):,} ({val_pct:.1f}%)")
    print(f"  Test:       {len(X_test):,} ({test_pct:.1f}%)")
    
    # Calculate and display fraud rates
    train_fraud_rate = y_train.sum() / len(y_train) * 100
    val_fraud_rate = y_val.sum() / len(y_val) * 100
    test_fraud_rate = y_test.sum() / len(y_test) * 100
    
    print(f"\nFraud rates (stratification check):")
    print(f"  Train:      {y_train.sum():,} frauds ({train_fraud_rate:.4f}%)")
    print(f"  Validation: {y_val.sum():,} frauds ({val_fraud_rate:.4f}%)")
    print(f"  Test:       {y_test.sum():,} frauds ({test_fraud_rate:.4f}%)")
    
    # Verify stratification worked
    fraud_rate_diff = max(train_fraud_rate, val_fraud_rate, test_fraud_rate) - \
                      min(train_fraud_rate, val_fraud_rate, test_fraud_rate)
    
    if fraud_rate_diff < 0.01:  # Less than 0.01% difference
        print(f"\n✓ Stratification successful! (max difference: {fraud_rate_diff:.4f}%)")
    else:
        print(f"\n⚠️  Warning: Fraud rate variance detected ({fraud_rate_diff:.4f}%)")
    
    print("\n" + "="*70)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    prefix: str = ""
) -> Dict[str, Path]:
    """
    Save split datasets to disk.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        prefix: Optional prefix for filenames (e.g., 'preprocessed_')
    
    Returns:
        Dictionary mapping dataset names to file paths
    """
    print("\n" + "="*70)
    print("SAVING SPLIT DATASETS")
    print("="*70)

    processed_path = get_processed_data_path()
    saved_files = {}

    # Define file paths
    datasets = {
        'X_train': (X_train, f'{prefix}X_train.csv'),
        'X_val': (X_val, f'{prefix}X_val.csv'),
        'X_test': (X_test, f'{prefix}X_test.csv'),
        'y_train': (y_train, f'{prefix}y_train.csv'),
        'y_val': (y_val, f'{prefix}y_val.csv'),
        'y_test': (y_test, f'{prefix}y_test.csv')
    }

    # Save each dataset
    for name, (data, filename) in datasets.items():
        filepath = processed_path / filename

        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:  # Series
            data.to_csv(filepath, index=False, header=['Class'])

        saved_files[name] = filepath
        print(f"  ✓ Saved {name}: {filepath}")

    # Save metadata
    metadata = {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_fraud_rate': float(y_train.sum() / len(y_train)),
        'val_fraud_rate': float(y_val.sum() / len(y_val)),
        'test_fraud_rate': float(y_test.sum() / len(y_test)),
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns)
    }

    metadata_path = processed_path / f'{prefix}split_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    saved_files['metadata'] = metadata_path
    print(f"  ✓ Saved metadata: {metadata_path}")

    print("\n✓ All datasets saved successfully!")
    print("="*70)

    return saved_files


def load_split_data(prefix: str = "") -> Dict[str, Any]:
    """
    Load previously saved split datasets.

    Args:
        prefix: Optional prefix for filenames (e.g., 'preprocessed_')

    Returns:
        Dictionary containing loaded datasets and metadata
    """
    print("\n" + "="*70)
    print("LOADING SPLIT DATASETS")
    print("="*70)

    processed_path = get_processed_data_path()

    # Load datasets
    X_train = pd.read_csv(processed_path / f'{prefix}X_train.csv')
    X_val = pd.read_csv(processed_path / f'{prefix}X_val.csv')
    X_test = pd.read_csv(processed_path / f'{prefix}X_test.csv')

    y_train = pd.read_csv(processed_path / f'{prefix}y_train.csv')['Class']
    y_val = pd.read_csv(processed_path / f'{prefix}y_val.csv')['Class']
    y_test = pd.read_csv(processed_path / f'{prefix}y_test.csv')['Class']

    # Load metadata
    metadata_path = processed_path / f'{prefix}split_metadata.pkl'
    metadata = joblib.load(metadata_path)

    print(f"\n✓ Loaded datasets:")
    print(f"  Train:      {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test:       {len(X_test):,} samples")

    print(f"\n✓ Fraud rates:")
    print(f"  Train:      {metadata['train_fraud_rate']*100:.4f}%")
    print(f"  Validation: {metadata['val_fraud_rate']*100:.4f}%")
    print(f"  Test:       {metadata['test_fraud_rate']*100:.4f}%")

    print("\n" + "="*70)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'metadata': metadata
    }


def create_and_save_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Complete pipeline: split data and optionally save to disk.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test set
        val_size: Proportion of validation set
        random_state: Random seed for reproducibility
        save: Whether to save splits to disk
        prefix: Optional prefix for saved files

    Returns:
        Dictionary containing all splits and metadata
    """
    # Perform stratified split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # Save if requested
    saved_files = None
    if save:
        saved_files = save_split_data(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            prefix=prefix
        )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'saved_files': saved_files
    }


if __name__ == "__main__":
    """
    Example usage and testing of the split module.
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_data

    print("="*70)
    print("TESTING DATA SPLITTING MODULE")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    X, y = load_data(verbose=True)

    # Test 1: Create and save splits
    print("\n2. Creating stratified train/val/test splits...")
    results = create_and_save_splits(
        X, y,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        save=True,
        prefix=""
    )

    print("\n3. Verifying saved files...")
    if results['saved_files']:
        print("\nSaved files:")
        for name, path in results['saved_files'].items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name}: {path.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ✗ {name}: NOT FOUND")

    # Test 2: Load splits back
    print("\n4. Testing load functionality...")
    loaded_data = load_split_data(prefix="")

    # Verify loaded data matches original
    print("\n5. Verifying data integrity...")
    checks = [
        ('X_train shape', results['X_train'].shape == loaded_data['X_train'].shape),
        ('X_val shape', results['X_val'].shape == loaded_data['X_val'].shape),
        ('X_test shape', results['X_test'].shape == loaded_data['X_test'].shape),
        ('y_train sum', results['y_train'].sum() == loaded_data['y_train'].sum()),
        ('y_val sum', results['y_val'].sum() == loaded_data['y_val'].sum()),
        ('y_test sum', results['y_test'].sum() == loaded_data['y_test'].sum()),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    # Display summary statistics
    print("\n6. Summary Statistics:")
    print("\nDataset Sizes:")
    print(f"  Training:   {len(results['X_train']):,} samples")
    print(f"  Validation: {len(results['X_val']):,} samples")
    print(f"  Test:       {len(results['X_test']):,} samples")
    print(f"  Total:      {len(results['X_train']) + len(results['X_val']) + len(results['X_test']):,} samples")

    print("\nFraud Distribution:")
    print(f"  Training:   {results['y_train'].sum():,} frauds / {len(results['y_train']):,} total")
    print(f"  Validation: {results['y_val'].sum():,} frauds / {len(results['y_val']):,} total")
    print(f"  Test:       {results['y_test'].sum():,} frauds / {len(results['y_test']):,} total")

    print("\nClass Balance (Fraud %):")
    print(f"  Training:   {results['y_train'].sum()/len(results['y_train'])*100:.4f}%")
    print(f"  Validation: {results['y_val'].sum()/len(results['y_val'])*100:.4f}%")
    print(f"  Test:       {results['y_test'].sum()/len(results['y_test'])*100:.4f}%")

    # Final status
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("="*70)

