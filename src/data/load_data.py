"""
Data loading module for fraud detection system.

This module provides functions to load and prepare credit card transaction data
for fraud detection modeling.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def get_data_path(filename: str = "creditcard.csv") -> Path:
    """
    Get the path to the data file.
    
    Args:
        filename: Name of the data file (default: "creditcard.csv")
    
    Returns:
        Path object pointing to the data file
    """
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    return data_path


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw credit card transaction data.
    
    Args:
        filepath: Optional path to the data file. If None, uses default path.
    
    Returns:
        DataFrame containing the raw transaction data
    """
    if filepath is None:
        filepath = get_data_path()
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get summary information about the dataset.
    
    Args:
        df: DataFrame containing transaction data
    
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'total_transactions': len(df),
        'total_features': df.shape[1] - 1,  # Excluding target
        'fraud_count': df['Class'].sum(),
        'legitimate_count': (df['Class'] == 0).sum(),
        'fraud_percentage': (df['Class'].sum() / len(df)) * 100,
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return info


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset.
    
    Args:
        df: DataFrame containing transaction data
    """
    info = get_data_info(df)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total Transactions: {info['total_transactions']:,}")
    print(f"Total Features: {info['total_features']}")
    print(f"Fraudulent Transactions: {info['fraud_count']:,} ({info['fraud_percentage']:.2f}%)")
    print(f"Legitimate Transactions: {info['legitimate_count']:,}")
    print(f"Missing Values: {info['missing_values']}")
    print(f"Duplicate Rows: {info['duplicate_rows']}")
    print("="*50 + "\n")


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features and target variable.
    
    Args:
        df: DataFrame containing transaction data
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y


def load_data(filepath: Optional[str] = None, 
              verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data and split into features and target.
    
    Args:
        filepath: Optional path to the data file
        verbose: Whether to print data summary
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df = load_raw_data(filepath)
    
    if verbose:
        print_data_summary(df)
    
    X, y = split_features_target(df)
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Testing data loading functionality...\n")
    
    # Load raw data
    df = load_raw_data()
    
    # Print summary
    print_data_summary(df)
    
    # Split features and target
    X, y = split_features_target(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFirst few feature columns: {list(X.columns[:5])}")
    print(f"Target distribution:\n{y.value_counts()}")

