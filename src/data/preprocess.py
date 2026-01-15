"""
Data preprocessing module for fraud detection system.

This module provides functions to preprocess credit card transaction data
including feature scaling, outlier handling, and pipeline management.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy import stats


class FraudDataPreprocessor:
    """
    Preprocessor for fraud detection data.
    
    Handles:
    - Feature scaling (Time and Amount)
    - Outlier detection and handling
    - Train-test splitting with stratification
    - Pipeline persistence
    """
    
    def __init__(self, scaler_type: str = 'standard', outlier_method: str = 'iqr'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
            outlier_method: Method for outlier detection ('iqr', 'zscore', or 'none')
        """
        self.scaler_type = scaler_type
        self.outlier_method = outlier_method
        self.scaler = None
        self.outlier_stats = {}
        self.feature_names = None
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def detect_outliers_iqr(self, data: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """
        Detect outliers using IQR method.
        
        Args:
            data: Series to check for outliers
            multiplier: IQR multiplier (default: 1.5)
        
        Returns:
            Boolean array indicating outliers
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)

        # Store stats
        self.outlier_stats[data.name] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': outliers.sum()
        }

        return outliers.values
    
    def detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using Z-score method.
        
        Args:
            data: Series to check for outliers
            threshold: Z-score threshold (default: 3.0)
        
        Returns:
            Boolean array indicating outliers
        """
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        
        # Store stats
        self.outlier_stats[data.name] = {
            'mean': data.mean(),
            'std': data.std(),
            'threshold': threshold,
            'n_outliers': outliers.sum()
        }
        
        return outliers
    
    def handle_outliers(self, X: pd.DataFrame, features: list = None, 
                       action: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers in specified features.
        
        Args:
            X: Feature DataFrame
            features: List of features to check (default: ['Time', 'Amount'])
            action: Action to take ('cap', 'remove', or 'none')
        
        Returns:
            DataFrame with outliers handled
        """
        if features is None:
            features = ['Time', 'Amount']
        
        X_processed = X.copy()
        
        if self.outlier_method == 'none' or action == 'none':
            print("Skipping outlier handling")
            return X_processed
        
        for feature in features:
            if feature not in X_processed.columns:
                continue
            
            # Detect outliers
            if self.outlier_method == 'iqr':
                outliers = self.detect_outliers_iqr(X_processed[feature])
            elif self.outlier_method == 'zscore':
                outliers = self.detect_outliers_zscore(X_processed[feature])
            else:
                continue
            
            # Handle outliers
            if action == 'cap':
                if self.outlier_method == 'iqr':
                    stats_dict = self.outlier_stats[feature]
                    X_processed.loc[outliers, feature] = X_processed.loc[outliers, feature].clip(
                        lower=stats_dict['lower_bound'],
                        upper=stats_dict['upper_bound']
                    )

            print(f"  {feature}: {outliers.sum()} outliers detected and {action}ped")

        return X_processed

    def scale_features(self, X: pd.DataFrame, features: list = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale specified features.

        Args:
            X: Feature DataFrame
            features: List of features to scale (default: ['Time', 'Amount'])
            fit: Whether to fit the scaler (True for training, False for test)

        Returns:
            DataFrame with scaled features
        """
        if features is None:
            features = ['Time', 'Amount']

        X_scaled = X.copy()

        # Only scale features that exist in the DataFrame
        features_to_scale = [f for f in features if f in X_scaled.columns]

        if not features_to_scale:
            print("No features to scale")
            return X_scaled

        if fit:
            # Fit and transform
            X_scaled[features_to_scale] = self.scaler.fit_transform(X_scaled[features_to_scale])
            print(f"Fitted and scaled features: {features_to_scale}")
        else:
            # Transform only
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled[features_to_scale] = self.scaler.transform(X_scaled[features_to_scale])
            print(f"Scaled features: {features_to_scale}")

        return X_scaled

    def fit_transform(self, X: pd.DataFrame, handle_outliers: bool = True,
                     outlier_action: str = 'cap') -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.

        Args:
            X: Feature DataFrame
            handle_outliers: Whether to handle outliers
            outlier_action: Action for outliers ('cap', 'remove', or 'none')

        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "="*50)
        print("FITTING PREPROCESSOR")
        print("="*50)

        self.feature_names = X.columns.tolist()
        X_processed = X.copy()

        # Handle outliers
        if handle_outliers:
            print("\nHandling outliers...")
            X_processed = self.handle_outliers(X_processed, action=outlier_action)

        # Scale features
        print("\nScaling features...")
        X_processed = self.scale_features(X_processed, fit=True)

        print("\n✓ Preprocessing complete!")
        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Feature DataFrame

        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "="*50)
        print("TRANSFORMING DATA")
        print("="*50)

        X_processed = X.copy()

        # Scale features (no outlier handling for test data)
        print("\nScaling features...")
        X_processed = self.scale_features(X_processed, fit=False)

        print("\n✓ Transformation complete!")
        return X_processed

    def save(self, filepath: str) -> None:
        """
        Save the preprocessor to disk.

        Args:
            filepath: Path to save the preprocessor
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save the preprocessor
        joblib.dump(self, filepath)
        print(f"\n✓ Preprocessor saved to: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'FraudDataPreprocessor':
        """
        Load a preprocessor from disk.

        Args:
            filepath: Path to the saved preprocessor

        Returns:
            Loaded preprocessor
        """
        preprocessor = joblib.load(filepath)
        print(f"\n✓ Preprocessor loaded from: {filepath}")
        return preprocessor


def prepare_train_test_split(X: pd.DataFrame, y: pd.Series,
                             test_size: float = 0.2,
                             random_state: int = 42,
                             stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                              pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split by target

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)

    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    print(f"\nTrain set size: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    if stratify:
        print(f"\nTrain fraud rate: {y_train.sum()/len(y_train)*100:.4f}%")
        print(f"Test fraud rate:  {y_test.sum()/len(y_test)*100:.4f}%")

    print("\n✓ Data split complete!")

    return X_train, X_test, y_train, y_test


def get_preprocessing_pipeline_path(filename: str = "preprocessor.pkl") -> Path:
    """
    Get the path to save/load preprocessing pipeline.

    Args:
        filename: Name of the preprocessor file

    Returns:
        Path object pointing to the preprocessor file
    """
    project_root = Path(__file__).parent.parent.parent
    pipeline_path = project_root / "models" / filename
    return pipeline_path


def preprocess_full_pipeline(X: pd.DataFrame, y: pd.Series,
                             test_size: float = 0.2,
                             scaler_type: str = 'standard',
                             outlier_method: str = 'iqr',
                             outlier_action: str = 'cap',
                             save_preprocessor: bool = True,
                             random_state: int = 42) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Test set proportion
        scaler_type: Type of scaler ('standard' or 'robust')
        outlier_method: Outlier detection method ('iqr', 'zscore', or 'none')
        outlier_action: Action for outliers ('cap', 'remove', or 'none')
        save_preprocessor: Whether to save the preprocessor
        random_state: Random seed

    Returns:
        Dictionary containing preprocessed data and preprocessor
    """
    print("\n" + "="*70)
    print("FULL PREPROCESSING PIPELINE")
    print("="*70)

    # Step 1: Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 2: Initialize preprocessor
    preprocessor = FraudDataPreprocessor(
        scaler_type=scaler_type,
        outlier_method=outlier_method
    )

    # Step 3: Fit and transform training data
    X_train_processed = preprocessor.fit_transform(
        X_train,
        handle_outliers=True,
        outlier_action=outlier_action
    )

    # Step 4: Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Step 5: Save preprocessor
    if save_preprocessor:
        preprocessor_path = get_preprocessing_pipeline_path()
        preprocessor.save(str(preprocessor_path))

    # Return results
    results = {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'outlier_stats': preprocessor.outlier_stats
    }

    print("\n" + "="*70)
    print("✓ FULL PIPELINE COMPLETE!")
    print("="*70)

    return results


if __name__ == "__main__":
    """
    Example usage of the preprocessing module.
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_data

    print("="*70)
    print("TESTING PREPROCESSING MODULE")
    print("="*70)

    # Load data
    X, y = load_data(verbose=True)

    # Run full preprocessing pipeline
    results = preprocess_full_pipeline(
        X, y,
        test_size=0.2,
        scaler_type='standard',
        outlier_method='iqr',
        outlier_action='cap',
        save_preprocessor=True,
        random_state=42
    )

    # Display results
    print("\n" + "="*70)
    print("PREPROCESSING RESULTS")
    print("="*70)

    print(f"\nTraining set shape: {results['X_train'].shape}")
    print(f"Test set shape: {results['X_test'].shape}")

    print(f"\nTraining set fraud rate: {results['y_train'].sum()/len(results['y_train'])*100:.4f}%")
    print(f"Test set fraud rate: {results['y_test'].sum()/len(results['y_test'])*100:.4f}%")

    print("\nOutlier Statistics:")
    for feature, stats in results['outlier_stats'].items():
        print(f"\n{feature}:")
        for key, value in stats.items():
            if isinstance(value, (int, np.integer)):
                print(f"  {key}: {value:,}")
            elif isinstance(value, (float, np.floating)):
                print(f"  {key}: {value:.2f}")

    print("\nSample of preprocessed training data:")
    print(results['X_train'].head())

    print("\n" + "="*70)
    print("✓ TESTING COMPLETE!")
    print("="*70)

