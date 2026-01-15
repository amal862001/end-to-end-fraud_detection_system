"""
Unit tests for preprocessing module

Tests the preprocessing functionality in src/data/preprocess.py

Author: Your Name
Date: 2026-01-15
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data.preprocess import (
    FraudDataPreprocessor,
    prepare_train_test_split,
    preprocess_full_pipeline
)


@pytest.mark.unit
class TestFraudDataPreprocessor:
    """Tests for FraudDataPreprocessor class."""
    
    def test_preprocessor_initialization_standard(self):
        """Test preprocessor initialization with standard scaler."""
        preprocessor = FraudDataPreprocessor(scaler_type='standard')
        assert preprocessor.scaler_type == 'standard'
        assert preprocessor.scaler is not None
    
    def test_preprocessor_initialization_robust(self):
        """Test preprocessor initialization with robust scaler."""
        preprocessor = FraudDataPreprocessor(scaler_type='robust')
        assert preprocessor.scaler_type == 'robust'
        assert preprocessor.scaler is not None
    
    def test_preprocessor_invalid_scaler_raises_error(self):
        """Test that invalid scaler type raises ValueError."""
        with pytest.raises(ValueError):
            FraudDataPreprocessor(scaler_type='invalid')
    
    def test_detect_outliers_iqr(self, sample_dataframe):
        """Test IQR outlier detection."""
        preprocessor = FraudDataPreprocessor(outlier_method='iqr')
        outliers = preprocessor.detect_outliers_iqr(sample_dataframe['Amount'])
        
        assert isinstance(outliers, np.ndarray)
        assert len(outliers) == len(sample_dataframe)
        assert outliers.dtype == bool
    
    def test_detect_outliers_zscore(self, sample_dataframe):
        """Test Z-score outlier detection."""
        preprocessor = FraudDataPreprocessor(outlier_method='zscore')
        outliers = preprocessor.detect_outliers_zscore(sample_dataframe['Amount'])
        
        assert isinstance(outliers, np.ndarray)
        assert len(outliers) == len(sample_dataframe)
        assert outliers.dtype == bool
    
    def test_scale_features_fit_transform(self, sample_features_target):
        """Test scaling features with fit=True."""
        X, y = sample_features_target
        preprocessor = FraudDataPreprocessor(scaler_type='standard')
        
        X_scaled = preprocessor.scale_features(X, features=['Time', 'Amount'], fit=True)
        
        assert isinstance(X_scaled, pd.DataFrame)
        assert X_scaled.shape == X.shape
        assert preprocessor.scaler is not None
    
    def test_scale_features_transform_only(self, sample_features_target):
        """Test scaling features with fit=False."""
        X, y = sample_features_target
        preprocessor = FraudDataPreprocessor(scaler_type='standard')
        
        # First fit
        X_scaled_fit = preprocessor.scale_features(X, features=['Time', 'Amount'], fit=True)
        
        # Then transform only
        X_scaled_transform = preprocessor.scale_features(X, features=['Time', 'Amount'], fit=False)
        
        # Should be the same
        pd.testing.assert_frame_equal(X_scaled_fit, X_scaled_transform)
    
    def test_scale_features_raises_if_not_fitted(self, sample_features_target):
        """Test that transform without fit raises error."""
        X, y = sample_features_target
        preprocessor = FraudDataPreprocessor(scaler_type='standard')
        
        with pytest.raises(ValueError):
            preprocessor.scale_features(X, features=['Time', 'Amount'], fit=False)
    
    def test_fit_transform(self, sample_features_target):
        """Test fit_transform method."""
        X, y = sample_features_target
        preprocessor = FraudDataPreprocessor(scaler_type='standard', outlier_method='iqr')
        
        X_processed = preprocessor.fit_transform(X, handle_outliers=True, outlier_action='cap')
        
        assert isinstance(X_processed, pd.DataFrame)
        assert X_processed.shape == X.shape
        assert preprocessor.feature_names is not None
    
    def test_transform(self, sample_features_target):
        """Test transform method."""
        X, y = sample_features_target
        preprocessor = FraudDataPreprocessor(scaler_type='standard')
        
        # First fit
        X_train_processed = preprocessor.fit_transform(X, handle_outliers=False)
        
        # Then transform
        X_test_processed = preprocessor.transform(X)
        
        assert isinstance(X_test_processed, pd.DataFrame)
        assert X_test_processed.shape == X.shape


@pytest.mark.unit
class TestPrepareTrainTestSplit:
    """Tests for prepare_train_test_split function."""
    
    def test_train_test_split_returns_four_values(self, sample_features_target):
        """Test that function returns 4 values."""
        X, y = sample_features_target
        result = prepare_train_test_split(X, y, test_size=0.2, random_state=42)
        
        assert len(result) == 4
    
    def test_train_test_split_correct_sizes(self, sample_features_target):
        """Test that train/test split has correct sizes."""
        X, y = sample_features_target
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        total_size = len(X)
        train_size = len(X_train)
        test_size = len(X_test)
        
        assert train_size + test_size == total_size
        assert abs(test_size / total_size - 0.2) < 0.05  # Allow 5% tolerance
    
    def test_train_test_split_stratified(self, sample_features_target):
        """Test that stratified split maintains class balance."""
        X, y = sample_features_target
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=True
        )
        
        # Calculate fraud rates
        train_fraud_rate = y_train.sum() / len(y_train)
        test_fraud_rate = y_test.sum() / len(y_test)
        overall_fraud_rate = y.sum() / len(y)
        
        # Rates should be similar (within 10% relative difference)
        assert abs(train_fraud_rate - overall_fraud_rate) / overall_fraud_rate < 0.1
        assert abs(test_fraud_rate - overall_fraud_rate) / overall_fraud_rate < 0.1


@pytest.mark.unit
@pytest.mark.slow
class TestPreprocessFullPipeline:
    """Tests for preprocess_full_pipeline function."""
    
    def test_full_pipeline_returns_dict(self, sample_features_target):
        """Test that full pipeline returns a dictionary."""
        X, y = sample_features_target
        results = preprocess_full_pipeline(
            X, y,
            test_size=0.2,
            scaler_type='standard',
            outlier_method='iqr',
            outlier_action='cap',
            save_preprocessor=False,
            random_state=42
        )
        
        assert isinstance(results, dict)
    
    def test_full_pipeline_has_required_keys(self, sample_features_target):
        """Test that results dict has all required keys."""
        X, y = sample_features_target
        results = preprocess_full_pipeline(
            X, y,
            save_preprocessor=False,
            random_state=42
        )
        
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'preprocessor', 'outlier_stats']
        
        for key in required_keys:
            assert key in results
    
    def test_full_pipeline_correct_shapes(self, sample_features_target):
        """Test that processed data has correct shapes."""
        X, y = sample_features_target
        results = preprocess_full_pipeline(
            X, y,
            test_size=0.2,
            save_preprocessor=False,
            random_state=42
        )
        
        assert results['X_train'].shape[1] == X.shape[1]
        assert results['X_test'].shape[1] == X.shape[1]
        assert len(results['y_train']) == len(results['X_train'])
        assert len(results['y_test']) == len(results['X_test'])

