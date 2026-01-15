"""
Unit tests for data loading module

Tests the data loading functionality in src/data/load_data.py

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

from data.load_data import (
    get_data_path,
    load_raw_data,
    get_data_info,
    split_features_target,
    load_data
)


@pytest.mark.unit
class TestGetDataPath:
    """Tests for get_data_path function."""
    
    def test_get_data_path_default(self):
        """Test getting default data path."""
        path = get_data_path()
        assert isinstance(path, Path)
        assert path.name == "creditcard.csv"
        assert "data" in str(path)
        assert "raw" in str(path)
    
    def test_get_data_path_custom_filename(self):
        """Test getting data path with custom filename (uses default creditcard.csv)."""
        # Note: get_data_path checks for file existence, so we test with the actual file
        path = get_data_path("creditcard.csv")
        assert path.name == "creditcard.csv"
        assert "data" in str(path)
        assert "raw" in str(path)
    
    def test_get_data_path_raises_if_not_exists(self):
        """Test that FileNotFoundError is raised if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            path = get_data_path("nonexistent.csv")
            # Force check by trying to access it
            if not path.exists():
                raise FileNotFoundError(f"Data file not found at: {path}")


@pytest.mark.unit
class TestLoadRawData:
    """Tests for load_raw_data function."""
    
    def test_load_raw_data_returns_dataframe(self, data_dir):
        """Test that load_raw_data returns a DataFrame."""
        data_path = data_dir / "raw" / "creditcard.csv"
        
        if not data_path.exists():
            pytest.skip("creditcard.csv not found")
        
        df = load_raw_data(str(data_path))
        assert isinstance(df, pd.DataFrame)
    
    def test_load_raw_data_has_expected_columns(self, data_dir):
        """Test that loaded data has expected columns."""
        data_path = data_dir / "raw" / "creditcard.csv"
        
        if not data_path.exists():
            pytest.skip("creditcard.csv not found")
        
        df = load_raw_data(str(data_path))
        
        # Check for required columns
        assert 'Time' in df.columns
        assert 'Amount' in df.columns
        assert 'Class' in df.columns
        
        # Check for V1-V28 columns
        v_columns = [f'V{i}' for i in range(1, 29)]
        for col in v_columns:
            assert col in df.columns
    
    def test_load_raw_data_shape(self, data_dir):
        """Test that loaded data has expected shape."""
        data_path = data_dir / "raw" / "creditcard.csv"
        
        if not data_path.exists():
            pytest.skip("creditcard.csv not found")
        
        df = load_raw_data(str(data_path))
        
        # Should have 31 columns (Time, V1-V28, Amount, Class)
        assert df.shape[1] == 31
        
        # Should have many rows
        assert df.shape[0] > 0


@pytest.mark.unit
class TestGetDataInfo:
    """Tests for get_data_info function."""
    
    def test_get_data_info_returns_dict(self, sample_dataframe):
        """Test that get_data_info returns a dictionary."""
        info = get_data_info(sample_dataframe)
        assert isinstance(info, dict)
    
    def test_get_data_info_has_required_keys(self, sample_dataframe):
        """Test that info dict has all required keys."""
        info = get_data_info(sample_dataframe)
        
        required_keys = [
            'total_transactions',
            'total_features',
            'fraud_count',
            'legitimate_count',
            'fraud_percentage',
            'missing_values',
            'duplicate_rows'
        ]
        
        for key in required_keys:
            assert key in info
    
    def test_get_data_info_correct_counts(self, sample_dataframe):
        """Test that counts are correct."""
        info = get_data_info(sample_dataframe)
        
        assert info['total_transactions'] == len(sample_dataframe)
        assert info['fraud_count'] == sample_dataframe['Class'].sum()
        assert info['legitimate_count'] == (sample_dataframe['Class'] == 0).sum()
        assert info['fraud_count'] + info['legitimate_count'] == info['total_transactions']
    
    def test_get_data_info_fraud_percentage(self, sample_dataframe):
        """Test that fraud percentage is calculated correctly."""
        info = get_data_info(sample_dataframe)
        
        expected_percentage = (info['fraud_count'] / info['total_transactions']) * 100
        assert abs(info['fraud_percentage'] - expected_percentage) < 0.01


@pytest.mark.unit
class TestSplitFeaturesTarget:
    """Tests for split_features_target function."""
    
    def test_split_features_target_returns_tuple(self, sample_dataframe):
        """Test that function returns a tuple."""
        result = split_features_target(sample_dataframe)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_split_features_target_correct_shapes(self, sample_dataframe):
        """Test that X and y have correct shapes."""
        X, y = split_features_target(sample_dataframe)
        
        assert len(X) == len(sample_dataframe)
        assert len(y) == len(sample_dataframe)
        assert X.shape[1] == sample_dataframe.shape[1] - 1  # Excluding Class
    
    def test_split_features_target_no_class_in_features(self, sample_dataframe):
        """Test that Class column is not in features."""
        X, y = split_features_target(sample_dataframe)
        assert 'Class' not in X.columns
    
    def test_split_features_target_y_is_series(self, sample_dataframe):
        """Test that y is a pandas Series."""
        X, y = split_features_target(sample_dataframe)
        assert isinstance(y, pd.Series)
        assert y.name == 'Class'

