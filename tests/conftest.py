"""
Pytest configuration and shared fixtures

This file contains pytest configuration and fixtures that are shared
across all test modules.

Author: Your Name
Date: 2026-01-15
"""

import pytest
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from tests.fixtures.sample_data import (
    get_sample_transaction,
    get_sample_dataframe,
    get_batch_transactions,
    LEGITIMATE_TRANSACTION,
    FRAUD_TRANSACTION
)


@pytest.fixture
def sample_legitimate_transaction():
    """Fixture for a sample legitimate transaction."""
    return get_sample_transaction(fraud=False)


@pytest.fixture
def sample_fraud_transaction():
    """Fixture for a sample fraudulent transaction."""
    return get_sample_transaction(fraud=True)


@pytest.fixture
def sample_dataframe():
    """Fixture for a sample DataFrame with 100 transactions."""
    return get_sample_dataframe(n_samples=100, fraud_ratio=0.1)


@pytest.fixture
def sample_small_dataframe():
    """Fixture for a small DataFrame with 20 transactions."""
    return get_sample_dataframe(n_samples=20, fraud_ratio=0.2)


@pytest.fixture
def sample_batch_transactions():
    """Fixture for a batch of 5 transactions."""
    return get_batch_transactions(n=5, fraud_ratio=0.2)


@pytest.fixture
def sample_features_target(sample_dataframe):
    """Fixture for features and target split."""
    X = sample_dataframe.drop('Class', axis=1)
    y = sample_dataframe['Class']
    return X, y


@pytest.fixture
def artifacts_dir():
    """Fixture for artifacts directory path."""
    return project_root / 'artifacts'


@pytest.fixture
def models_dir():
    """Fixture for models directory path."""
    return project_root / 'models'


@pytest.fixture
def data_dir():
    """Fixture for data directory path."""
    return project_root / 'data'


@pytest.fixture
def model_artifacts(artifacts_dir):
    """
    Fixture for loading model artifacts.
    
    Returns tuple of (model, scaler, threshold) if artifacts exist,
    otherwise skips the test.
    """
    model_path = artifacts_dir / 'fraud_model.pkl'
    scaler_path = artifacts_dir / 'scaler.pkl'
    threshold_path = artifacts_dir / 'threshold.txt'
    
    if not all([model_path.exists(), scaler_path.exists(), threshold_path.exists()]):
        pytest.skip("Model artifacts not found. Run model serialization first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    
    return model, scaler, threshold


@pytest.fixture
def api_client():
    """
    Fixture for FastAPI test client.
    
    Returns TestClient for API testing.
    """
    from fastapi.testclient import TestClient
    
    # Import the app
    sys.path.insert(0, str(project_root / 'api'))
    
    try:
        from api.main import app
        client = TestClient(app)
        return client
    except Exception as e:
        pytest.skip(f"Could not create API client: {e}")


@pytest.fixture(scope="session")
def random_seed():
    """Fixture for consistent random seed."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed):
    """Automatically set random seed for all tests."""
    np.random.seed(random_seed)


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )

