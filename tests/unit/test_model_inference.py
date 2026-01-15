"""
Unit tests for model inference

Tests model loading and prediction logic.

Author: Your Name
Date: 2026-01-15
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


@pytest.mark.unit
class TestModelArtifacts:
    """Tests for model artifacts."""
    
    def test_model_artifacts_exist(self, artifacts_dir):
        """Test that model artifacts exist."""
        model_path = artifacts_dir / 'fraud_model.pkl'
        scaler_path = artifacts_dir / 'scaler.pkl'
        threshold_path = artifacts_dir / 'threshold.txt'
        
        if not artifacts_dir.exists():
            pytest.skip("Artifacts directory not found")
        
        # At least one of the artifacts should exist
        assert any([
            model_path.exists(),
            scaler_path.exists(),
            threshold_path.exists()
        ]), "No model artifacts found. Run model serialization first."
    
    def test_model_can_be_loaded(self, model_artifacts):
        """Test that model can be loaded."""
        model, scaler, threshold = model_artifacts
        
        assert model is not None
        assert scaler is not None
        assert threshold is not None
        assert isinstance(threshold, float)
    
    def test_model_has_predict_method(self, model_artifacts):
        """Test that model has predict method."""
        model, scaler, threshold = model_artifacts
        
        assert hasattr(model, 'predict')
        assert callable(model.predict)
    
    def test_model_has_predict_proba_method(self, model_artifacts):
        """Test that model has predict_proba method."""
        model, scaler, threshold = model_artifacts
        
        assert hasattr(model, 'predict_proba')
        assert callable(model.predict_proba)
    
    def test_scaler_has_transform_method(self, model_artifacts):
        """Test that scaler has transform method."""
        model, scaler, threshold = model_artifacts
        
        assert hasattr(scaler, 'transform')
        assert callable(scaler.transform)
    
    def test_threshold_in_valid_range(self, model_artifacts):
        """Test that threshold is in valid range [0, 1]."""
        model, scaler, threshold = model_artifacts
        
        assert 0.0 <= threshold <= 1.0


@pytest.mark.unit
class TestModelPrediction:
    """Tests for model prediction."""
    
    def test_model_predict_single_transaction(self, model_artifacts, sample_legitimate_transaction):
        """Test prediction on a single transaction."""
        model, scaler, threshold = model_artifacts
        
        # Prepare transaction
        trans_df = pd.DataFrame([sample_legitimate_transaction])
        trans_scaled = scaler.transform(trans_df)
        
        # Predict
        prediction = model.predict(trans_scaled)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    def test_model_predict_proba_single_transaction(self, model_artifacts, sample_legitimate_transaction):
        """Test predict_proba on a single transaction."""
        model, scaler, threshold = model_artifacts
        
        # Prepare transaction
        trans_df = pd.DataFrame([sample_legitimate_transaction])
        trans_scaled = scaler.transform(trans_df)
        
        # Predict probabilities
        proba = model.predict_proba(trans_scaled)
        
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all((proba >= 0) & (proba <= 1))  # Probabilities in [0, 1]
    
    def test_model_predict_batch(self, model_artifacts, sample_batch_transactions):
        """Test prediction on a batch of transactions."""
        model, scaler, threshold = model_artifacts
        
        # Prepare batch
        batch_df = pd.DataFrame(sample_batch_transactions)
        batch_scaled = scaler.transform(batch_df)
        
        # Predict
        predictions = model.predict(batch_scaled)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_batch_transactions)
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_predict_proba_batch(self, model_artifacts, sample_batch_transactions):
        """Test predict_proba on a batch of transactions."""
        model, scaler, threshold = model_artifacts
        
        # Prepare batch
        batch_df = pd.DataFrame(sample_batch_transactions)
        batch_scaled = scaler.transform(batch_df)
        
        # Predict probabilities
        proba = model.predict_proba(batch_scaled)
        
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (len(sample_batch_transactions), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_fraud_probability_extraction(self, model_artifacts, sample_legitimate_transaction):
        """Test extracting fraud probability from predict_proba."""
        model, scaler, threshold = model_artifacts
        
        # Prepare transaction
        trans_df = pd.DataFrame([sample_legitimate_transaction])
        trans_scaled = scaler.transform(trans_df)
        
        # Get fraud probability
        proba = model.predict_proba(trans_scaled)
        fraud_prob = proba[0, 1]
        
        assert isinstance(fraud_prob, (float, np.floating))
        assert 0.0 <= fraud_prob <= 1.0
    
    def test_threshold_based_prediction(self, model_artifacts, sample_legitimate_transaction):
        """Test threshold-based prediction."""
        model, scaler, threshold = model_artifacts
        
        # Prepare transaction
        trans_df = pd.DataFrame([sample_legitimate_transaction])
        trans_scaled = scaler.transform(trans_df)
        
        # Get fraud probability
        proba = model.predict_proba(trans_scaled)
        fraud_prob = proba[0, 1]
        
        # Apply threshold
        is_fraud = fraud_prob >= threshold
        
        assert isinstance(is_fraud, (bool, np.bool_))


@pytest.mark.unit
class TestScalerTransformation:
    """Tests for scaler transformation."""
    
    def test_scaler_transform_preserves_shape(self, model_artifacts, sample_legitimate_transaction):
        """Test that scaler preserves data shape."""
        model, scaler, threshold = model_artifacts
        
        trans_df = pd.DataFrame([sample_legitimate_transaction])
        trans_scaled = scaler.transform(trans_df)
        
        assert trans_scaled.shape == trans_df.shape
    
    def test_scaler_transform_returns_array(self, model_artifacts, sample_legitimate_transaction):
        """Test that scaler returns numpy array."""
        model, scaler, threshold = model_artifacts
        
        trans_df = pd.DataFrame([sample_legitimate_transaction])
        trans_scaled = scaler.transform(trans_df)
        
        assert isinstance(trans_scaled, np.ndarray)
    
    def test_scaler_handles_batch(self, model_artifacts, sample_batch_transactions):
        """Test that scaler handles batch transformation."""
        model, scaler, threshold = model_artifacts
        
        batch_df = pd.DataFrame(sample_batch_transactions)
        batch_scaled = scaler.transform(batch_df)
        
        assert batch_scaled.shape == batch_df.shape
        assert isinstance(batch_scaled, np.ndarray)

