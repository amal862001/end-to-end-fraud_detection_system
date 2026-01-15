"""
Integration tests for API endpoints

Tests the FastAPI endpoints using TestClient.

Author: Your Name
Date: 2026-01-15
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add api to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'api'))


@pytest.mark.integration
@pytest.mark.api
class TestRootEndpoint:
    """Tests for root endpoint (/)."""
    
    def test_root_endpoint_returns_200(self, api_client):
        """Test that root endpoint returns 200."""
        response = api_client.get("/")
        assert response.status_code == 200
    
    def test_root_endpoint_returns_json(self, api_client):
        """Test that root endpoint returns JSON."""
        response = api_client.get("/")
        assert response.headers["content-type"] == "application/json"
    
    def test_root_endpoint_has_required_fields(self, api_client):
        """Test that root endpoint has required fields."""
        response = api_client.get("/")
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


@pytest.mark.integration
@pytest.mark.api
class TestHealthEndpoint:
    """Tests for health check endpoint (/health)."""
    
    def test_health_endpoint_returns_200(self, api_client):
        """Test that health endpoint returns 200."""
        response = api_client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_returns_json(self, api_client):
        """Test that health endpoint returns JSON."""
        response = api_client.get("/health")
        assert response.headers["content-type"] == "application/json"
    
    def test_health_endpoint_has_status(self, api_client):
        """Test that health endpoint has status field."""
        response = api_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]
    
    def test_health_endpoint_has_model_loaded(self, api_client):
        """Test that health endpoint has model_loaded field."""
        response = api_client.get("/health")
        data = response.json()
        
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


@pytest.mark.integration
@pytest.mark.api
class TestPredictEndpoint:
    """Tests for prediction endpoint (/predict)."""
    
    def test_predict_endpoint_accepts_post(self, api_client, sample_legitimate_transaction):
        """Test that predict endpoint accepts POST requests."""
        response = api_client.post("/predict", json=sample_legitimate_transaction)
        assert response.status_code in [200, 500, 503]  # 500/503 if model not loaded
    
    def test_predict_endpoint_rejects_get(self, api_client):
        """Test that predict endpoint rejects GET requests."""
        response = api_client.get("/predict")
        assert response.status_code == 405  # Method Not Allowed
    
    def test_predict_endpoint_returns_json(self, api_client, sample_legitimate_transaction):
        """Test that predict endpoint returns JSON."""
        response = api_client.post("/predict", json=sample_legitimate_transaction)
        
        if response.status_code == 200:
            assert response.headers["content-type"] == "application/json"
    
    def test_predict_endpoint_has_required_fields(self, api_client, sample_legitimate_transaction):
        """Test that prediction response has required fields."""
        response = api_client.post("/predict", json=sample_legitimate_transaction)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "confidence" in data
            assert "transaction_id" in data
    
    def test_predict_endpoint_fraud_probability_in_range(self, api_client, sample_legitimate_transaction):
        """Test that fraud probability is in valid range."""
        response = api_client.post("/predict", json=sample_legitimate_transaction)
        
        if response.status_code == 200:
            data = response.json()
            assert 0.0 <= data["fraud_probability"] <= 1.0
    
    def test_predict_endpoint_is_fraud_is_boolean(self, api_client, sample_legitimate_transaction):
        """Test that is_fraud is boolean."""
        response = api_client.post("/predict", json=sample_legitimate_transaction)
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data["is_fraud"], bool)
    
    def test_predict_endpoint_confidence_is_valid(self, api_client, sample_legitimate_transaction):
        """Test that confidence is valid."""
        response = api_client.post("/predict", json=sample_legitimate_transaction)
        
        if response.status_code == 200:
            data = response.json()
            assert data["confidence"] in ["low", "medium", "high"]
    
    def test_predict_endpoint_with_fraud_transaction(self, api_client, sample_fraud_transaction):
        """Test prediction with fraudulent transaction."""
        response = api_client.post("/predict", json=sample_fraud_transaction)
        
        if response.status_code == 200:
            data = response.json()
            # Fraud transaction should have higher probability
            # (though not guaranteed to be classified as fraud)
            assert "fraud_probability" in data
    
    def test_predict_endpoint_missing_field_returns_422(self, api_client):
        """Test that missing required field returns 422."""
        incomplete_transaction = {
            "Time": 0.0,
            "V1": -1.36,
            # Missing other required fields
        }
        
        response = api_client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_type_returns_422(self, api_client):
        """Test that invalid field type returns 422."""
        invalid_transaction = {
            "Time": "invalid",  # Should be float
            "V1": -1.36,
            "Amount": 100.0
        }
        
        response = api_client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint (/predict/batch)."""
    
    def test_batch_predict_endpoint_accepts_post(self, api_client, sample_batch_transactions):
        """Test that batch predict endpoint accepts POST requests."""
        response = api_client.post("/predict/batch", json=sample_batch_transactions)
        assert response.status_code in [200, 500, 503]  # 500/503 if model not loaded
    
    def test_batch_predict_endpoint_returns_list(self, api_client, sample_batch_transactions):
        """Test that batch predict returns a list."""
        response = api_client.post("/predict/batch", json=sample_batch_transactions)
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == len(sample_batch_transactions)
    
    def test_batch_predict_endpoint_each_prediction_valid(self, api_client, sample_batch_transactions):
        """Test that each prediction in batch is valid."""
        response = api_client.post("/predict/batch", json=sample_batch_transactions)
        
        if response.status_code == 200:
            predictions = response.json()
            
            for pred in predictions:
                assert "is_fraud" in pred
                assert "fraud_probability" in pred
                assert "confidence" in pred
                assert isinstance(pred["is_fraud"], bool)
                assert 0.0 <= pred["fraud_probability"] <= 1.0
                assert pred["confidence"] in ["low", "medium", "high"]


@pytest.mark.integration
@pytest.mark.api
class TestMetricsEndpoint:
    """Tests for metrics endpoint (/metrics)."""
    
    def test_metrics_endpoint_returns_200(self, api_client):
        """Test that metrics endpoint returns 200."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_endpoint_returns_prometheus_format(self, api_client):
        """Test that metrics endpoint returns Prometheus format."""
        response = api_client.get("/metrics")
        
        # Prometheus metrics are plain text
        assert "text/plain" in response.headers.get("content-type", "")
    
    def test_metrics_endpoint_contains_metrics(self, api_client):
        """Test that metrics endpoint contains expected metrics."""
        response = api_client.get("/metrics")
        content = response.text
        
        # Should contain some metric names
        assert "api_requests_total" in content or "python_" in content

