"""
Test API Client

This script tests the FastAPI fraud detection service.

Author: Your Name
Date: 2026-01-15
"""

import requests
import json
from typing import Dict, List

# API base URL
BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint."""
    print("\n" + "="*60)
    print("TEST: Root Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_health():
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_predict_legitimate():
    """Test prediction with legitimate transaction."""
    print("\n" + "="*60)
    print("TEST: Predict Legitimate Transaction")
    print("="*60)
    
    # Example legitimate transaction
    transaction = {
        "Time": 0.0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Confidence: {result['confidence']}")
    
    return response.status_code == 200


def test_predict_fraud():
    """Test prediction with fraudulent transaction."""
    print("\n" + "="*60)
    print("TEST: Predict Fraudulent Transaction")
    print("="*60)
    
    # Example fraudulent transaction (high values in certain features)
    transaction = {
        "Time": 406.0,
        "V1": -2.3122265423263,
        "V2": 1.95199201064158,
        "V3": -1.60985073229769,
        "V4": 3.9979055875468,
        "V5": -0.522187864667764,
        "V6": -1.42654531920595,
        "V7": -2.53738730624579,
        "V8": 1.39165724829804,
        "V9": -2.77008927719433,
        "V10": -2.77227214465915,
        "V11": 3.20203320709635,
        "V12": -2.89990738849473,
        "V13": -0.595221881324605,
        "V14": -4.28925378244217,
        "V15": 0.389724120274487,
        "V16": -1.14074717980657,
        "V17": -2.83005567450437,
        "V18": -0.0168224681808257,
        "V19": 0.416955705037907,
        "V20": 0.126910559061474,
        "V21": 0.517232370861764,
        "V22": -0.0350493686052974,
        "V23": -0.465211076182388,
        "V24": 0.320198198514526,
        "V25": 0.0445191674731724,
        "V26": 0.177839798284401,
        "V27": 0.261145002567677,
        "V28": -0.143275874698919,
        "Amount": 0.0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Confidence: {result['confidence']}")
    
    return response.status_code == 200


def test_batch_predict():
    """Test batch prediction."""
    print("\n" + "="*60)
    print("TEST: Batch Prediction")
    print("="*60)
    
    # Multiple transactions
    transactions = [
        {
            "Time": 0.0,
            "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
            "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
            "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
            "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
            "V26": -0.19, "V27": 0.13, "V28": -0.02,
            "Amount": 149.62
        },
        {
            "Time": 100.0,
            "V1": 1.19, "V2": 0.27, "V3": 0.17, "V4": 0.45, "V5": -0.08,
            "V6": -0.08, "V7": -0.08, "V8": 0.09, "V9": -0.26, "V10": -0.17,
            "V11": 1.61, "V12": 1.07, "V13": 0.49, "V14": -0.14, "V15": 0.64,
            "V16": -0.47, "V17": -0.27, "V18": -0.15, "V19": -0.05, "V20": -0.23,
            "V21": -0.17, "V22": 0.13, "V23": 0.01, "V24": 0.14, "V25": 0.02,
            "V26": 0.31, "V27": 0.01, "V28": 0.01,
            "Amount": 2.69
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=transactions)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\n✓ Processed {len(results)} transactions:")
        for i, result in enumerate(results, 1):
            print(f"\n  Transaction {i}:")
            print(f"    Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
            print(f"    Probability: {result['fraud_probability']:.4f}")
            print(f"    Confidence: {result['confidence']}")
    
    return response.status_code == 200


def test_invalid_input():
    """Test with invalid input."""
    print("\n" + "="*60)
    print("TEST: Invalid Input (Negative Amount)")
    print("="*60)
    
    # Invalid transaction (negative amount)
    transaction = {
        "Time": 0.0,
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0,
        "Amount": -100.0  # Invalid: negative amount
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 422  # Validation error expected


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FRAUD DETECTION API - TEST SUITE")
    print("="*60)
    print(f"\nAPI URL: {BASE_URL}")
    print("\nMake sure the API is running:")
    print("  cd api")
    print("  uvicorn main:app --reload")
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Predict Legitimate", test_predict_legitimate),
        ("Predict Fraud", test_predict_fraud),
        ("Batch Prediction", test_batch_predict),
        ("Invalid Input", test_invalid_input),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ ERROR: Cannot connect to API at {BASE_URL}")
            print("   Make sure the API is running!")
            return
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()

