"""
Generate Sample Traffic for Monitoring

This script generates sample API traffic to test the monitoring system.

Author: Your Name
Date: 2026-01-15
"""

import requests
import random
import time
import json
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000"

# Sample legitimate transaction
LEGITIMATE_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
}

# Sample fraudulent transaction
FRAUDULENT_TRANSACTION = {
    "Time": 406.0,
    "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99, "V5": -0.52,
    "V6": -1.43, "V7": -2.54, "V8": 1.39, "V9": -2.77, "V10": -2.77,
    "V11": 3.20, "V12": -2.90, "V13": -0.60, "V14": -4.29, "V15": 0.39,
    "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42, "V20": 0.13,
    "V21": 0.52, "V22": -0.04, "V23": -0.47, "V24": 0.32, "V25": 0.04,
    "V26": 0.18, "V27": 0.26, "V28": -0.14,
    "Amount": 0.0
}


def generate_random_transaction():
    """Generate a random transaction (80% legitimate, 20% fraud)."""
    if random.random() < 0.8:
        # Legitimate transaction with some variation
        transaction = LEGITIMATE_TRANSACTION.copy()
        transaction["Amount"] = random.uniform(10, 500)
        transaction["Time"] = random.uniform(0, 172800)  # 48 hours
    else:
        # Fraudulent transaction with some variation
        transaction = FRAUDULENT_TRANSACTION.copy()
        transaction["Amount"] = random.uniform(0, 100)
        transaction["Time"] = random.uniform(0, 172800)
    
    return transaction


def send_prediction_request(transaction):
    """Send a prediction request to the API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            return False, response.text
            
    except Exception as e:
        return False, str(e)


def generate_traffic(duration_seconds=300, requests_per_second=2):
    """
    Generate traffic for a specified duration.
    
    Args:
        duration_seconds: How long to generate traffic (default: 5 minutes)
        requests_per_second: Target request rate (default: 2 req/s)
    """
    print("="*60)
    print("TRAFFIC GENERATOR FOR MONITORING")
    print("="*60)
    print(f"\nDuration: {duration_seconds} seconds")
    print(f"Target rate: {requests_per_second} requests/second")
    print(f"Total requests: ~{duration_seconds * requests_per_second}")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    start_time = time.time()
    request_count = 0
    success_count = 0
    fraud_count = 0
    legitimate_count = 0
    
    interval = 1.0 / requests_per_second
    
    print("\nGenerating traffic... (Press Ctrl+C to stop)\n")
    
    try:
        while time.time() - start_time < duration_seconds:
            # Generate and send request
            transaction = generate_random_transaction()
            success, result = send_prediction_request(transaction)
            
            request_count += 1
            
            if success:
                success_count += 1
                if result['is_fraud']:
                    fraud_count += 1
                else:
                    legitimate_count += 1
                
                # Print status every 10 requests
                if request_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = request_count / elapsed
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Requests: {request_count} | "
                          f"Success: {success_count} | "
                          f"Fraud: {fraud_count} | "
                          f"Legitimate: {legitimate_count} | "
                          f"Rate: {rate:.2f} req/s")
            else:
                print(f"❌ Request failed: {result}")
            
            # Wait for next request
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Duration: {elapsed:.2f} seconds")
    print(f"Total requests: {request_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {request_count - success_count}")
    print(f"Fraud detected: {fraud_count}")
    print(f"Legitimate: {legitimate_count}")
    print(f"Average rate: {request_count / elapsed:.2f} req/s")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate API traffic for monitoring")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds (default: 300)")
    parser.add_argument("--rate", type=float, default=2.0, help="Requests per second (default: 2.0)")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API URL")
    
    args = parser.parse_args()
    
    API_URL = args.url
    
    generate_traffic(
        duration_seconds=args.duration,
        requests_per_second=args.rate
    )

