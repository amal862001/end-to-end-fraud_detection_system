"""
Sample test data fixtures

Provides sample transactions for testing purposes.

Author: Your Name
Date: 2026-01-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List


# Sample legitimate transaction (from actual dataset)
LEGITIMATE_TRANSACTION = {
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


# Sample fraudulent transaction (synthetic, high fraud probability)
FRAUD_TRANSACTION = {
    "Time": 50000.0,
    "V1": -3.5,
    "V2": 4.2,
    "V3": -5.1,
    "V4": 3.8,
    "V5": -2.9,
    "V6": 1.5,
    "V7": -4.3,
    "V8": 2.7,
    "V9": -3.2,
    "V10": 4.1,
    "V11": -2.5,
    "V12": 3.6,
    "V13": -4.8,
    "V14": 5.2,
    "V15": -3.7,
    "V16": 2.9,
    "V17": -4.5,
    "V18": 3.3,
    "V19": -2.8,
    "V20": 4.6,
    "V21": -3.1,
    "V22": 2.4,
    "V23": -5.3,
    "V24": 4.9,
    "V25": -3.4,
    "V26": 2.1,
    "V27": -4.7,
    "V28": 5.5,
    "Amount": 1500.00
}


def get_sample_transaction(fraud: bool = False) -> Dict[str, float]:
    """
    Get a sample transaction.
    
    Args:
        fraud: If True, return fraudulent transaction, else legitimate
    
    Returns:
        Dictionary with transaction features
    """
    return FRAUD_TRANSACTION.copy() if fraud else LEGITIMATE_TRANSACTION.copy()


def get_sample_dataframe(n_samples: int = 100, fraud_ratio: float = 0.1) -> pd.DataFrame:
    """
    Generate a sample DataFrame for testing.
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Ratio of fraudulent transactions
    
    Returns:
        DataFrame with sample transactions
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Generate legitimate transactions (small variations)
    legit_data = []
    for _ in range(n_legit):
        trans = LEGITIMATE_TRANSACTION.copy()
        # Add small random noise
        for key in trans:
            if key != 'Time':
                trans[key] += np.random.normal(0, 0.1)
        trans['Class'] = 0
        legit_data.append(trans)
    
    # Generate fraudulent transactions (small variations)
    fraud_data = []
    for _ in range(n_fraud):
        trans = FRAUD_TRANSACTION.copy()
        # Add small random noise
        for key in trans:
            if key != 'Time':
                trans[key] += np.random.normal(0, 0.1)
        trans['Class'] = 1
        fraud_data.append(trans)
    
    # Combine and shuffle
    all_data = legit_data + fraud_data
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def get_batch_transactions(n: int = 5, fraud_ratio: float = 0.2) -> List[Dict[str, float]]:
    """
    Get a batch of sample transactions.
    
    Args:
        n: Number of transactions
        fraud_ratio: Ratio of fraudulent transactions
    
    Returns:
        List of transaction dictionaries
    """
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud
    
    transactions = []
    
    # Add legitimate transactions
    for _ in range(n_legit):
        transactions.append(get_sample_transaction(fraud=False))
    
    # Add fraudulent transactions
    for _ in range(n_fraud):
        transactions.append(get_sample_transaction(fraud=True))
    
    return transactions

