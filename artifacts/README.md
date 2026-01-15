# Fraud Detection Model - Deployment Artifacts

## 📦 Contents

This directory contains the serialized model artifacts ready for deployment:

- **fraud_model.pkl** - Trained fraud detection model
- **scaler.pkl** - Feature scaler (StandardScaler)
- **threshold.txt** - Optimal classification threshold
- **model_info.json** - Model metadata and performance metrics

## 🎯 Model Information

**Model Type:** XGBoost
**Optimal Threshold:** 0.9
**Training Date:** N/A

### Performance Metrics


## 🚀 Usage

### Load Model

```python
import joblib

# Load model and scaler
model = joblib.load('artifacts/fraud_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

# Load threshold
with open('artifacts/threshold.txt', 'r') as f:
    threshold = float(f.read().strip())

print(f"Model loaded! Threshold: {threshold}")
```

### Make Predictions

```python
import pandas as pd

# Prepare your data (30 features)
# X should be a DataFrame or array with shape (n_samples, 30)

# Scale features
X_scaled = scaler.transform(X)

# Get probabilities
fraud_probabilities = model.predict_proba(X_scaled)[:, 1]

# Apply threshold
predictions = (fraud_probabilities >= threshold).astype(int)

# Results
print(f"Fraudulent transactions: {predictions.sum()}")
```

## 📋 Deployment Checklist

- [ ] Copy all files from artifacts/ to deployment environment
- [ ] Install required dependencies (scikit-learn, numpy, pandas)
- [ ] Test model loading and prediction
- [ ] Set up monitoring for model performance
- [ ] Configure logging for predictions
- [ ] Implement data validation
- [ ] Set up alerts for high fraud probability transactions

## 🔧 Requirements

```
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
joblib>=1.0.0
```

## 📊 Model Details

**Input:** 30 numerical features (Time, V1-V28, Amount)
**Output:** Fraud probability (0.0 to 1.0)
**Threshold:** 0.9 (optimized for balanced precision-recall)

## 🎯 Deployment Options

### Option 1: FastAPI

```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('artifacts/fraud_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

with open('artifacts/threshold.txt', 'r') as f:
    threshold = float(f.read().strip())

@app.post("/predict")
def predict(features: list):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    is_fraud = int(prob >= threshold)
    return {"probability": prob, "is_fraud": is_fraud}
```

### Option 2: Flask

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('artifacts/fraud_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

with open('artifacts/threshold.txt', 'r') as f:
    threshold = float(f.read().strip())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array(data['features']).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    is_fraud = int(prob >= threshold)
    return jsonify({'probability': float(prob), 'is_fraud': is_fraud})
```

## 📝 Notes

- Always scale features before prediction
- Use the optimal threshold for classification
- Monitor model performance in production
- Retrain periodically with new data

---

**Generated:** N/A
**Model:** XGBoost
**Threshold:** 0.9
