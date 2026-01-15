"""
Model Serialization for Deployment

Objective: Prepare model for deployment by serializing model, scaler, and threshold.

This script:
1. Loads the production model from models/production/
2. Serializes model & scaler
3. Stores in artifacts/ directory for deployment

Output:
- artifacts/fraud_model.pkl
- artifacts/scaler.pkl
- artifacts/threshold.txt

Author: Your Name
Date: 2026-01-15
"""

import os
import sys
import json
import joblib
import shutil
from pathlib import Path

# Add src to path
sys.path.append('src')


def create_artifacts_directory():
    """Create artifacts directory if it doesn't exist."""
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    print(f"✓ Artifacts directory ready: {artifacts_dir.absolute()}")
    return artifacts_dir


def load_production_model():
    """Load production model, scaler, and threshold."""
    print("\n" + "="*60)
    print("LOADING PRODUCTION MODEL")
    print("="*60)
    
    production_dir = Path('models/production')
    
    # Check if production model exists
    if not production_dir.exists():
        raise FileNotFoundError(
            "Production model not found! Please run final_selection.py first:\n"
            "  python src/models/final_selection.py"
        )
    
    # Load model
    model_path = production_dir / 'production_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"✓ Loaded model: {model_path}")
    
    # Load scaler
    scaler_path = production_dir / 'production_scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded scaler: {scaler_path}")
    
    # Load threshold
    threshold_path = production_dir / 'production_threshold.json'
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold not found: {threshold_path}")
    with open(threshold_path, 'r', encoding='utf-8') as f:
        threshold_config = json.load(f)
    threshold = threshold_config['optimal_threshold']
    print(f"✓ Loaded threshold: {threshold}")
    
    # Load metadata
    metadata_path = production_dir / 'production_metadata.json'
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✓ Loaded metadata: {metadata_path}")
    
    return model, scaler, threshold, metadata


def serialize_model(model, scaler, threshold, metadata, artifacts_dir):
    """Serialize model, scaler, and threshold to artifacts directory."""
    print("\n" + "="*60)
    print("SERIALIZING MODEL FOR DEPLOYMENT")
    print("="*60)
    
    # 1. Save model
    model_path = artifacts_dir / 'fraud_model.pkl'
    joblib.dump(model, model_path)
    model_size = model_path.stat().st_size / 1024  # KB
    print(f"✓ Saved model: {model_path} ({model_size:.2f} KB)")
    
    # 2. Save scaler
    scaler_path = artifacts_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    scaler_size = scaler_path.stat().st_size / 1024  # KB
    print(f"✓ Saved scaler: {scaler_path} ({scaler_size:.2f} KB)")
    
    # 3. Save threshold
    threshold_path = artifacts_dir / 'threshold.txt'
    with open(threshold_path, 'w', encoding='utf-8') as f:
        f.write(f"{threshold}\n")
    print(f"✓ Saved threshold: {threshold_path} (threshold={threshold})")
    
    # 4. Save metadata (optional, for reference)
    if metadata:
        metadata_path = artifacts_dir / 'model_info.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")
    
    # 5. Create README
    create_deployment_readme(artifacts_dir, threshold, metadata)
    
    return model_path, scaler_path, threshold_path


def create_deployment_readme(artifacts_dir, threshold, metadata):
    """Create README for deployment."""
    readme_content = f"""# Fraud Detection Model - Deployment Artifacts

## 📦 Contents

This directory contains the serialized model artifacts ready for deployment:

- **fraud_model.pkl** - Trained fraud detection model
- **scaler.pkl** - Feature scaler (StandardScaler)
- **threshold.txt** - Optimal classification threshold
- **model_info.json** - Model metadata and performance metrics

## 🎯 Model Information

**Model Type:** {metadata.get('model_name', 'N/A') if metadata else 'N/A'}
**Optimal Threshold:** {threshold}
**Training Date:** {metadata.get('training_date', 'N/A') if metadata else 'N/A'}

### Performance Metrics

"""
    
    if metadata and 'performance' in metadata:
        perf = metadata['performance']
        readme_content += f"""- **PR-AUC:** {perf.get('PR-AUC', 'N/A'):.4f}
- **ROC-AUC:** {perf.get('ROC-AUC', 'N/A'):.4f}
- **Precision:** {perf.get('Precision', 'N/A'):.4f}
- **Recall:** {perf.get('Recall', 'N/A'):.4f}
- **F1-Score:** {perf.get('F1-Score', 'N/A'):.4f}
"""
    
    readme_content += f"""
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

print(f"Model loaded! Threshold: {{threshold}}")
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
print(f"Fraudulent transactions: {{predictions.sum()}}")
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
**Threshold:** {threshold} (optimized for balanced precision-recall)

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
    return {{"probability": prob, "is_fraud": is_fraud}}
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
    return jsonify({{'probability': float(prob), 'is_fraud': is_fraud}})
```

## 📝 Notes

- Always scale features before prediction
- Use the optimal threshold for classification
- Monitor model performance in production
- Retrain periodically with new data

---

**Generated:** {metadata.get('training_date', 'N/A') if metadata else 'N/A'}
**Model:** {metadata.get('model_name', 'N/A') if metadata else 'N/A'}
**Threshold:** {threshold}
"""
    
    readme_path = artifacts_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Saved README: {readme_path}")


def main():
    """Main serialization pipeline."""
    print("\n" + "="*60)
    print("MODEL SERIALIZATION FOR DEPLOYMENT")
    print("="*60)
    print("\nObjective: Prepare model for deployment")
    print("Output: artifacts/fraud_model.pkl, scaler.pkl, threshold.txt")
    
    try:
        # 1. Create artifacts directory
        artifacts_dir = create_artifacts_directory()
        
        # 2. Load production model
        model, scaler, threshold, metadata = load_production_model()
        
        # 3. Serialize to artifacts
        model_path, scaler_path, threshold_path = serialize_model(
            model, scaler, threshold, metadata, artifacts_dir
        )
        
        # 4. Summary
        print("\n" + "="*60)
        print("✓ MODEL SERIALIZATION COMPLETE!")
        print("="*60)
        print(f"\n📁 Artifacts saved to: {artifacts_dir.absolute()}")
        print("\nFiles created:")
        print(f"  ✓ fraud_model.pkl")
        print(f"  ✓ scaler.pkl")
        print(f"  ✓ threshold.txt")
        print(f"  ✓ model_info.json")
        print(f"  ✓ README.md")
        
        print("\n🚀 READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("  1. Review artifacts/README.md for deployment instructions")
        print("  2. Copy artifacts/ to your deployment environment")
        print("  3. Use fraud_model.pkl, scaler.pkl, and threshold.txt")
        
        print("\n💡 Quick test:")
        print("  python -c \"import joblib; m=joblib.load('artifacts/fraud_model.pkl'); print('Model loaded:', type(m))\"")
        
        return artifacts_dir
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solution:")
        print("  Run final model selection first:")
        print("  python src/models/final_selection.py")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()

