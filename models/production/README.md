# Production Model

## Model Information

**Model:** XGBoost
**Type:** advanced
**Version:** 1.0.0
**Selection Date:** 2026-01-15 04:39:35

---

## Performance Metrics

| Metric | Score |
|--------|-------|
| **PR-AUC** | 0.8786 ⭐ |
| **ROC-AUC** | 0.9684 |
| **Precision** | 0.8817 |
| **Recall** | 0.8367 |
| **F1-Score** | 0.8586 |
| **Accuracy** | 0.9995 |

---

## Optimal Threshold

**Threshold:** 0.90

At this threshold:
- **Precision:** 0.9419
- **Recall:** 0.8265
- **F1-Score:** 0.8804

**Confusion Matrix:**
```
TP:   81  |  FP:    5
FN:   17  |  TN: 56859
```

---

## Usage

### Load Model

```python
import joblib
import json

# Load model and scaler
model = joblib.load('models/production/production_model.pkl')
scaler = joblib.load('models/production/production_scaler.pkl')

# Load threshold
with open('models/production/production_threshold.json', 'r') as f:
    threshold_config = json.load(f)
    optimal_threshold = threshold_config['optimal_threshold']
```

### Make Predictions

```python
# Scale features
X_scaled = scaler.transform(X)

# Get probabilities
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Apply optimal threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)
```

---

## Files

- `production_model.pkl` - Trained model
- `production_scaler.pkl` - Feature scaler
- `production_threshold.json` - Optimal threshold configuration
- `production_metadata.json` - Model metadata and metrics
- `model_comparison.csv` - Comparison with other models
- `model_comparison.png` - Visual comparison
- `README.md` - This file

---

**Model is locked and ready for production deployment!** 🚀
