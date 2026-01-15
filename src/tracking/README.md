# 🔬 Experiment Tracking with MLflow + DagsHub

This module provides comprehensive experiment tracking for the fraud detection project using **MLflow** with optional **DagsHub** integration.

---

## 📋 **What Gets Tracked**

### **1. Parameters**
- Model hyperparameters (max_depth, learning_rate, etc.)
- Data configuration (train/test split, scaling method)
- Preprocessing steps
- Random seeds

### **2. Metrics**
- **PR-AUC** (Precision-Recall AUC) - Primary metric for imbalanced data
- **ROC-AUC** (Receiver Operating Characteristic AUC)
- **Precision, Recall, F1-Score**
- **Confusion Matrix** (TP, FP, TN, FN)
- **Accuracy, FPR, FNR**

### **3. Artifacts**
- **Models** (saved in MLflow format)
- **Plots**:
  - Confusion Matrix
  - Precision-Recall Curve
  - Feature Importance
- **Data**:
  - Feature importance CSV
  - Parameters JSON
  - Metrics summary

### **4. Model Registry**
- Version control for models
- Stage management (Staging, Production)
- Model lineage tracking

---

## 🚀 **Quick Start**

### **Option 1: Simple Function API**

```python
from tracking.mlflow_utils import track_experiment

# Track a complete experiment
metrics = track_experiment(
    experiment_name="fraud-detection",
    run_name="lightgbm-v1",
    model=trained_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    params={"max_depth": 9, "learning_rate": 0.05},
    tags={"model_type": "LightGBM", "version": "v1"}
)
```

### **Option 2: Class-Based API (More Control)**

```python
from tracking.mlflow_utils import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(experiment_name="fraud-detection")

# Start a run
tracker.start_run(
    run_name="lightgbm-v1",
    tags={"model_type": "LightGBM"},
    description="LightGBM with default parameters"
)

# Log parameters
tracker.log_params({
    "max_depth": 9,
    "learning_rate": 0.05,
    "n_estimators": 100
})

# Log metrics
tracker.log_metrics({
    "pr_auc": 0.8689,
    "roc_auc": 0.9684,
    "precision": 0.88,
    "recall": 0.84
})

# Log complete evaluation (metrics + plots + model)
tracker.log_complete_evaluation(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    model_name="lightgbm"
)

# End run
tracker.end_run()
```

### **Option 3: Context Manager**

```python
from tracking.mlflow_utils import MLflowTracker

with MLflowTracker(experiment_name="fraud-detection") as tracker:
    tracker.start_run(run_name="experiment-1")
    tracker.log_params(params)
    tracker.log_metrics(metrics)
    tracker.log_model(model, "model")
    # Automatically ends run when exiting context
```

---

## 📊 **Track All Models**

Run the provided script to track all your trained models:

```bash
python src/tracking/track_experiments.py
```

This will track:
- ✅ Random Forest (baseline)
- ✅ XGBoost
- ✅ LightGBM (default) - **BEST MODEL**
- ✅ LightGBM (optimized)

---

## 🌐 **DagsHub Integration**

### **Setup DagsHub (Optional)**

1. **Create a DagsHub account**: https://dagshub.com
2. **Create a repository**: `fraud-detection`
3. **Set environment variables**:

```bash
# Windows (PowerShell)
$env:DAGSHUB_USER = "your-username"
$env:DAGSHUB_TOKEN = "your-token"

# Linux/Mac
export DAGSHUB_USER="your-username"
export DAGSHUB_TOKEN="your-token"
```

4. **Initialize tracker with DagsHub**:

```python
tracker = MLflowTracker(
    experiment_name="fraud-detection",
    dagshub_repo="fraud-detection",
    dagshub_user="your-username"
)
```

5. **View experiments**: https://dagshub.com/your-username/fraud-detection/experiments

---

## 🖥️ **View Experiments Locally**

### **Start MLflow UI**

```bash
mlflow ui
```

Then open: http://localhost:5000

### **What You'll See**

- **Experiments**: All your experiment runs
- **Metrics**: Compare PR-AUC, ROC-AUC, etc. across runs
- **Parameters**: See which hyperparameters were used
- **Artifacts**: Download models, plots, and data
- **Charts**: Visualize metric trends

---

## 📁 **File Structure**

```
src/tracking/
├── mlflow_utils.py          # Main tracking utilities
├── track_experiments.py     # Script to track all models
└── README.md               # This file

mlruns/                      # Local MLflow tracking data
├── 0/                      # Experiment ID
│   ├── meta.yaml
│   └── <run-id>/          # Individual runs
│       ├── artifacts/     # Models, plots, data
│       ├── metrics/       # Metric values
│       ├── params/        # Parameter values
│       └── tags/          # Tags and metadata
```

---

## 🎯 **Best Practices**

### **1. Naming Conventions**

```python
# Good run names
run_name="lightgbm-default-v1"
run_name="xgboost-tuned-2026-01-15"
run_name="rf-baseline"

# Bad run names
run_name="test"
run_name="model1"
```

### **2. Use Tags**

```python
tags={
    "model_type": "LightGBM",
    "version": "v1",
    "best": "true",
    "optimizer": "Optuna"
}
```

### **3. Add Descriptions**

```python
description="LightGBM with default parameters. Best model with PR-AUC=0.8689"
```

### **4. Log Everything**

- ✅ All hyperparameters
- ✅ Data statistics (train size, fraud ratio)
- ✅ Preprocessing steps
- ✅ Random seeds
- ✅ Training time

---

## 📈 **Example Output**

```
============================================================
MLFLOW RUN STARTED
============================================================
Run ID: a1b2c3d4e5f6
Run Name: lightgbm-default
Description: LightGBM with default parameters - BEST MODEL
============================================================

✓ Logged 8 parameters
✓ Logged 12 metrics
✓ Logged confusion matrix
✓ Logged PR curve
✓ Logged feature importance (plot + CSV)
✓ Logged model: lightgbm_default

============================================================
✓ COMPLETE EVALUATION LOGGED
============================================================
```

---

## 🔧 **Troubleshooting**

### **Issue: DagsHub connection fails**

**Solution**: Check credentials and network connection
```python
# Verify environment variables
import os
print(os.getenv('DAGSHUB_USER'))
print(os.getenv('DAGSHUB_TOKEN'))
```

### **Issue: MLflow UI shows no experiments**

**Solution**: Check tracking URI
```python
import mlflow
print(mlflow.get_tracking_uri())
# Should be: file:./mlruns or DagsHub URL
```

### **Issue: Model logging fails**

**Solution**: Ensure model type is supported
```python
# Supported: sklearn, xgboost, lightgbm
# For custom models, use mlflow.pyfunc
```

---

## 📚 **Resources**

- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **DagsHub Documentation**: https://dagshub.com/docs
- **MLflow Tracking**: https://mlflow.org/docs/latest/tracking.html
- **Model Registry**: https://mlflow.org/docs/latest/model-registry.html

---

## ✨ **Summary**

This tracking module provides:
- ✅ **Complete experiment tracking** (params, metrics, artifacts)
- ✅ **Easy-to-use API** (function or class-based)
- ✅ **DagsHub integration** (optional remote tracking)
- ✅ **Model registry** (versioning and deployment)
- ✅ **Visualization** (MLflow UI or DagsHub)

**Perfect for:**
- 🎯 Comparing different models
- 📊 Tracking hyperparameter tuning
- 🚀 Model versioning and deployment
- 📝 Documenting experiments for portfolio/interviews

---

**Happy Tracking!** 🚀

