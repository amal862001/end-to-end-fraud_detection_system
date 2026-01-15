# 🎯 End-to-End Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-68%20Passed-success)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-ready** machine learning system for detecting fraudulent credit card transactions. This project demonstrates end-to-end ML engineering: from data preprocessing and model training to API deployment, monitoring, and testing.

---

## 📊 **Project Overview**

### **Problem Statement**
Detect fraudulent credit card transactions in a highly imbalanced dataset (0.17% fraud rate) while minimizing false positives.

### **Dataset**
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Cases:** 492 (0.17%)
- **Features:** 30 (PCA-transformed for privacy)

### **Key Metrics**
- **PR-AUC** (Precision-Recall AUC) - Primary metric for imbalanced data
- **Recall** - Fraud detection rate (minimize false negatives)
- **Precision** - Accuracy of fraud predictions (minimize false positives)

### **Best Model Performance**
| Metric | Value | Meaning |
|--------|-------|---------|
| **PR-AUC** | **0.8734** | Excellent for imbalanced data |
| **ROC-AUC** | 0.9834 | Outstanding overall performance |
| **Recall** | 86.7% | Catches 85-87 out of 98 frauds |
| **Precision** | 86.7% | 87% of predictions are correct |
| **F1-Score** | 0.8673 | Excellent balance |

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRAUD DETECTION SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Data       │───▶│  Training    │───▶│   Model      │
│   Pipeline   │    │  Pipeline    │    │  Selection   │
└──────────────┘    └──────────────┘    └──────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Preprocessing│    │  Imbalance   │    │ Optimization │
│ • Scaling    │    │  Handling    │    │ • Optuna     │
│ • Outliers   │    │ • SMOTE      │    │ • HPO        │
└──────────────┘    │ • Threshold  │    │ • CV         │
                    └──────────────┘    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Production  │
                    │    Model     │
                    └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   FastAPI    │    │  Monitoring  │    │   Testing    │
│   • REST     │    │ • Prometheus │    │ • Unit       │
│   • Batch    │    │ • Grafana    │    │ • Integration│
│   • Metrics  │    │ • Alerts     │    │ • 68 Tests   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                  │
        ▼                  ▼
┌──────────────────────────────────┐
│         Docker Deployment         │
│  • API Container                  │
│  • Monitoring Stack               │
│  • MLflow Tracking                │
└──────────────────────────────────┘
```

---

## 🚀 **Quick Start**

### **1. Clone & Setup**

```bash
# Clone repository
git clone https://github.com/yourusername/end-to-end-fraud_detection_system.git
cd end-to-end-fraud_detection_system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### **2. Download Dataset**

Download from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/raw/`.

### **3. Train Models**

```bash
# Train baseline models
python src/models/train.py

# Train advanced models (XGBoost, LightGBM)
python src/models/advanced_models.py

# Optimize hyperparameters
python src/models/optimize.py

# Select final production model
python src/models/final_selection.py
```

### **4. Start API**

```bash
# Start FastAPI server
python run_api.py

# Or using Docker
docker-compose -f api/docker-compose.yml up
```

### **5. Run Tests**

```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage --html
```

---

## 📁 **Project Structure**

```
end-to-end-fraud_detection_system/
│
├── data/
│   ├── raw/                    # Raw dataset (creditcard.csv)
│   └── processed/              # Processed data
│
├── src/
│   ├── data/
│   │   ├── load_data.py       # Data loading utilities
│   │   └── preprocess.py      # Preprocessing pipeline
│   │
│   ├── models/
│   │   ├── train.py           # Baseline models
│   │   ├── handle_imbalance.py # Imbalance handling
│   │   ├── advanced_models.py  # XGBoost & LightGBM
│   │   ├── optimize.py        # Hyperparameter optimization
│   │   ├── final_selection.py # Model selection
│   │   └── serialize_model.py # Model serialization
│   │
│   └── tracking/
│       └── mlflow_utils.py    # MLflow experiment tracking
│
├── api/
│   ├── main.py                # FastAPI application
│   ├── Dockerfile             # API container
│   └── docker-compose.yml     # Docker orchestration
│
├── monitoring/
│   ├── prometheus.yml         # Prometheus config
│   ├── grafana/               # Grafana dashboards
│   └── docker-compose.yml     # Monitoring stack
│
├── tests/
│   ├── unit/                  # Unit tests (45 tests)
│   ├── integration/           # Integration tests (23 tests)
│   └── conftest.py            # Test fixtures
│
├── models/                    # Saved models
│   ├── baseline/              # Baseline models
│   ├── advanced/              # Advanced models
│   ├── optimized/             # Optimized models
│   └── production/            # Production model
│
├── artifacts/                 # Production artifacts
│   ├── fraud_model.pkl        # Serialized model
│   ├── scaler.pkl             # Feature scaler
│   └── threshold.txt          # Decision threshold
│
├── notebooks/
│   ├── 01_data_profiling.ipynb # Data profiling
│   └── 02_eda.ipynb           # Exploratory analysis
│
└── examples/
    ├── demo_imbalance_handling.py  # Quick demo
    └── use_production_model.py     # Model usage
```

---

## 🎯 **Model Development & Results**

### **1. Baseline Models**

| Model | PR-AUC | Recall | Precision | Runtime |
|-------|--------|--------|-----------|---------|
| Logistic Regression | 0.7612 | 76.5% | 76.5% | ~2 min |
| Random Forest | 0.8171 | 82.7% | 81.8% | ~3 min |

### **2. Imbalance Handling**

**Techniques Implemented:**
- ✅ Class Weights
- ✅ SMOTE (Synthetic Minority Oversampling)
- ✅ Threshold Tuning

**Best Result:** Threshold 0.3 → Recall 88.8%, Precision 56.1%

### **3. Advanced Models**

| Model | PR-AUC | Recall | Precision | Improvement |
|-------|--------|--------|-----------|-------------|
| XGBoost | 0.8567 | 85.7% | 85.7% | +4.8% |
| LightGBM | 0.8689 | 86.7% | 86.7% | +6.3% |

### **4. Hyperparameter Optimization**

**Method:** Optuna with TPE sampler (50 trials)

**Final Model:** LightGBM (Optimized)
- **PR-AUC:** 0.8734 (+6.9% over baseline)
- **ROC-AUC:** 0.9834
- **Recall:** 86.7%
- **Precision:** 86.7%
- **F1-Score:** 0.8673

---

## 🔧 **Model Decisions**

### **Why LightGBM?**
1. **Performance:** Best PR-AUC (0.8734) among all models
2. **Speed:** Faster training than XGBoost
3. **Memory:** Lower memory footprint
4. **Scalability:** Handles large datasets efficiently

### **Why PR-AUC over Accuracy?**
- Accuracy is misleading for imbalanced data (99.83% by predicting all as legitimate)
- PR-AUC focuses on minority class (fraud) performance
- Better reflects real-world fraud detection needs

### **Threshold Selection**
- **Default (0.5):** Balanced precision and recall
- **Production:** Adjustable based on business needs
  - Lower threshold → Higher recall (catch more frauds)
  - Higher threshold → Higher precision (fewer false alarms)

---

## 🌐 **API Usage**

### **Start API Server**

```bash
# Local development
python run_api.py

# Docker
docker-compose -f api/docker-compose.yml up

# Access API
# http://localhost:8000
# Docs: http://localhost:8000/docs
```

### **API Endpoints**

#### **1. Health Check**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "name": "LightGBM",
    "version": "1.0.0",
    "pr_auc": 0.8734
  }
}
```

#### **2. Single Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    ...
    "Amount": 149.62
  }'
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "confidence": "high",
  "transaction_id": "txn_123",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### **3. Batch Prediction**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"Time": 0, "V1": -1.36, ..., "Amount": 149.62},
    {"Time": 1, "V1": 1.19, ..., "Amount": 2.69}
  ]'
```

#### **4. Metrics (Prometheus)**
```bash
curl http://localhost:8000/metrics
```

### **Python Client Example**

```python
import requests

# Single prediction
transaction = {
    "Time": 0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    # ... other features
    "Amount": 149.62
}

response = requests.post(
    "http://localhost:8000/predict",
    json=transaction
)

result = response.json()
print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.4f}")
```

---

## 📊 **Monitoring Strategy**

### **Metrics Tracked**

**1. Model Performance**
- Prediction latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate

**2. Business Metrics**
- Fraud detection rate
- False positive rate
- Daily transaction volume

**3. System Health**
- CPU/Memory usage
- API uptime
- Response times

### **Monitoring Stack**

```bash
# Start monitoring
docker-compose -f monitoring/docker-compose.yml up

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### **Alerts Configured**

1. **High Error Rate:** > 5% errors in 5 minutes
2. **Slow Response:** p95 latency > 500ms
3. **Model Drift:** Fraud rate deviation > 50%
4. **System Down:** API unavailable for > 1 minute

### **Grafana Dashboards**

- **API Performance:** Request rate, latency, errors
- **Model Metrics:** Predictions, fraud rate, confidence distribution
- **System Health:** CPU, memory, disk usage

---

## 🧪 **Testing**

### **Test Coverage**

```bash
# Run all tests (68 tests)
python run_tests.py

# Run with coverage
python run_tests.py --coverage --html

# Run specific categories
python run_tests.py --unit          # Unit tests (45)
python run_tests.py --integration   # Integration tests (23)
python run_tests.py --api           # API tests
```

### **Test Categories**

**Unit Tests (45 tests):**
- ✅ Data loading (13 tests)
- ✅ Preprocessing (13 tests)
- ✅ Model inference (15 tests)
- ✅ Pipeline validation (4 tests)

**Integration Tests (23 tests):**
- ✅ API endpoints (23 tests)
- ✅ End-to-end workflows
- ✅ Error handling

**Coverage:** > 80% overall, 100% on critical paths

---

## 📚 **Documentation**

- **[API Documentation](api/README.md)** - FastAPI endpoints and usage
- **[Monitoring Guide](monitoring/README.md)** - Prometheus & Grafana setup
- **[Testing Guide](tests/README.md)** - Test suite documentation
- **[Notebooks](notebooks/README.md)** - EDA and analysis

---

## 📖 **References**

- **Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **FastAPI:** [Official Documentation](https://fastapi.tiangolo.com/)
- **LightGBM:** [Official Documentation](https://lightgbm.readthedocs.io/)
- **Prometheus:** [Official Documentation](https://prometheus.io/docs/)
- **MLflow:** [Official Documentation](https://mlflow.org/docs/latest/index.html)

---

## 📝 **License**

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 👤 **Author**

**AMAL A P**
- GitHub: [amal862001](https://github.com/amal862001)  
- LinkedIn: [Amal A P](https://linkedin.com/in/amal-a-p)

---

*Star ⭐ this repo if you find it helpful!*
