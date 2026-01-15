# 🏗️ System Architecture

Comprehensive architecture documentation for the Fraud Detection System.

---

## 📋 **Table of Contents**

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Pipeline](#model-pipeline)
5. [API Architecture](#api-architecture)
6. [Monitoring Architecture](#monitoring-architecture)
7. [Deployment Architecture](#deployment-architecture)
8. [Design Decisions](#design-decisions)

---

## 🎯 **System Overview**

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRAUD DETECTION SYSTEM                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│                  │         │                  │         │                  │
│   Data Layer     │────────▶│  Model Layer     │────────▶│  Serving Layer   │
│                  │         │                  │         │                  │
└──────────────────┘         └──────────────────┘         └──────────────────┘
        │                            │                            │
        │                            │                            │
        ▼                            ▼                            ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  • Data Loading  │         │  • Training      │         │  • FastAPI       │
│  • Preprocessing │         │  • Optimization  │         │  • REST API      │
│  • Validation    │         │  • Evaluation    │         │  • Batch Predict │
└──────────────────┘         │  • Selection     │         └──────────────────┘
                             └──────────────────┘                 │
                                     │                            │
                                     ▼                            ▼
                             ┌──────────────────┐         ┌──────────────────┐
                             │  • MLflow        │         │  • Prometheus    │
                             │  • Experiment    │         │  • Grafana       │
                             │    Tracking      │         │  • Alerting      │
                             └──────────────────┘         └──────────────────┘
```

### **Technology Stack**

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Data** | Pandas, NumPy | Data manipulation |
| **Preprocessing** | Scikit-learn | Feature scaling, outlier detection |
| **Models** | LightGBM, XGBoost, Scikit-learn | ML algorithms |
| **Optimization** | Optuna | Hyperparameter tuning |
| **Tracking** | MLflow | Experiment tracking |
| **API** | FastAPI, Uvicorn | REST API serving |
| **Monitoring** | Prometheus, Grafana | Metrics & dashboards |
| **Testing** | Pytest | Automated testing |
| **Containerization** | Docker, Docker Compose | Deployment |

---

## 🧩 **Component Architecture**

### **1. Data Layer**

```
data/
├── raw/                    # Raw dataset
│   └── creditcard.csv     # 284,807 transactions
│
└── processed/             # Processed data
    ├── train.csv          # Training set (80%)
    └── test.csv           # Test set (20%)
```

**Components:**
- **`load_data.py`** - Data loading utilities
  - `get_data_path()` - Path resolution
  - `load_raw_data()` - CSV loading
  - `get_data_info()` - Data statistics
  - `split_features_target()` - Feature/target separation

- **`preprocess.py`** - Preprocessing pipeline
  - `FraudDataPreprocessor` - Main preprocessing class
  - Outlier detection (IQR, Z-score)
  - Feature scaling (Standard, Robust)
  - Train-test splitting with stratification

**Data Flow:**
```
Raw CSV → Load → Validate → Preprocess → Split → Train/Test
```

---

### **2. Model Layer**

```
src/models/
├── train.py               # Baseline models
├── handle_imbalance.py    # Imbalance handling
├── advanced_models.py     # XGBoost & LightGBM
├── optimize.py            # Hyperparameter optimization
├── final_selection.py     # Model selection
└── serialize_model.py     # Model serialization
```

**Model Pipeline:**

```
┌─────────────┐
│  Baseline   │
│  Models     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Imbalance  │
│  Handling   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Advanced   │
│  Models     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Optimization│
│  (Optuna)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Model     │
│  Selection  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Serialization│
│  (Pickle)   │
└─────────────┘
```

**Model Storage:**

```
models/
├── baseline/              # Baseline models
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── advanced/              # Advanced models
│   ├── xgboost.pkl
│   └── lightgbm.pkl
│
├── optimized/             # Optimized models
│   └── lightgbm_optimized.pkl
│
└── production/            # Production model
    ├── fraud_model.pkl
    ├── scaler.pkl
    └── metadata.json
```

---

### **3. Tracking Layer**

```
src/tracking/
└── mlflow_utils.py        # MLflow utilities
```

**MLflow Architecture:**

```
┌──────────────────────────────────────────┐
│           MLflow Tracking Server          │
└──────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│Parameters│  │ Metrics  │  │Artifacts │
└──────────┘  └──────────┘  └──────────┘
│            │            │
│ • n_est   │ • PR-AUC   │ • Models   │
│ • lr      │ • Recall   │ • Plots    │
│ • depth   │ • Precision│ • Configs  │
└──────────┘  └──────────┘  └──────────┘
```

**Tracked Information:**
- **Parameters:** Model hyperparameters
- **Metrics:** PR-AUC, ROC-AUC, Recall, Precision, F1
- **Artifacts:** Trained models, plots, confusion matrices
- **Tags:** Model type, experiment name, version

---

## 🌐 **API Architecture**

### **FastAPI Application Structure**

```
api/
├── main.py                # Main application
├── routers/               # API routers (future)
├── Dockerfile             # Container definition
├── docker-compose.yml     # Orchestration
└── requirements.txt       # Dependencies
```

**API Endpoints:**

```
┌─────────────────────────────────────────┐
│         FastAPI Application              │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│   GET    │  │   POST   │  │   GET    │
│    /     │  │ /predict │  │ /health  │
└──────────┘  └──────────┘  └──────────┘
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│   POST   │  │   GET    │  │          │
│/predict/ │  │ /metrics │  │          │
│  batch   │  │          │  │          │
└──────────┘  └──────────┘  └──────────┘
```

**Request/Response Flow:**

```
Client Request
      │
      ▼
┌─────────────┐
│  FastAPI    │
│  Endpoint   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Validate   │
│  Input      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Load       │
│  Model      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Preprocess │
│  Features   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Predict    │
│  (Model)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Format     │
│  Response   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Update     │
│  Metrics    │
└──────┬──────┘
       │
       ▼
Client Response
```

---

## 📊 **Monitoring Architecture**

### **Monitoring Stack**

```
monitoring/
├── prometheus.yml         # Prometheus config
├── alertmanager.yml       # Alert rules
├── alerts.yml             # Alert definitions
├── grafana/               # Grafana dashboards
│   └── dashboards/
│       ├── api_performance.json
│       └── model_metrics.json
└── docker-compose.yml     # Stack orchestration
```

**Monitoring Flow:**

```
┌──────────────┐
│   FastAPI    │
│   /metrics   │
└──────┬───────┘
       │
       │ (scrape every 15s)
       ▼
┌──────────────┐
│  Prometheus  │
│   (Storage)  │
└──────┬───────┘
       │
       │ (query)
       ▼
┌──────────────┐
│   Grafana    │
│ (Dashboards) │
└──────────────┘
       │
       │ (alerts)
       ▼
┌──────────────┐
│ Alertmanager │
│ (Notifications)│
└──────────────┘
```

**Metrics Collected:**

1. **API Metrics:**
   - `api_requests_total` - Total requests
   - `api_request_duration_seconds` - Request latency
   - `api_errors_total` - Error count

2. **Model Metrics:**
   - `predictions_total` - Total predictions
   - `fraud_predictions_total` - Fraud predictions
   - `prediction_confidence` - Confidence distribution

3. **System Metrics:**
   - CPU usage
   - Memory usage
   - Disk I/O

---

## 🐳 **Deployment Architecture**

### **Docker Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Host                           │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   API        │  │  Prometheus  │  │   Grafana    │
│  Container   │  │  Container   │  │  Container   │
└──────────────┘  └──────────────┘  └──────────────┘
│              │  │              │  │              │
│ Port: 8000   │  │ Port: 9090   │  │ Port: 3000   │
│              │  │              │  │              │
│ • FastAPI    │  │ • Metrics    │  │ • Dashboards │
│ • Model      │  │ • Storage    │  │ • Alerts     │
│ • Artifacts  │  │ • Scraping   │  │ • Queries    │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Container Specifications:**

| Container | Base Image | Ports | Volumes |
|-----------|------------|-------|---------|
| API | `python:3.10-slim` | 8000 | `./artifacts`, `./models` |
| Prometheus | `prom/prometheus` | 9090 | `./prometheus.yml` |
| Grafana | `grafana/grafana` | 3000 | `./grafana/dashboards` |

---

## 🎯 **Design Decisions**

### **1. Model Selection: LightGBM**

**Why LightGBM over XGBoost?**

| Aspect | LightGBM | XGBoost | Decision |
|--------|----------|---------|----------|
| **Performance** | PR-AUC: 0.8734 | PR-AUC: 0.8567 | ✅ LightGBM |
| **Training Speed** | ~30s | ~45s | ✅ LightGBM |
| **Memory** | Lower | Higher | ✅ LightGBM |
| **Scalability** | Better | Good | ✅ LightGBM |

**Conclusion:** LightGBM offers best balance of performance, speed, and resource efficiency.

---

### **2. Metric Selection: PR-AUC**

**Why PR-AUC over Accuracy?**

| Metric | Value | Issue |
|--------|-------|-------|
| **Accuracy** | 99.83% | Misleading (predict all as legitimate) |
| **ROC-AUC** | 0.9834 | Optimistic for imbalanced data |
| **PR-AUC** | 0.8734 | Focuses on minority class (fraud) |

**Conclusion:** PR-AUC is the most appropriate metric for highly imbalanced fraud detection.

---

### **3. Imbalance Handling: Threshold Tuning**

**Comparison of Techniques:**

| Technique | Pros | Cons | Selected |
|-----------|------|------|----------|
| **Class Weights** | Simple | Limited improvement | ❌ |
| **SMOTE** | Synthetic samples | Overfitting risk | ❌ |
| **Threshold Tuning** | Flexible, production-ready | Requires calibration | ✅ |

**Conclusion:** Threshold tuning offers most flexibility for production deployment.

---

### **4. API Framework: FastAPI**

**Why FastAPI over Flask/Django?**

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| **Performance** | High (async) | Medium | Medium |
| **Auto Docs** | ✅ (Swagger) | ❌ | ❌ |
| **Type Validation** | ✅ (Pydantic) | ❌ | ❌ |
| **Async Support** | ✅ Native | ❌ | ✅ Limited |
| **Learning Curve** | Easy | Easy | Steep |

**Conclusion:** FastAPI provides best developer experience and performance.

---

### **5. Monitoring: Prometheus + Grafana**

**Why Prometheus over alternatives?**

| Feature | Prometheus | CloudWatch | Datadog |
|---------|------------|------------|---------|
| **Cost** | Free | Paid | Paid |
| **Flexibility** | High | Medium | High |
| **Self-hosted** | ✅ | ❌ | ❌ |
| **Integration** | Excellent | AWS-focused | Excellent |

**Conclusion:** Prometheus offers best balance of features and cost for self-hosted deployment.

---

### **6. Experiment Tracking: MLflow**

**Why MLflow?**

✅ **Open Source:** Free and self-hosted
✅ **Framework Agnostic:** Works with any ML library
✅ **Complete:** Tracking, registry, deployment
✅ **UI:** Built-in web interface
✅ **Integration:** Easy integration with existing code

---

### **7. Testing Strategy: Pytest**

**Test Coverage Strategy:**

```
┌─────────────────────────────────────┐
│         Test Pyramid                 │
└─────────────────────────────────────┘

              ┌──────┐
              │  E2E │  (Future)
              └──────┘
           ┌────────────┐
           │Integration │  (23 tests)
           │   Tests    │
           └────────────┘
        ┌──────────────────┐
        │   Unit Tests     │  (45 tests)
        │                  │
        └──────────────────┘
```

**Coverage Targets:**
- **Overall:** > 80%
- **Critical Paths:** > 90%
- **API Endpoints:** 100%

---

### **8. Containerization: Docker**

**Why Docker?**

✅ **Reproducibility:** Same environment everywhere
✅ **Isolation:** Dependencies don't conflict
✅ **Portability:** Deploy anywhere
✅ **Scalability:** Easy to scale horizontally
✅ **CI/CD:** Integrates with pipelines

---

## 🔄 **Data Flow**

### **Training Pipeline**

```
Raw Data (CSV)
      │
      ▼
┌─────────────┐
│   Load      │
│   Data      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Preprocess  │
│ • Scale     │
│ • Outliers  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Split     │
│ Train/Test  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Train     │
│   Models    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Optimize   │
│  (Optuna)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Evaluate   │
│  & Select   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Serialize   │
│   Model     │
└──────┬──────┘
       │
       ▼
Production Artifacts
```

### **Inference Pipeline**

```
Transaction Data
      │
      ▼
┌─────────────┐
│  Validate   │
│   Input     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Load      │
│  Scaler     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Transform  │
│  Features   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Load      │
│   Model     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Predict    │
│ Probability │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Apply     │
│ Threshold   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Format    │
│  Response   │
└──────┬──────┘
       │
       ▼
Prediction Result
```

---

## 🔐 **Security Considerations**

### **Current Implementation**

✅ **Input Validation:** Pydantic models validate all inputs
✅ **Error Handling:** Graceful error responses
✅ **Logging:** Comprehensive logging for debugging

### **Production Recommendations**

🔒 **Authentication:** Add API key or OAuth2
🔒 **Rate Limiting:** Prevent abuse
🔒 **HTTPS:** Encrypt data in transit
🔒 **Input Sanitization:** Prevent injection attacks
🔒 **Secrets Management:** Use environment variables

---

## 📈 **Scalability Considerations**

### **Current Capacity**

- **Throughput:** ~100 requests/second (single instance)
- **Latency:** p95 < 100ms
- **Memory:** ~500MB per instance

### **Scaling Strategies**

**Horizontal Scaling:**
```
Load Balancer
      │
      ├──────┬──────┬──────┐
      │      │      │      │
      ▼      ▼      ▼      ▼
   API-1  API-2  API-3  API-N
```

**Vertical Scaling:**
- Increase CPU/Memory per instance
- Use GPU for faster inference (if needed)

**Caching:**
- Redis for frequent predictions
- Model caching in memory

---

## 🎓 **Key Takeaways**

### **Architecture Principles**

1. **Modularity:** Each component has single responsibility
2. **Scalability:** Designed for horizontal scaling
3. **Observability:** Comprehensive monitoring and logging
4. **Testability:** 68 automated tests with >80% coverage
5. **Reproducibility:** Docker ensures consistent environments

### **Production Readiness**

✅ **API:** FastAPI with auto-docs and validation
✅ **Monitoring:** Prometheus + Grafana
✅ **Testing:** Comprehensive test suite
✅ **Containerization:** Docker-ready
✅ **Documentation:** Complete architecture docs

---

## 📚 **References**

- **FastAPI:** [Official Documentation](https://fastapi.tiangolo.com/)
- **Prometheus:** [Official Documentation](https://prometheus.io/docs/)
- **Grafana:** [Official Documentation](https://grafana.com/docs/)
- **MLflow:** [Official Documentation](https://mlflow.org/docs/latest/index.html)
- **Docker:** [Official Documentation](https://docs.docker.com/)

---

**This architecture demonstrates production-ready ML system design!** 🏗️✅


