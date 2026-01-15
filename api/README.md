# 🚀 FastAPI Fraud Detection Service

Real-time fraud detection API built with FastAPI.

## 📋 Features

- ✅ **REST API** for real-time fraud predictions
- ✅ **/predict endpoint** for single transaction prediction
- ✅ **/predict/batch endpoint** for batch predictions
- ✅ **Input validation** with Pydantic
- ✅ **Model artifacts** loaded at startup
- ✅ **Health check** endpoint
- ✅ **Swagger documentation** (auto-generated)
- ✅ **CORS support** for web applications
- ✅ **Logging** for monitoring

---

## 🛠️ Setup

### **1. Install Dependencies**

```bash
cd api
pip install -r requirements.txt
```

### **2. Ensure Model Artifacts Exist**

The API requires model artifacts in the `artifacts/` directory:

```bash
# From project root
python src/models/serialize_model.py
```

This creates:
- `artifacts/fraud_model.pkl`
- `artifacts/scaler.pkl`
- `artifacts/threshold.txt`

### **3. Start the API**

```bash
# From api directory
uvicorn main:app --reload

# Or from project root
uvicorn api.main:app --reload
```

The API will be available at: **http://localhost:8000**

---

## 📚 API Documentation

### **Interactive Docs**

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### **Endpoints**

#### **1. Root** - `GET /`
Get API information

**Response:**
```json
{
  "message": "Fraud Detection API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

#### **2. Health Check** - `GET /health`
Check API health and model status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "threshold": 0.9,
  "timestamp": "2026-01-15T10:30:00"
}
```

#### **3. Predict** - `POST /predict`
Predict fraud for a single transaction

**Request Body:**
```json
{
  "Time": 0.0,
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
  "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
  "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
  "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
  "V26": -0.19, "V27": 0.13, "V28": -0.02,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0234,
  "threshold": 0.9,
  "confidence": "high",
  "timestamp": "2026-01-15T10:30:00"
}
```

#### **4. Batch Predict** - `POST /predict/batch`
Predict fraud for multiple transactions

**Request Body:** Array of transaction objects

**Response:** Array of prediction objects

---

## 🧪 Testing

### **Run Test Suite**

```bash
# Make sure API is running first
python api/test_api.py
```

### **Manual Testing with cURL**

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

### **Testing with Python**

```python
import requests

# Predict
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "Time": 0.0,
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
        "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
        "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
        "V26": -0.19, "V27": 0.13, "V28": -0.02,
        "Amount": 149.62
    }
)

print(response.json())
```

---

## 🐳 Docker Deployment

### **Build Image**

```bash
docker build -t fraud-detection-api .
```

### **Run Container**

```bash
docker run -p 8000:8000 fraud-detection-api
```

---

## 📊 Model Information

**Model:** XGBoost Classifier  
**Threshold:** 0.9 (optimized for balanced precision-recall)  
**Performance:**
- PR-AUC: 0.8786
- Precision: 94.19%
- Recall: 82.65%
- F1-Score: 88.04%

---

## 🔒 Input Validation

The API validates all inputs:
- ✅ All 30 features required (Time, V1-V28, Amount)
- ✅ Amount must be >= 0
- ✅ All features must be numeric
- ✅ Returns 422 error for invalid inputs

---

## 📝 Response Fields

- **is_fraud:** 0 (legitimate) or 1 (fraud)
- **fraud_probability:** Probability score (0.0 to 1.0)
- **threshold:** Classification threshold used
- **confidence:** Confidence level (low/medium/high)
- **timestamp:** Prediction timestamp (ISO format)

---

## 🚀 Production Deployment

### **Environment Variables**

```bash
export MODEL_PATH=artifacts/fraud_model.pkl
export SCALER_PATH=artifacts/scaler.pkl
export THRESHOLD_PATH=artifacts/threshold.txt
export API_HOST=0.0.0.0
export API_PORT=8000
```

### **Run in Production**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📈 Monitoring

The API logs all predictions:

```
INFO: Prediction: is_fraud=0, probability=0.0234, amount=149.62
```

Monitor logs for:
- Prediction patterns
- High fraud probability transactions
- API errors
- Performance metrics

---

## ✅ Summary

**API Features:**
- ✅ Real-time fraud detection
- ✅ Single and batch predictions
- ✅ Input validation
- ✅ Auto-generated documentation
- ✅ Health monitoring
- ✅ Production-ready

**Perfect for:**
- Production deployment
- Integration with web apps
- Real-time fraud detection
- Portfolio demonstration
- Interview discussions

---

**Your FastAPI fraud detection service is ready!** 🎉

