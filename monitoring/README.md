# 📊 Fraud Detection API - Monitoring System

Complete monitoring solution using **Prometheus** and **Grafana** for observing system health.

---

## 🎯 **What We Monitor**

### **1. Request Metrics**
- ✅ **Request Rate** - Requests per second
- ✅ **Request Latency** - p50, p95, p99 percentiles
- ✅ **Active Requests** - Current concurrent requests
- ✅ **Request Count** - Total requests by endpoint and status

### **2. Fraud Prediction Metrics**
- ✅ **Fraud Detection Rate** - Percentage of transactions flagged as fraud
- ✅ **Prediction Count** - Total predictions by type (fraud/legitimate)
- ✅ **Fraud Probability Distribution** - Distribution of fraud scores
- ✅ **Confidence Levels** - Predictions by confidence (low/medium/high)

### **3. Error Metrics**
- ✅ **Error Count** - Total errors by type
- ✅ **Error Rate** - Errors per second
- ✅ **Model Prediction Errors** - Failed predictions

### **4. System Metrics**
- ✅ **CPU Usage** - Container and host CPU
- ✅ **Memory Usage** - Container and host memory
- ✅ **Model Status** - Whether model is loaded
- ✅ **Container Health** - Docker container metrics

---

## 🚀 **Quick Start**

### **1. Start the Monitoring Stack**

```bash
# From monitoring directory
cd monitoring
docker-compose up -d
```

This starts:
- **Fraud Detection API** on port 8000
- **Prometheus** on port 9090
- **Grafana** on port 3000
- **Node Exporter** on port 9100
- **cAdvisor** on port 8080
- **Alertmanager** on port 9093

### **2. Access the Dashboards**

**Grafana:**
- URL: http://localhost:3000
- Username: `admin`
- Password: `admin`

**Prometheus:**
- URL: http://localhost:9090

**API Metrics:**
- URL: http://localhost:8000/metrics

### **3. Generate Test Traffic**

```bash
# Generate traffic for 5 minutes at 2 req/s
python monitoring/generate_traffic.py --duration 300 --rate 2.0
```

---

## 📊 **Grafana Dashboards**

### **Pre-configured Dashboard: "Fraud Detection API - Overview"**

**Panels:**
1. **Request Rate** - Real-time request rate gauge
2. **Request Latency** - p50 and p95 latency over time
3. **Fraud Detection Rate** - Percentage of fraud detected
4. **Prediction Count** - Fraud vs Legitimate predictions
5. **Error Rate** - API errors over time
6. **Active Requests** - Current concurrent requests
7. **Model Status** - Whether model is loaded
8. **Memory Usage** - API memory consumption

**Access:** http://localhost:3000/d/fraud-detection-overview

---

## 🔔 **Alerts**

### **Configured Alerts:**

#### **Critical Alerts:**
- **APIDown** - API is unreachable for > 1 minute
- **ModelPredictionErrors** - Model failing to make predictions

#### **Warning Alerts:**
- **HighErrorRate** - Error rate > 5% for 5 minutes
- **HighLatency** - p95 latency > 1 second for 5 minutes
- **HighMemoryUsage** - Memory usage > 1GB for 5 minutes

#### **Info Alerts:**
- **HighFraudRate** - Fraud rate > 10% for 10 minutes
- **LowRequestRate** - Request rate < 0.1 req/s for 10 minutes

### **View Alerts:**
- **Prometheus Alerts:** http://localhost:9090/alerts
- **Alertmanager:** http://localhost:9093

---

## 📈 **Prometheus Queries**

### **Request Metrics:**

```promql
# Request rate
rate(api_requests_total[5m])

# Request latency (p95)
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Active requests
api_active_requests
```

### **Prediction Metrics:**

```promql
# Fraud detection rate
rate(fraud_predictions_total{prediction="fraud"}[5m]) / rate(fraud_predictions_total[5m])

# Total predictions
sum(rate(fraud_predictions_total[5m]))

# Fraud probability distribution
histogram_quantile(0.95, rate(fraud_probability_score_bucket[5m]))
```

### **Error Metrics:**

```promql
# Error rate
rate(api_errors_total[5m])

# Model errors
rate(model_prediction_errors_total[5m])
```

### **System Metrics:**

```promql
# Memory usage (MB)
process_resident_memory_bytes / 1024 / 1024

# Model loaded status
model_loaded
```

---

## 🛠️ **Configuration Files**

### **Prometheus Configuration** (`prometheus.yml`)
- Scrape intervals
- Target endpoints
- Alert rules

### **Alert Rules** (`alerts.yml`)
- Alert definitions
- Thresholds
- Severity levels

### **Alertmanager Configuration** (`alertmanager.yml`)
- Alert routing
- Notification channels
- Inhibition rules

### **Grafana Provisioning**
- `grafana/provisioning/datasources/` - Prometheus datasource
- `grafana/provisioning/dashboards/` - Dashboard configuration
- `grafana/dashboards/` - Dashboard JSON files

---

## 🧪 **Testing the Monitoring**

### **1. Check Metrics Endpoint**

```bash
curl http://localhost:8000/metrics
```

You should see Prometheus metrics like:
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 42.0

# HELP api_request_duration_seconds API request latency in seconds
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="0.1"} 40.0
```

### **2. Generate Traffic**

```bash
# Short test (1 minute, 5 req/s)
python monitoring/generate_traffic.py --duration 60 --rate 5.0

# Long test (10 minutes, 2 req/s)
python monitoring/generate_traffic.py --duration 600 --rate 2.0
```

### **3. View Metrics in Prometheus**

1. Go to http://localhost:9090
2. Enter a query (e.g., `rate(api_requests_total[5m])`)
3. Click "Execute"
4. View graph

### **4. View Dashboard in Grafana**

1. Go to http://localhost:3000
2. Login (admin/admin)
3. Navigate to "Fraud Detection API - Overview" dashboard
4. Watch metrics update in real-time

---

## 📊 **Metrics Reference**

### **Request Metrics:**
- `api_requests_total` - Counter of total requests
- `api_request_duration_seconds` - Histogram of request latency
- `api_active_requests` - Gauge of active requests

### **Prediction Metrics:**
- `fraud_predictions_total` - Counter of predictions
- `fraud_probability_score` - Histogram of fraud probabilities

### **Error Metrics:**
- `api_errors_total` - Counter of API errors
- `model_prediction_errors_total` - Counter of model errors

### **System Metrics:**
- `model_loaded` - Gauge (1 if loaded, 0 if not)
- `process_resident_memory_bytes` - Memory usage

---

## 🔧 **Customization**

### **Add Custom Metrics**

Edit `api/main.py`:

```python
from prometheus_client import Counter

# Define metric
CUSTOM_METRIC = Counter('custom_metric_total', 'Description')

# Increment metric
CUSTOM_METRIC.inc()
```

### **Add Custom Alerts**

Edit `monitoring/alerts.yml`:

```yaml
- alert: CustomAlert
  expr: custom_metric_total > 100
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom alert triggered"
```

### **Create Custom Dashboard**

1. Go to Grafana
2. Click "+" → "Dashboard"
3. Add panels with Prometheus queries
4. Save dashboard
5. Export JSON to `monitoring/grafana/dashboards/`

---

## 🐳 **Docker Commands**

```bash
# Start monitoring stack
docker-compose up -d

# View logs
docker-compose logs -f

# Stop monitoring stack
docker-compose down

# Restart specific service
docker-compose restart prometheus

# View service status
docker-compose ps
```

---

## 🎤 **For Interviews**

**Key Talking Points:**

1. **Comprehensive Monitoring:**
   - "Implemented full monitoring stack with Prometheus and Grafana"
   - "Track request latency, fraud detection rate, and error count"
   - "Real-time dashboards for system health"

2. **Production Metrics:**
   - "Monitor p95 latency to ensure SLA compliance"
   - "Track fraud detection rate to identify anomalies"
   - "Alert on high error rates and API downtime"

3. **Observability:**
   - "Prometheus for metrics collection"
   - "Grafana for visualization"
   - "Alertmanager for notifications"
   - "cAdvisor for container metrics"

4. **Best Practices:**
   - "Metrics exposed via /metrics endpoint"
   - "Histogram for latency percentiles"
   - "Counter for request and error tracking"
   - "Gauge for current state (active requests, model status)"

---

## ✅ **Summary**

**Monitoring Stack Includes:**
- ✅ Prometheus for metrics collection
- ✅ Grafana for visualization
- ✅ Pre-configured dashboards
- ✅ Alert rules for critical issues
- ✅ Traffic generator for testing
- ✅ Complete documentation

**Perfect for:**
- ✅ Production deployment
- ✅ Performance monitoring
- ✅ Incident detection
- ✅ Portfolio demonstration
- ✅ Interview discussions

---

**Your monitoring system is ready!** 🎉📊🚀

