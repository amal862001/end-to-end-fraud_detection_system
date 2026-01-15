# 📊 Monitoring Quick Reference

## 🚀 Quick Start

```bash
# Start monitoring stack
cd monitoring
docker-compose up -d

# Generate test traffic
python monitoring/generate_traffic.py --duration 300 --rate 2.0

# View logs
docker-compose logs -f

# Stop monitoring stack
docker-compose down
```

---

## 🌐 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3000 | admin/admin |
| **Prometheus** | http://localhost:9090 | - |
| **API** | http://localhost:8000 | - |
| **API Metrics** | http://localhost:8000/metrics | - |
| **API Docs** | http://localhost:8000/docs | - |
| **cAdvisor** | http://localhost:8080 | - |
| **Alertmanager** | http://localhost:9093 | - |

---

## 📊 Key Metrics

### Request Metrics
```promql
# Request rate
rate(api_requests_total[5m])

# Request latency (p95)
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Active requests
api_active_requests
```

### Prediction Metrics
```promql
# Fraud detection rate
rate(fraud_predictions_total{prediction="fraud"}[5m]) / rate(fraud_predictions_total[5m])

# Total predictions
sum(rate(fraud_predictions_total[5m]))
```

### Error Metrics
```promql
# Error rate
rate(api_errors_total[5m])

# Model errors
rate(model_prediction_errors_total[5m])
```

---

## 🔔 Alerts

### Critical
- **APIDown** - API unreachable > 1 min
- **ModelPredictionErrors** - Model failing

### Warning
- **HighErrorRate** - Error rate > 5%
- **HighLatency** - p95 > 1 second
- **HighMemoryUsage** - Memory > 1GB

### Info
- **HighFraudRate** - Fraud rate > 10%
- **LowRequestRate** - Rate < 0.1 req/s

---

## 🐳 Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f [service]

# Status
docker-compose ps

# Restart
docker-compose restart [service]
```

---

## 🧪 Testing

```bash
# Generate traffic (5 min, 2 req/s)
python monitoring/generate_traffic.py --duration 300 --rate 2.0

# Check metrics endpoint
curl http://localhost:8000/metrics

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_transaction.json
```

---

## 📈 Grafana Dashboard

**Dashboard:** "Fraud Detection API - Overview"

**Panels:**
1. Request Rate
2. Request Latency (p50, p95)
3. Fraud Detection Rate
4. Prediction Count
5. Error Rate
6. Active Requests
7. Model Status
8. Memory Usage

**Access:** http://localhost:3000/d/fraud-detection-overview

---

## 🔧 Troubleshooting

### Services not starting
```bash
# Check Docker
docker --version
docker-compose --version

# Check logs
docker-compose logs [service]

# Restart service
docker-compose restart [service]
```

### Metrics not showing
```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
# Go to: http://localhost:9090/targets

# Restart Prometheus
docker-compose restart prometheus
```

### Dashboard not loading
```bash
# Check Grafana logs
docker-compose logs grafana

# Restart Grafana
docker-compose restart grafana

# Re-provision datasource
docker-compose restart grafana
```

---

## 📝 Files

```
monitoring/
├── prometheus.yml          # Prometheus config
├── alerts.yml             # Alert rules
├── alertmanager.yml       # Alertmanager config
├── docker-compose.yml     # Stack orchestration
├── generate_traffic.py    # Traffic generator
├── start_monitoring.py    # Startup script
└── grafana/
    ├── provisioning/      # Auto-provisioning
    └── dashboards/        # Dashboard JSON
```

---

## 🎯 For Interviews

**Key Points:**
- ✅ Full observability with Prometheus + Grafana
- ✅ Monitor request latency, fraud rate, errors
- ✅ Real-time dashboards and alerts
- ✅ Production-ready configuration
- ✅ Docker-based deployment

**Metrics:**
- Request rate, latency (p50, p95, p99)
- Fraud detection rate
- Error rate
- System metrics (CPU, memory)

**Tech Stack:**
- Prometheus (metrics)
- Grafana (visualization)
- Alertmanager (alerts)
- cAdvisor (containers)
- Node Exporter (system)

---

**Quick Reference Complete!** 📊✅

