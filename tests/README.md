# 🧪 Testing & Validation

Comprehensive test suite for the Fraud Detection System.

---

## 📋 **Overview**

This test suite provides **automated test coverage** for:
- ✅ **Data Loading** - Tests for data loading and validation
- ✅ **Preprocessing** - Tests for data preprocessing and transformation
- ✅ **Model Inference** - Tests for model loading and predictions
- ✅ **API Endpoints** - Integration tests for FastAPI endpoints

---

## 📁 **Test Structure**

```
tests/
├── __init__.py                    # Tests package
├── conftest.py                    # Pytest fixtures and configuration
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Testing dependencies
│
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── test_load_data.py         # Data loading tests
│   ├── test_preprocess.py        # Preprocessing tests
│   └── test_model_inference.py   # Model inference tests
│
├── integration/                   # Integration tests
│   ├── __init__.py
│   └── test_api_endpoints.py     # API endpoint tests
│
└── fixtures/                      # Test fixtures and sample data
    ├── __init__.py
    └── sample_data.py            # Sample transactions for testing
```

---

## 🚀 **Quick Start**

### **1. Install Testing Dependencies**

```bash
# Install test requirements
pip install -r tests/requirements.txt
```

### **2. Run All Tests**

```bash
# Using Python script
python run_tests.py

# Using pytest directly
pytest tests/

# Using batch/shell script
run_tests.bat          # Windows
./run_tests.sh         # Linux/Mac
```

### **3. Run Specific Test Types**

```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only
python run_tests.py --integration

# API tests only
python run_tests.py --api

# Fast tests (skip slow tests)
python run_tests.py --fast
```

### **4. Run with Coverage**

```bash
# Coverage report in terminal
python run_tests.py --coverage

# Coverage report with HTML
python run_tests.py --coverage --html

# View HTML report
# Open: htmlcov/index.html
```

---

## 📊 **Test Categories**

### **Unit Tests** (`tests/unit/`)

**Data Loading Tests** (`test_load_data.py`):
- ✅ Test `get_data_path()` function
- ✅ Test `load_raw_data()` function
- ✅ Test `get_data_info()` function
- ✅ Test `split_features_target()` function
- ✅ Validate data structure and columns

**Preprocessing Tests** (`test_preprocess.py`):
- ✅ Test `FraudDataPreprocessor` class
- ✅ Test scaler initialization (standard, robust)
- ✅ Test outlier detection (IQR, Z-score)
- ✅ Test feature scaling
- ✅ Test fit_transform and transform methods
- ✅ Test train-test split with stratification
- ✅ Test full preprocessing pipeline

**Model Inference Tests** (`test_model_inference.py`):
- ✅ Test model artifacts exist
- ✅ Test model loading
- ✅ Test model has required methods (predict, predict_proba)
- ✅ Test scaler transformation
- ✅ Test single transaction prediction
- ✅ Test batch prediction
- ✅ Test fraud probability extraction
- ✅ Test threshold-based classification

### **Integration Tests** (`tests/integration/`)

**API Endpoint Tests** (`test_api_endpoints.py`):
- ✅ Test root endpoint (`/`)
- ✅ Test health check endpoint (`/health`)
- ✅ Test prediction endpoint (`/predict`)
- ✅ Test batch prediction endpoint (`/predict/batch`)
- ✅ Test metrics endpoint (`/metrics`)
- ✅ Test input validation
- ✅ Test error handling
- ✅ Test response format

---

## 🧪 **Running Tests**

### **Using Python Script** (Recommended)

```bash
# All tests
python run_tests.py

# Unit tests
python run_tests.py --unit

# Integration tests
python run_tests.py --integration

# API tests
python run_tests.py --api

# With coverage
python run_tests.py --coverage

# With HTML coverage report
python run_tests.py --coverage --html

# Verbose output
python run_tests.py --verbose

# Fast mode (skip slow tests)
python run_tests.py --fast

# Parallel execution
python run_tests.py --parallel
```

### **Using Pytest Directly**

```bash
# All tests
pytest tests/

# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Specific test file
pytest tests/unit/test_load_data.py

# Specific test class
pytest tests/unit/test_load_data.py::TestGetDataPath

# Specific test function
pytest tests/unit/test_load_data.py::TestGetDataPath::test_get_data_path_default

# With coverage
pytest tests/ --cov=src --cov=api --cov-report=html

# With markers
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m api           # Only API tests
pytest -m "not slow"    # Skip slow tests

# Verbose output
pytest tests/ -vv

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l
```

### **Using Batch/Shell Scripts**

```bash
# Windows
run_tests.bat              # All tests
run_tests.bat unit         # Unit tests
run_tests.bat integration  # Integration tests
run_tests.bat coverage     # With coverage

# Linux/Mac
chmod +x run_tests.sh      # Make executable (first time only)
./run_tests.sh             # All tests
./run_tests.sh unit        # Unit tests
./run_tests.sh integration # Integration tests
./run_tests.sh coverage    # With coverage
```

---

## 📈 **Coverage Reports**

### **Generate Coverage Report**

```bash
# Terminal report
python run_tests.py --coverage

# HTML report
python run_tests.py --coverage --html
```

### **View HTML Coverage Report**

```bash
# Report is generated in htmlcov/
# Open in browser:
htmlcov/index.html
```

### **Coverage Metrics**

The test suite aims for:
- **Overall Coverage**: > 80%
- **Critical Paths**: > 90%
- **API Endpoints**: 100%

---

## 🎯 **Test Fixtures**

### **Available Fixtures** (in `conftest.py`)

```python
# Sample data fixtures
sample_legitimate_transaction  # Single legitimate transaction
sample_fraud_transaction       # Single fraudulent transaction
sample_dataframe              # DataFrame with 100 transactions
sample_small_dataframe        # DataFrame with 20 transactions
sample_batch_transactions     # Batch of 5 transactions
sample_features_target        # Features and target split

# Path fixtures
artifacts_dir                 # Path to artifacts directory
models_dir                    # Path to models directory
data_dir                      # Path to data directory

# Model fixtures
model_artifacts              # Loaded model, scaler, threshold

# API fixtures
api_client                   # FastAPI TestClient
```

### **Using Fixtures in Tests**

```python
def test_example(sample_legitimate_transaction, model_artifacts):
    """Example test using fixtures."""
    model, scaler, threshold = model_artifacts
    
    # Use the fixtures
    trans_df = pd.DataFrame([sample_legitimate_transaction])
    trans_scaled = scaler.transform(trans_df)
    prediction = model.predict(trans_scaled)
    
    assert prediction is not None
```

---

## 🏷️ **Test Markers**

Tests are marked with pytest markers for selective execution:

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.api           # API test
@pytest.mark.slow          # Slow test (can be skipped)
```

### **Run Tests by Marker**

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Only API tests
pytest -m api

# Skip slow tests
pytest -m "not slow"
```

---

## ✅ **Test Checklist**

### **Before Running Tests**

- [ ] Install testing dependencies: `pip install -r tests/requirements.txt`
- [ ] Ensure model artifacts exist: `artifacts/fraud_model.pkl`, `scaler.pkl`, `threshold.txt`
- [ ] (Optional) Have dataset available: `data/raw/creditcard.csv`

### **What Gets Tested**

**Data Loading:**
- [ ] Data path resolution
- [ ] CSV loading
- [ ] Column validation
- [ ] Data info extraction
- [ ] Features/target splitting

**Preprocessing:**
- [ ] Scaler initialization
- [ ] Outlier detection
- [ ] Feature scaling
- [ ] Train-test splitting
- [ ] Full pipeline

**Model Inference:**
- [ ] Model loading
- [ ] Scaler loading
- [ ] Single prediction
- [ ] Batch prediction
- [ ] Probability extraction
- [ ] Threshold application

**API Endpoints:**
- [ ] Root endpoint
- [ ] Health check
- [ ] Single prediction
- [ ] Batch prediction
- [ ] Metrics endpoint
- [ ] Input validation
- [ ] Error handling

---

## 🐛 **Troubleshooting**

### **Tests Fail: Model Artifacts Not Found**

```bash
# Solution: Run model serialization
python src/models/serialize_model.py
```

### **Tests Fail: Dataset Not Found**

```bash
# Solution: Download dataset or skip tests that require it
pytest -m "not requires_data"
```

### **Import Errors**

```bash
# Solution: Install dependencies
pip install -r requirements.txt
pip install -r tests/requirements.txt
```

### **API Tests Fail**

```bash
# Solution: Ensure FastAPI dependencies are installed
pip install fastapi[all]
pip install httpx
```

---

## 📚 **Additional Resources**

- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py Documentation**: https://coverage.readthedocs.io/
- **FastAPI Testing**: https://fastapi.tiangolo.com/tutorial/testing/

---

## ✨ **Summary**

**Test Suite Includes:**
- ✅ 50+ automated tests
- ✅ Unit tests for all core components
- ✅ Integration tests for API endpoints
- ✅ Test fixtures for reusable data
- ✅ Coverage reporting
- ✅ Multiple execution options
- ✅ Comprehensive documentation

**Perfect for:**
- ✅ Continuous Integration (CI)
- ✅ Pre-deployment validation
- ✅ Regression testing
- ✅ Code quality assurance
- ✅ Portfolio demonstration

---

**Your testing suite is complete and ready to use!** 🧪✅🚀

