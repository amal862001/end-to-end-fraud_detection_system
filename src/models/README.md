# 🤖 Model Training

This directory contains scripts for training fraud detection models.

## 📁 Files

- **`train.py`** - Baseline model training script
- **`__init__.py`** - Package initialization

---

## 🎯 train.py - Baseline Models

**Purpose:** Train and evaluate baseline models for fraud detection.

### Models Trained:

1. **Logistic Regression**
   - Simple, interpretable baseline
   - Uses class weights to handle imbalance
   - Fast training

2. **Random Forest**
   - Tree-based ensemble model
   - Handles non-linear patterns
   - Feature importance available

### Evaluation Metrics:

- **PR-AUC** (Precision-Recall AUC) - Best for imbalanced data
- **ROC-AUC** (ROC curve AUC)
- **Precision** - How many predicted frauds are actual frauds
- **Recall** - How many actual frauds are detected
- **F1-Score** - Harmonic mean of Precision and Recall
- **Confusion Matrix** - TP, FP, TN, FN

---

## 🚀 How to Run

### Option 1: Run from command line

```bash
# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Run training
python src/models/train.py
```

### Option 2: Import in Python

```python
from src.models.train import main

# Train all models
models, results, comparison = main()
```

### Option 3: Train individual models

```python
from src.models.train import prepare_data, train_logistic_regression, train_random_forest

# Prepare data
X_train, X_test, y_train, y_test, scaler = prepare_data()

# Train Logistic Regression
lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)

# Train Random Forest
rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
```

---

## 📊 Expected Output

```
============================================================
BASELINE MODEL TRAINING
============================================================
Loading data...
Train set: 227845 samples
Test set: 56962 samples
Fraud rate - Train: 0.17%, Test: 0.17%

============================================================
TRAINING LOGISTIC REGRESSION
============================================================
Training...

Results:
  PR-AUC:    0.7xxx
  ROC-AUC:   0.9xxx
  Precision: 0.8xxx
  Recall:    0.6xxx
  F1-Score:  0.7xxx

Confusion Matrix:
  TP:   XX  |  FP:   XX
  FN:   XX  |  TN: XXXXX

============================================================
TRAINING RANDOM FOREST
============================================================
Training...

Results:
  PR-AUC:    0.8xxx
  ROC-AUC:   0.9xxx
  Precision: 0.9xxx
  Recall:    0.7xxx
  F1-Score:  0.8xxx

Confusion Matrix:
  TP:   XX  |  FP:   XX
  FN:   XX  |  TN: XXXXX

============================================================
MODEL COMPARISON
============================================================

                      PR-AUC  ROC-AUC  Precision  Recall  F1-Score
Logistic Regression   0.7xxx   0.9xxx     0.8xxx  0.6xxx    0.7xxx
Random Forest         0.8xxx   0.9xxx     0.9xxx  0.7xxx    0.8xxx

✓ Best Model (by PR-AUC): Random Forest

============================================================
SAVING MODELS
============================================================
✓ Saved Logistic Regression to models/baseline/logistic_regression.pkl
✓ Saved Random Forest to models/baseline/random_forest.pkl
✓ Saved scaler to models/baseline/scaler.pkl

============================================================
✓ TRAINING COMPLETE!
============================================================
```

---

## 💾 Output Files

**Location:** `models/baseline/`

1. **`logistic_regression.pkl`** - Trained Logistic Regression model
2. **`random_forest.pkl`** - Trained Random Forest model
3. **`scaler.pkl`** - StandardScaler for feature scaling

---

## ⏱️ Runtime

- **Data loading:** ~30 seconds
- **Logistic Regression:** ~10 seconds
- **Random Forest:** ~1-2 minutes
- **Total:** ~3 minutes

---

## 🎓 Key Features for Freshers

### ✅ **Clean Code**
- Simple, readable functions
- Clear variable names
- Good documentation

### ✅ **Best Practices**
- Train/test split with stratification
- Feature scaling
- Class weight balancing
- Proper evaluation metrics

### ✅ **Professional Output**
- Formatted results
- Model comparison
- Saved models for deployment

---

## 📈 Understanding the Metrics

### **PR-AUC (Precision-Recall AUC)**
- **Best metric for imbalanced data**
- Ranges from 0 to 1 (higher is better)
- Focuses on positive class (fraud)

### **Precision**
- Of all predicted frauds, how many are correct?
- High precision = fewer false alarms

### **Recall**
- Of all actual frauds, how many did we catch?
- High recall = fewer missed frauds

### **F1-Score**
- Balance between Precision and Recall
- Good overall metric

---

## 🔄 Next Steps

After running baseline models:

1. **Analyze Results**
   - Which model performs better?
   - What's the precision-recall tradeoff?
   - Are we catching enough frauds?

2. **Improve Models**
   - Try SMOTE for oversampling
   - Tune hyperparameters
   - Try XGBoost/LightGBM

3. **Deploy Best Model**
   - Use saved .pkl files
   - Create prediction API
   - Monitor performance

---

Perfect for showcasing in your portfolio! 🎯

---

## 🎯 handle_imbalance.py - Class Imbalance Handling

**Purpose:** Improve fraud recall through various imbalance handling techniques.

### Techniques Implemented:

1. **Class Weights**
   - Test different weight strategies
   - Balanced, custom 1:10, 1:50, 1:100
   - Simple and effective

2. **SMOTE (Synthetic Minority Oversampling)**
   - Create synthetic fraud examples
   - Test different sampling ratios
   - ⚠️ Use carefully (can overfit)

3. **Threshold Tuning** ⭐ **RECOMMENDED**
   - Adjust decision boundary
   - Test thresholds 0.1 to 0.9
   - Most flexible approach

### Key Metrics:

- **Recall (Primary)** - Fraud detection rate
- **Precision** - Accuracy of fraud predictions
- **F1-Score** - Balance of recall and precision
- **False Positives** - False alarms
- **False Negatives** - Missed frauds

---

## 🚀 How to Run

### Quick Demo (2-3 minutes):

```bash
python examples/demo_imbalance_handling.py
```

**Output:**
- Class weights comparison
- Threshold tuning results
- Key insights

### Full Analysis (10-15 minutes):

```bash
python src/models/handle_imbalance.py
```

**Output:**
- Comprehensive comparison
- Visualizations
- Saved models
- Detailed recommendations

---

## 📊 Expected Results

### Quick Demo:

```
Class Weights:
  No Weights:    Recall 81.6%, Precision 94.1%
  Balanced:      Recall 82.7%, Precision 79.4%
  Heavy (1:50):  Recall 82.7%, Precision 91.0%

Threshold Tuning:
  0.1: Recall 89.8%, Precision 11.6%, FP 673
  0.2: Recall 88.8%, Precision 36.6%, FP 151
  0.3: Recall 88.8%, Precision 56.1%, FP 68  ⭐ BEST
  0.4: Recall 84.7%, Precision 70.9%, FP 34
  0.5: Recall 82.7%, Precision 79.4%, FP 21
```

### Full Analysis:

- All class weight strategies tested
- All SMOTE ratios tested
- Complete threshold analysis
- 4 visualizations created
- Best model saved

---

## 💾 Output Files

**Quick Demo:** None (console output only)

**Full Analysis:**

**Location:** `models/imbalance/`
1. `random_forest_tuned.pkl` - Best model
2. `scaler.pkl` - Feature scaler
3. `best_threshold.txt` - Optimal threshold

**Location:** `reports/figures/`
1. `imbalance_handling.png` - Comparison visualizations

---

## 💡 Key Findings

### **1. Threshold 0.3 is Optimal**
- Recall: 88.8% (catch 87 out of 98 frauds)
- Precision: 56.1%
- False Positives: 68
- False Negatives: 11

### **2. Improvement Over Baseline**
- Baseline recall: 82.7%
- Improved recall: 88.8%
- **+6.1% improvement**
- Catch 6 more frauds, miss 6 fewer

### **3. Business Impact**
- Cost of missed fraud: $122 (avg fraud amount)
- Cost of false alarm: $10 (investigation)
- **ROI: Saves $232 per ~57K transactions**

### **4. Recommendation**
- ✅ Use threshold tuning (most flexible)
- ✅ Set threshold = 0.3 for production
- ✅ Monitor and adjust based on feedback
- ⚠️ Use SMOTE with caution

---

## 🎓 For Freshers

This demonstrates:

**Technical Skills:**
- ✅ Handle imbalanced datasets
- ✅ Understand recall-precision trade-off
- ✅ Apply multiple techniques
- ✅ Compare and evaluate

**Business Skills:**
- ✅ Align ML with business goals
- ✅ Calculate ROI
- ✅ Communicate trade-offs
- ✅ Make recommendations

**Perfect for interviews!** 🎯

---

## 📚 Documentation

- **`IMBALANCE_HANDLING_GUIDE.md`** - Comprehensive guide
- **`IMBALANCE_RESULTS.md`** - Results summary
- **`examples/demo_imbalance_handling.py`** - Quick demo

---

## ✨ Summary

**Two scripts available:**

1. **Quick Demo** (2-3 min)
   - Fast overview
   - Key techniques
   - Console output only

2. **Full Analysis** (10-15 min)
   - Comprehensive testing
   - Visualizations
   - Saved models

**Both are portfolio-ready!** 🚀

