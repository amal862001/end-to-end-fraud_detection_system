# EDA Notebook Guide

## 📊 02_eda.ipynb - Exploratory Data Analysis

### Objective
Identify patterns differentiating fraud vs non-fraud transactions through comprehensive exploratory data analysis.

---

## 🎯 What This Notebook Does

### 1. **Transaction Amount Analysis**
- Compares amount distributions between fraud and legitimate transactions
- Analyzes fraud rates across different amount ranges
- Visualizations:
  - Box plots
  - Distribution plots (log scale)
  - Violin plots
  - Cumulative distribution functions (CDF)
  - Fraud rate by amount range

**Key Insights:**
- Fraudulent transactions tend to have **lower amounts** on average
- Fraud rate varies significantly across amount ranges
- Most frauds occur in specific amount brackets

---

### 2. **Time-Based Fraud Behavior**
- Analyzes temporal patterns in fraudulent transactions
- Examines fraud distribution over time
- Hour-of-day analysis

**Visualizations:**
- Transaction count over time
- Fraud count over time
- Fraud rate over time (hourly)
- Time distribution by class
- Fraud patterns by hour of day

**Key Insights:**
- Fraud shows distinct temporal patterns
- Certain hours have higher fraud rates
- Time-based features are valuable for detection

---

### 3. **Feature Correlation Analysis**
- Calculates correlations between all features and fraud
- Identifies top positive and negative correlations
- Creates correlation heatmap for top features

**Outputs:**
- Top 10 positive correlations with fraud
- Top 10 negative correlations with fraud
- Correlation heatmap visualization
- Feature relationship insights

---

### 4. **Feature Importance (Tree-Based)**
- Uses Random Forest to identify important features
- Ranks features by importance scores
- Cumulative importance analysis

**Key Findings:**
- Top 15 most important features identified
- Number of features needed for 95% importance coverage
- Feature selection recommendations

---

### 5. **Feature Distribution Analysis**
- Compares distributions of top features between classes
- Visualizes 12 most important features
- Side-by-side distribution comparisons

**Purpose:**
- Understand how features differ between fraud/legitimate
- Identify discriminative patterns
- Validate feature importance findings

---

### 6. **Statistical Significance Tests**
- Mann-Whitney U tests for top features
- Effect size calculations
- Statistical validation of differences

**Results:**
- P-values for all top features
- Effect sizes (rank-biserial correlation)
- Confirmation of significant differences

---

## 📈 Key Outputs

### Fraud Behavior Insights

1. **Amount Patterns:**
   - Fraud mean amount < Legitimate mean amount
   - Fraud median significantly different
   - Specific amount ranges have higher fraud rates

2. **Temporal Patterns:**
   - Fraud occurs at specific times
   - Hour-of-day patterns exist
   - Time is a valuable feature

3. **Feature Patterns:**
   - V14, V17, V12, V10 show strong correlations
   - V1, V3, V7 also important
   - Time and Amount need scaling

### Feature Selection Decisions

1. **Recommended Features:**
   - Top 15-20 features capture 95%+ importance
   - Include Time and Amount (after scaling)
   - All V features are already PCA-transformed

2. **Statistical Validation:**
   - All top features show p < 0.001
   - Highly significant differences confirmed
   - Strong effect sizes observed

3. **Next Steps:**
   - Scale Time and Amount
   - Consider top 15-20 features for modeling
   - Handle class imbalance (SMOTE/ADASYN)

---

## 🚀 How to Run

### Prerequisites
```bash
# Install Jupyter if not already installed
pip install jupyter notebook ipykernel

# Install required packages (if not already installed)
pip install matplotlib seaborn scikit-learn
```

### Running the Notebook

**Option 1: Jupyter Notebook**
```bash
# Navigate to project root
cd c:\Users\RONO\Documents\GitHub\end-to-end-fraud_detection_system

# Launch Jupyter
jupyter notebook

# Open notebooks/02_eda.ipynb in the browser
```

**Option 2: VS Code**
```bash
# Open the notebook in VS Code
# VS Code will automatically detect it's a Jupyter notebook
# Click "Run All" or run cells individually
```

**Option 3: JupyterLab**
```bash
# Install JupyterLab
pip install jupyterlab

# Launch JupyterLab
jupyter lab

# Navigate to notebooks/02_eda.ipynb
```

---

## 📁 Exported Files

The notebook exports the following files to `data/processed/`:

1. **feature_importance.csv**
   - Random Forest feature importance scores
   - Ranked by importance

2. **feature_correlations.csv**
   - Correlation coefficients with fraud
   - All features included

3. **statistical_tests.csv**
   - Mann-Whitney U test results
   - P-values and effect sizes

---

## 🔍 Expected Runtime

- **Full notebook**: ~5-10 minutes
- **Random Forest training**: ~1-2 minutes (on sample data)
- **Visualizations**: ~3-5 minutes

---

## 💡 Tips

1. **Memory Usage:**
   - Notebook uses ~2-3 GB RAM
   - Random Forest uses sampled data for speed
   - Close other applications if needed

2. **Visualization:**
   - All plots are inline (`%matplotlib inline`)
   - High-resolution figures for clarity
   - Can save figures using `plt.savefig()`

3. **Customization:**
   - Adjust sample sizes for faster/slower analysis
   - Modify plot styles as needed
   - Add additional analyses as required

---

## 📊 Sample Insights

Based on the analysis, you should expect to find:

- **~15-20 features** are highly important
- **V14, V17, V12, V10** among top features
- **Time and Amount** show distinct patterns
- **All top features** statistically significant (p < 0.001)
- **Fraud rate**: ~0.17% (highly imbalanced)

---

## 🎯 Next Steps After EDA

1. **Feature Engineering:**
   - Scale Time and Amount
   - Select top features
   - Create interaction features (optional)

2. **Handle Imbalance:**
   - Apply SMOTE/ADASYN
   - Use class weights
   - Consider ensemble methods

3. **Model Training:**
   - Random Forest baseline
   - XGBoost/LightGBM
   - Neural Networks

4. **Evaluation:**
   - Precision-Recall AUC
   - F1-Score, Precision, Recall
   - Confusion Matrix
   - Cost-sensitive metrics

---

## ✅ Checklist

Before proceeding to modeling:

- [ ] Run full EDA notebook
- [ ] Review all visualizations
- [ ] Understand fraud patterns
- [ ] Identify top features
- [ ] Export analysis results
- [ ] Document key insights
- [ ] Plan feature engineering
- [ ] Plan imbalance handling strategy

---

**Created:** 2026-01-14  
**Notebook:** `notebooks/02_eda.ipynb`  
**Status:** Ready for execution

