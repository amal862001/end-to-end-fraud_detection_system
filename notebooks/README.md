# 📓 Notebooks

This directory contains Jupyter notebooks for exploratory data analysis and experimentation.

## 📊 02_eda.ipynb - Exploratory Data Analysis

**Purpose:** Identify patterns that differentiate fraudulent from legitimate credit card transactions.

### What's Inside:

1. **Transaction Amount Analysis**
   - Compare fraud vs legitimate transaction amounts
   - Visualize distributions with box plots and histograms
   - Key finding: Fraud transactions have different amount patterns

2. **Feature Correlations**
   - Calculate correlation of all features with fraud
   - Identify top features correlated with fraud
   - Visualize top 15 features

3. **Feature Importance (Random Forest)**
   - Train Random Forest classifier
   - Rank features by importance
   - Identify most predictive features

4. **Time-Based Analysis**
   - Analyze fraud patterns by hour of day
   - Identify peak fraud hours
   - Visualize temporal patterns

5. **Key Insights & Recommendations**
   - Summary of findings
   - Recommendations for modeling
   - Next steps

### How to Run:

```bash
# 1. Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# 2. Install Jupyter (if not installed)
pip install jupyter notebook

# 3. Launch Jupyter
jupyter notebook

# 4. Open notebooks/02_eda.ipynb
# 5. Run all cells: Cell → Run All
```

### Expected Runtime:
- **Total:** 3-5 minutes
- Data loading: ~30 seconds
- Random Forest training: 1-2 minutes
- Visualizations: 1-2 minutes

### Outputs:

**Files created:**
- `data/processed/feature_importance.csv` - Feature importance scores
- `data/processed/feature_correlations.csv` - Correlation coefficients

**Visualizations:**
- Transaction amount distributions
- Feature correlation chart
- Feature importance chart
- Time-based fraud patterns

### Key Findings:

1. **Amount Patterns:** Fraud transactions show distinct amount characteristics
2. **Important Features:** V14, V17, V12, V10 are top predictors
3. **Time Patterns:** Fraud occurs at specific hours
4. **Class Imbalance:** 99.83% legitimate, 0.17% fraud

### Next Steps:

After running this notebook, you'll have:
- ✅ Understanding of fraud patterns
- ✅ Feature importance rankings
- ✅ Insights for feature engineering
- ✅ Recommendations for modeling

Use these insights to build your fraud detection model!

---

## 💡 Tips for Freshers:

This notebook demonstrates:
- **Data exploration skills** - Understanding your data before modeling
- **Visualization skills** - Creating clear, informative plots
- **Feature analysis** - Identifying important features
- **Business insights** - Translating data into actionable findings

Perfect for showcasing in your portfolio! 🎯

