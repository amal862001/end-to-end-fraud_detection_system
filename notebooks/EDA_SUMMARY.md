# 📊 EDA Notebook - Quick Reference

## ✅ What's Created

**File:** `notebooks/02_eda.ipynb`

A clean, beginner-friendly exploratory data analysis notebook perfect for fresher portfolios.

---

## 📋 Notebook Structure

### **8 Sections:**

1. **Setup** - Import libraries and configure settings
2. **Load Data** - Load credit card transaction dataset
3. **Transaction Amount Analysis** - Compare fraud vs legitimate amounts
4. **Feature Correlations** - Find features correlated with fraud
5. **Feature Importance** - Use Random Forest to rank features
6. **Time-Based Analysis** - Identify temporal fraud patterns
7. **Key Insights** - Summarize findings and recommendations
8. **Save Results** - Export analysis for modeling

---

## 🎯 Key Analyses

### 1. Transaction Amount Analysis
```
✓ Compare mean/median amounts
✓ Box plot visualization
✓ Histogram (log scale)
✓ Identify amount patterns
```

### 2. Feature Correlations
```
✓ Calculate correlation with fraud
✓ Identify top 15 features
✓ Visualize with bar chart
✓ Color-coded (green=positive, red=negative)
```

### 3. Feature Importance
```
✓ Train Random Forest classifier
✓ Extract feature importance scores
✓ Rank all features
✓ Visualize top 15 features
```

### 4. Time-Based Analysis
```
✓ Convert time to hours
✓ Analyze fraud by hour of day
✓ Identify peak fraud hours
✓ Compare fraud rate across hours
```

---

## 📊 Visualizations (6 Total)

1. **Amount Box Plot** - Compare distributions
2. **Amount Histogram** - Log scale comparison
3. **Correlation Bar Chart** - Top 15 features
4. **Feature Importance Chart** - Top 15 features
5. **Fraud Count by Hour** - Temporal patterns
6. **Fraud Rate by Hour** - Hourly fraud rates

---

## 💾 Outputs

**CSV Files Created:**
- `data/processed/feature_importance.csv`
- `data/processed/feature_correlations.csv`

**Console Output:**
- Dataset statistics
- Amount comparisons
- Top features
- Key insights
- Recommendations

---

## 🚀 How to Run

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Launch Jupyter
jupyter notebook

# 3. Open 02_eda.ipynb

# 4. Run all cells
```

**Runtime:** 3-5 minutes

---

## 💡 Why This Notebook is Great for Freshers

### ✅ **Simple & Clear**
- No complex code
- Easy to understand
- Well-commented
- Clean visualizations

### ✅ **Demonstrates Key Skills**
- Data exploration
- Statistical analysis
- Feature engineering insights
- Business understanding

### ✅ **Professional**
- Organized structure
- Clear objectives
- Actionable insights
- Proper documentation

### ✅ **Portfolio-Ready**
- Shows analytical thinking
- Demonstrates ML knowledge
- Clean, presentable code
- Real-world application

---

## 📈 Expected Insights

### Amount Patterns:
- Fraud: Mean ~$122, Median ~$9
- Legitimate: Mean ~$88, Median ~$22
- **Insight:** Fraud has different amount distribution

### Top Features:
1. V14 (highest importance)
2. V17
3. V12
4. V10
5. V11

### Time Patterns:
- Fraud occurs at specific hours
- Peak hours identified
- Temporal patterns exist

### Data Characteristics:
- Highly imbalanced (0.17% fraud)
- 30 features (28 PCA + Time + Amount)
- 284,807 transactions

---

## 🎓 Learning Outcomes

After completing this notebook, you demonstrate:

1. **Data Understanding**
   - Load and explore datasets
   - Understand class imbalance
   - Identify data characteristics

2. **Feature Analysis**
   - Calculate correlations
   - Use tree-based models for importance
   - Identify predictive features

3. **Visualization Skills**
   - Create meaningful plots
   - Compare distributions
   - Present insights clearly

4. **Business Insights**
   - Translate data to insights
   - Make recommendations
   - Understand fraud patterns

---

## 🔄 Next Steps

Use insights from this EDA to:

1. **Feature Engineering**
   - Scale Time and Amount
   - Select top 15-20 features
   - Create new features if needed

2. **Model Development**
   - Handle class imbalance (SMOTE)
   - Train classification models
   - Use appropriate metrics

3. **Evaluation**
   - Focus on Precision-Recall
   - Analyze confusion matrix
   - Optimize for business goals

---

## ✨ Perfect For:

- ✅ Data Science fresher portfolios
- ✅ ML engineer interviews
- ✅ GitHub showcases
- ✅ Learning EDA best practices
- ✅ Understanding fraud detection

---

**Ready to run and impress! 🎉**

