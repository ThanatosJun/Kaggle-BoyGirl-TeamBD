# Prototype EDA & Model Report

## 1. Overview
This report summarizes the initial Exploratory Data Analysis (EDA), data cleaning, feature engineering, and baseline modeling results for the "Boy or Girl" dataset.

**Dataset Status:**
- **Source:** `data/raw/train.csv`
- **Total Rows:** 423
- **Class Balance:**
  - Class 1 (Gender 1): 316 (74.7%)
  - Class 2 (Gender 2): 107 (25.3%)

## 2. Data Quality Issues Identified
| Issue Type | Description | Handling Strategy |
| :--- | :--- | :--- |
| **Missing Values** | All feature columns have missing values (approx. 18-25% per column). | **Numerical:** Imputed with Median.<br>**Categorical:** Imputed with Mode. |
| **Outliers** | Extreme values found in `height` and `weight` (e.g., `1e+111`). | Replaced infinity with NaN. Capped `height` > 300 and `weight` > 1000 as NaN, then imputed. |
| **Data Types** | `yt` column contained mixed text (e.g., "Cool", "#NUM!") and numbers. | Coerced errors to NaN, then treated as numeric. |

## 3. Feature Selection (Embedded Method)
Using a Random Forest Classifier to determine feature importance (Gini impurity reduction).

**Feature Ranking:**
1. **height** (36.16%) - *Dominant feature*
2. **weight** (19.78%)
3. **fb_friends** (10.42%)
4. **iq** (8.31%)
5. **self_intro** (8.16%)
6. **yt** (7.60%)
7. **star_sign** (4.85%)
8. **sleepiness** (2.77%)
9. **phone_os** (1.95%)

**Insight:** Physical attributes (`height`, `weight`) are the strongest predictors, followed by social metrics (`fb_friends`).

## 4. Baseline Model Performance
**Model:** Random Forest Classifier (n_estimators=100)
**Validation Split:** 80/20

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **83.5%** |
| **Macro Avg F1** | 0.75 |
| **Weighted Avg F1** | 0.83 |

**Detailed Classification Report:**
```text
              precision    recall  f1-score   support

     Class 1       0.87      0.92      0.90        65
     Class 2       0.69      0.55      0.61        20
```

**Observation:** The model performs well on the majority class (Class 1) but struggles with recall on the minority class (Class 2), which is expected given the class imbalance.

## 5. Next Steps
1. **Model Persistence:** The model is currently *not* saved. We need to add `joblib.dump()` to save it.
2. **Text Processing:** `self_intro` is currently just label-encoded. We should extract features like "length of intro" or "contains keywords".
3. **Balancing:** Try oversampling (SMOTE) or class weighting to improve Class 2 performance.
