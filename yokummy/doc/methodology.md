# Methodology Report

This document outlines the data processing, feature engineering, and modeling strategies implemented in the current prototype (`main.py`).

## 1. Data Preprocessing & Cleaning

### Handling Dirty Data (`yt` column)
The `yt` column contained mixed data types (numbers, text like "Cool", and errors like "#NUM!").
- **Method**: Coercion to Numeric.
- **Implementation**: `pd.to_numeric(df['yt'], errors='coerce')` converts non-numeric values to `NaN`.

### Outlier Detection & Handling
Extreme values were observed in `height` and `weight` (e.g., `1e+111`).
- **Method**: Heuristic Thresholding.
- **Rules applied**:
  - **Height**: Values > 300 cm are treated as anomalies and replaced with `NaN`.
  - **Weight**: Values > 1000 kg are treated as anomalies and replaced with `NaN`.
  - **Infinity**: Positive/Negative infinity values are replaced with `NaN`.
- **Reasoning**: These `NaN` values are then safely handled by the imputer in the next step, effectively correcting the outliers to the median.

### Missing Value Imputation
After outlier handling, missing values are imputed to ensure the model has complete input data.
- **Numerical Columns**: **Median Imputation**.
  - Chosen because the median is more robust to outliers than the mean.
- **Categorical Columns**: **Mode (Most Frequent) Imputation**.
  - Fills missing values with the most common category.

## 2. Feature Selection Strategy

- **Method**: **Embedded Method**.
- **Algorithm**: Random Forest Feature Importance (Mean Decrease in Impurity / Gini Importance).
- **Process**:
  1. A Random Forest model is trained on the full feature set.
  2. Feature importance scores are extracted.
  3. Features are ranked by their contribution to the model's predictive power.
- **Current Status**: The prototype calculates and displays these rankings to inform the user, but currently **retains all features** for the final model training.

## 3. Data Splitting

- **Train/Test Ratio**: 80% Training / 20% Testing.
- **Random Seed**: `42` (for reproducibility).
- **Stratification**: **No**. The current split is purely random and does not enforce the same class distribution in train and test sets (`stratify=None`).

## 4. Modeling & Class Imbalance

- **Algorithm**: Random Forest Classifier.
  - `n_estimators`: 100
  - `random_state`: 42
- **Handling Class Imbalance**:
  - **Current Status**: The baseline model does **not** explicitly address class imbalance (e.g., no `class_weight='balanced'`, no SMOTE resampling).
  - **Impact**: As noted in the evaluation report, this results in high precision/recall for the majority class (Gender 1) but lower performance for the minority class (Gender 2).
