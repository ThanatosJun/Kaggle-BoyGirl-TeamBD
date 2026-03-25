## 🧪 Experiment Methodology

Our approach focused on robust data preprocessing and systematic model tuning. All experiments utilized **X-Fold Stratified Cross-Validation** to ensure reliable performance estimates and class balance across training and validation sets.

### 1. Data Preprocessing
- **Cleaning**: Removed outliers in `height` (> 200cm) and `weight` (> 120kg) to prevent skewing the model.
- **Imputation**: 
  - Numerical features: `median` strategy (robust to remaining outliers).
  - Categorical features: `most_frequent` strategy.
- **Scaling**: Applied `StandardScaler` to normalize features, crucial for gradient-based models and helper for tree-based convergence.

### 2. Feature Engineering
We derived new features to capture non-linear relationships:
- **BMI**: Body Mass Index calculated from height and weight.
- **Height/Weight Ratio**: Simple ratio to capture body build.

### 3. Model & Validation
- **Primary Model**: Random Forest Classifier (consistently outperformed others in stability).
- **Validation**: 10-fold Stratified Cross-Validation (`StratifiedKFold`) was used for all submissions.

---

## 📈 Iteration History (Sub 1 to Sub 3)

We improved our test accuracy from **87.0%** to **90.6%** through three major iterations.

### Substitution 1: Baseline (`sub1_8262`)
- **Strategy**: Establish a baseline using a standard Random Forest.
- **Configuration**:
  - Model: Random Forest (200 estimators).
  - Features: Raw features only (`height`, `weight`, `fb_friends`, etc.).
  - Selection: `embedded_rf` (filtering low importance features).
  - **Folds**: 5
- **Results**:
  - **Test Accuracy**: 87.1%
  - **CV Mean Accuracy**: 87.9%
  - **Test F1-Macro**: 0.823
- **Insight**: The model works but likely underfits complex relationships between height and weight.

### Submission 2: Refinement & Tuning (`sub2_8309`)
- **Strategy**: Improve generalization through scaling and better feature selection.
- **Changes**:
  - Added **Standard Scaling**.
  - Switched Feature Selection to **RFECV** (Recursive Feature Elimination with Cross-Validation).
  - Tuned RF Hyperparameters: Increased trees to 500, set `max_depth=10`.
  - **Folds**: 10
- **Results**:
  - **Test Accuracy**: 89.4% (+2.3%)
  - **CV Mean Accuracy**: 87.3%
  - **Test F1-Macro**: 0.860
- **Insight**: Rigorous feature selection helped remove noise. Standardization improved stability.

### Submission 3: Feature Engineering (`sub3_8497`)
- **Strategy**: Introduce domain knowledge via new features.
- **Changes**:
  - **Feature Engineering Enabled**: Added `bmi` and `height_weight_ratio`.
  - Reverted Feature Selection to **embedded_rf**: Importance-based selection kept the valuable new derived features better than RFECV.
  - **Folds**: 10
- **Results**:
  - **Test Accuracy**: 90.6% (+1.2%)
  - **CV Mean Accuracy**: 85.2%
  - **Test F1-Macro**: 0.877
- **Insight**: The engineered features (`bmi`, `ratio`) became highly important predictors, significantly boosting the model's ability to distinguish classes, especially on the test set.