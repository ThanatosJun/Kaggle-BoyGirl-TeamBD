import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# Consolidated sklearn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import RFECV, RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import config  # Import configuration

# Optional imports for extra models
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# Use Seed from config
np.random.seed(config.SEED)

# 1. Load Data
try:
    df = pd.read_csv(config.DATA_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found. Make sure '{config.DATA_PATH}' exists.")
    exit()

# 2. EDA (kept minimal for pipeline)
print("\n--- EDA Summary ---")
print(f"Shape: {df.shape}")
print(f"Target Distribution:\n{df['gender'].value_counts()}")

# 3. Data Cleaning
print("\n--- Data Cleaning ---")
# Drop 'id'
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Ensure 'yt' is numeric
df['yt'] = pd.to_numeric(df['yt'], errors='coerce')

# Handle Outliers
# Replace infinity with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace unreasonable values with NaN based on config
df.loc[df['height'] > config.HEIGHT_THRESHOLD, 'height'] = np.nan
df.loc[df['weight'] > config.WEIGHT_THRESHOLD, 'weight'] = np.nan

# Handle Missing Values
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'gender' in num_cols: num_cols.remove('gender') # Don't impute target if numeric type

cat_cols = df.select_dtypes(include='object').columns

# Numeric Imputer
print(f"Imputing numeric values with strategy: {config.NUMERIC_IMPUTER_STRATEGY}")
if config.NUMERIC_IMPUTER_STRATEGY == 'knn':
    imputer_num = KNNImputer(n_neighbors=5)
else:
    imputer_num = SimpleImputer(strategy=config.NUMERIC_IMPUTER_STRATEGY)
    
if num_cols:
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Categorical Imputer
print(f"Imputing categorical values with strategy: {config.CATEGORICAL_IMPUTER_STRATEGY}")
imputer_cat = SimpleImputer(strategy=config.CATEGORICAL_IMPUTER_STRATEGY)
if len(cat_cols) > 0:
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Scaling
scaler = None
if hasattr(config, 'SCALING_STRATEGY') and config.SCALING_STRATEGY != 'none':
    print(f"Applying scaling: {config.SCALING_STRATEGY}")
    if config.SCALING_STRATEGY == 'standard':
        scaler = StandardScaler()
    elif config.SCALING_STRATEGY == 'minmax':
        scaler = MinMaxScaler()
    elif config.SCALING_STRATEGY == 'robust':
        scaler = RobustScaler()
        
    if scaler and num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

print("Missing values after cleaning:", df.isnull().sum().sum())

# 4. Feature Engineering
print("\n--- Feature Engineering ---")
# Encode Categorical Variables
le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on strings
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le
    print(f"Encoded {col}")

X = df.drop('gender', axis=1)
y = df['gender']

# XGBoost/LightGBM/CatBoost require 0-indexed classes [0, 1].
# Random Forest/Logistic Regression work fine with [1, 2], but for consistency
# and safety with all sklearn-compatible boosting libs, let's normalize y to start at 0.
# We will map them back later.
y_min = y.min()
y = y - y_min # Now y is [0, 1]

# 5. Feature Selection
print(f"\n--- Feature Selection: {config.FEATURE_SELECTOR} ---")
selected_features = X.columns.tolist()

if config.FEATURE_SELECTOR == 'embedded_rf':
    rf_selector = RandomForestClassifier(n_estimators=config.SELECTOR_N_ESTIMATORS, random_state=config.SEED)
    rf_selector.fit(X, y)
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature Ranking (All features kept):")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")
        
elif config.FEATURE_SELECTOR == 'embedded_l1':
    # Lasso requires scaling. If not scaled, this might perform poorly.
    l1_selector = SelectFromModel(Lasso(alpha=0.01, random_state=config.SEED))
    l1_selector.fit(X, y)
    mask = l1_selector.get_support()
    selected_features = X.columns[mask].tolist()
    print(f"Selected {len(selected_features)} features with Lasso: {selected_features}")
    X = X[selected_features]

elif config.FEATURE_SELECTOR == 'rfe':
    n_feat = config.RFE_N_FEATURES if config.RFE_N_FEATURES else 5
    rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=config.SEED), n_features_to_select=n_feat)
    rfe_selector.fit(X, y)
    mask = rfe_selector.support_
    selected_features = X.columns[mask].tolist()
    print(f"Selected {len(selected_features)} features with RFE: {selected_features}")
    X = X[selected_features]

elif config.FEATURE_SELECTOR == 'rfecv':
    rfecv_selector = RFECV(estimator=RandomForestClassifier(n_estimators=50, random_state=config.SEED), step=1, cv=StratifiedKFold(5), scoring='accuracy')
    rfecv_selector.fit(X, y)
    mask = rfecv_selector.support_
    selected_features = X.columns[mask].tolist()
    print(f"Optimal number of features: {rfecv_selector.n_features_}")
    print(f"Selected Features: {selected_features}")
    X = X[selected_features]

elif config.FEATURE_SELECTOR == 'select_k_best':
    k_best = 5 # Default k
    skb_selector = SelectKBest(f_classif, k=k_best)
    skb_selector.fit(X, y)
    mask = skb_selector.get_support()
    selected_features = X.columns[mask].tolist()
    print(f"Selected top {k_best} features with SelectKBest: {selected_features}")
    X = X[selected_features]

else:
    print("No feature selection applied (or 'none').")


# 6. Data Splitting
print("\n--- Data Splitting ---")
stratify_param = y if config.USE_STRATIFY else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config.TEST_SIZE, 
    random_state=config.SEED,
    stratify=stratify_param
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
if config.USE_STRATIFY:
    print("Stratified split applied.")


# 6.5. Cross-Validation (Optional)
print(f"\n--- Validation Strategy: {config.VALIDATION_STRATEGY} ---")

model_type = config.MODEL_TYPE
model_params = config.MODEL_PARAMS[model_type].copy()
print(f"Initializing {model_type} with params: {model_params}")

if model_type == 'random_forest':
    model = RandomForestClassifier(**model_params)
elif model_type == 'gradient_boosting':
    model = GradientBoostingClassifier(**model_params)
elif model_type == 'logistic_regression':
    model = LogisticRegression(**model_params)
    
elif model_type == 'xgboost':
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Please install it or choose another model.")
    model = XGBClassifier(**model_params)
    
elif model_type == 'lightgbm':
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed. Please install it or choose another model.")
    model = LGBMClassifier(**model_params)

elif model_type == 'catboost':
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed. Please install it or choose another model.")
    model = CatBoostClassifier(**model_params, verbose=0)
else:
    raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

if config.VALIDATION_STRATEGY == 'cross_validation':
    print(f"Executing {config.NUM_FOLDS}-Fold Cross-Validation on Training Data...")
    
    cv_strategy = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    
    # Evaluate accuracy
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
    print(f"CV Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Evaluate F1-macro (or weighted) for better class imbalance insight
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro')
    print(f"Mean CV F1-Macro: {f1_scores.mean():.4f}")


# 7. Model Training (Final Model on full Training Set)
print(f"\n--- Training Final Model ({config.MODEL_TYPE}) ---")
model.fit(X_train, y_train)


# 8. Evaluation
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report_str = classification_report(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:")
print(report_str)

# 9. Model Persistence & Artifacts Saving
print("\n--- Saving Artifacts ---")

# Generate Run ID based on timestamp
run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = os.path.join(config.RESULTS_DIR, run_id)
if not os.path.exists(config.RESULTS_DIR): # create parent dir if needed
    os.makedirs(config.RESULTS_DIR)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

print(f"Directory created: {run_dir}")

# 1. Save Config
config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__') and k.isupper()}
config_path = os.path.join(run_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config_dict, f, indent=4)
print(f"Config saved to {config_path}")

# 2. Save Model
model_path = os.path.join(run_dir, config.MODEL_FILENAME)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# 3. Save Results
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Calculate and sort feature importances
feature_importances = {}
if hasattr(model, "feature_importances_"):
    # Zip, sort by importance (descending), and convert to dict
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importances = {
        X.columns[i]: float(importances[i]) for i in sorted_idx
    }

results = {
    "run_id": run_id,
    "timestamp": datetime.now().isoformat(),
    "test_accuracy": acc,
    "classification_report": report_dict,
    "feature_importances": feature_importances
}

# If CV was run, add CV results
if config.VALIDATION_STRATEGY == 'cross_validation':
    results["cv_mean_accuracy"] = scores.mean()
    results["cv_std_accuracy"] = scores.std()
    results["cv_mean_f1_macro"] = f1_scores.mean()

result_path = os.path.join(run_dir, "result.json")
with open(result_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {result_path}")


# 10. Generate Submission File (Prediction on Test Data)
print("\n--- Generating Submission ---")
try:
    if os.path.exists(config.TEST_DATA_PATH):
        print(f"Loading test data from: {config.TEST_DATA_PATH}")
        test_df = pd.read_csv(config.TEST_DATA_PATH)
    else:
        # Fallback: try to find a file starting with 'test' in data/raw
        import glob
        test_files = glob.glob('data/raw/test*.csv')
        if test_files:
            test_df = pd.read_csv(test_files[0])
            print(f"Loading test data from found file: {test_files[0]}")
        else:
            raise FileNotFoundError("No test CSV found.")

    test_ids = test_df['id'] # Keep IDs for submission

    # --- Preprocess Test Data (Mirroring Train Data) ---
    if 'id' in test_df.columns:
        test_df_clean = test_df.drop('id', axis=1)
    else:
        test_df_clean = test_df.copy()

    # Convert 'yt' to numeric
    test_df_clean['yt'] = pd.to_numeric(test_df_clean['yt'], errors='coerce')

    # Handle Outliers (Infinity and Thresholds)
    # Cast to float first to ensure replace works on object cols if mixed
    # And convert large numbers (larger than float32 max) to NaN
    
    # Force numeric columns to be float
    cols_to_numeric = ['height', 'weight', 'fb_friends', 'iq', 'sleepiness']
    for c in cols_to_numeric:
        if c in test_df_clean.columns:
            # Handle mixed types explicitly, replace None or empty string first if needed
            test_df_clean[c] = pd.to_numeric(test_df_clean[c], errors='coerce')
            
    # Explicitly check for large values
    for c in cols_to_numeric:
        if c in test_df_clean.columns:
             # Float32 max is ~3.4e38, Python float goes up to ~1.7e308
             # But sklearn might warn/error on float32/64 overflow if value is too huge
             # Let's cap at a reasonable large number or treat as outlier
             # Assuming threshold config are reasonable
             pass

    test_df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if 'height' in test_df_clean.columns:
        test_df_clean.loc[test_df_clean['height'] > config.HEIGHT_THRESHOLD, 'height'] = np.nan
    if 'weight' in test_df_clean.columns:
        test_df_clean.loc[test_df_clean['weight'] > config.WEIGHT_THRESHOLD, 'weight'] = np.nan

    # Impute Missing Values
    # First ensure we don't have extremely large values that survived.
    # Replace anything > 1e12 with NaN as a safety catch-all for submission errors
    # Apply only to numeric columns to avoid TypeError with strings
    numeric_cols_mask = test_df_clean.select_dtypes(include=[np.number]).columns
    test_df_clean[numeric_cols_mask] = test_df_clean[numeric_cols_mask].mask(test_df_clean[numeric_cols_mask] > 1e12, np.nan)

    # Impute Missing Values (using the fitting from Train data)
    num_cols_test = [c for c in num_cols if c in test_df_clean.columns]
    if num_cols_test:
        test_df_clean[num_cols_test] = imputer_num.transform(test_df_clean[num_cols_test])
    
    cat_cols_test = [c for c in cat_cols if c in test_df_clean.columns]
    if cat_cols_test:
        test_df_clean[cat_cols_test] = imputer_cat.transform(test_df_clean[cat_cols_test])

    # Scaling
    if scaler and num_cols_test:
        # Ensure we have all columns required by scaler
        # We can only scale if columns match exactly those fitted.
        # If test set is missing columns, we might need to fill them first.
        # For now assume test set schema matches training schema for numeric cols.
        try:
             test_df_clean[num_cols] = scaler.transform(test_df_clean[num_cols])
        except KeyError:
             print("Warning: Test data missing numeric columns for scaling. Skipping scaling.")
        except ValueError:
             print("Warning: Scaling mismatch (shape). Skipping scaling.")

    # Feature Engineering (Encoding)
    for col, le in le_map.items():
        if col in test_df_clean.columns:
            # Handle unseen labels by mapping them to known classes (e.g., first class)
            # or converting everything to string first.
            test_vals = test_df_clean[col].astype(str)
            
            # Identify unseen labels
            known_labels = set(le.classes_)
            unknown_mask = ~test_vals.isin(known_labels)
            
            if unknown_mask.any():
                # Replace unknown with the mode (most frequent from training) or a placeholder
                # Here we use the first known class as a fallback
                fallback_label = le.classes_[0]
                test_vals[unknown_mask] = fallback_label
            
            test_df_clean[col] = le.transform(test_vals)

    # Ensure column order matches training, filling missing with 0 if necessary
    # (Though imputation should have handled missing, features might be dropped)
    # X.columns is from training data
    
    # Align columns: Add missing columns
    for col in X.columns:
        if col not in test_df_clean.columns:
            test_df_clean[col] = 0 # Or NaN, depending on imputation logic
            
    # Drop extra columns
    test_df_clean = test_df_clean[X.columns]

    # Predict
    predictions = model.predict(test_df_clean)
    
    # Map predictions back to original labels (e.g. 0->1, 1->2)
    predictions = predictions + y_min

    # Create Submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'gender': predictions
    })

    # Save Submission
    submission_path = os.path.join(run_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

except FileNotFoundError:
    print("Test file not found. Skipping submission generation.")
except Exception as e:
    print(f"Error generating submission: {e}")

print("\nDone!")
