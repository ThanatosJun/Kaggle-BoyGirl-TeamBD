import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import config  # Import configuration

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
imputer_num = SimpleImputer(strategy=config.NUMERIC_IMPUTER_STRATEGY)
if num_cols:
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Categorical Imputer
imputer_cat = SimpleImputer(strategy=config.CATEGORICAL_IMPUTER_STRATEGY)
if len(cat_cols) > 0:
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

print("Missing values after cleaning:", df.isnull().sum().sum())

# 4. Feature Engineering
print("\n--- Feature Engineering ---")
# Encode Categorical Variables
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"Encoded {col}")

X = df.drop('gender', axis=1)
y = df['gender']

# 5. Feature Selection
print(f"\n--- Feature Selection: {config.FEATURE_SELECTOR} ---")
if config.FEATURE_SELECTOR == 'embedded_rf':
    rf_selector = RandomForestClassifier(n_estimators=config.SELECTOR_N_ESTIMATORS, random_state=config.SEED)
    rf_selector.fit(X, y)
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature Ranking:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")
    # In this prototype, we just rank, but you could add logic to drop features here
    pass 
else:
    print("Skipping embedded feature selection.")


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
rf = RandomForestClassifier(
    n_estimators=config.RF_N_ESTIMATORS,
    random_state=config.SEED,
    class_weight=config.CLASS_WEIGHT,
    max_depth=config.RF_MAX_DEPTH,
    min_samples_split=config.RF_MIN_SAMPLES_SPLIT
)

if config.VALIDATION_STRATEGY == 'cross_validation':
    print(f"Executing {config.NUM_FOLDS}-Fold Cross-Validation on Training Data...")
    
    cv_strategy = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    
    # Evaluate accuracy
    scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='accuracy')
    print(f"CV Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Evaluate F1-macro (or weighted) for better class imbalance insight
    f1_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='f1_macro')
    print(f"Mean CV F1-Macro: {f1_scores.mean():.4f}")


# 7. Model Training (Final Model on full Training Set)
print(f"\n--- Training Final Model (Class Weight: {config.CLASS_WEIGHT}) ---")
rf.fit(X_train, y_train)


# 8. Evaluation
print("\n--- Model Evaluation ---")
y_pred = rf.predict(X_test)
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
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")

# 3. Save Results
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Calculate and sort feature importances
feature_importances = {}
if hasattr(rf, "feature_importances_"):
    # Zip, sort by importance (descending), and convert to dict
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importances = {
        X.columns[i]: importances[i] for i in sorted_idx
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

print("\nDone!")
