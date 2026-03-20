# Configuration for Boy Or Girl Prediction Pipeline

# --- Core Settings ---
# Random Seed for Reproducibility
SEED = 42

# --- Data Paths ---
DATA_PATH = 'data/raw/train.csv'
TEST_DATA_PATH = 'data/raw/test.csv' 

# --- Preprocessing ---
# Thresholds for outlier detection
HEIGHT_THRESHOLD = 200
WEIGHT_THRESHOLD = 120

# Numeric Imputation Options:
# 'mean'          : Replace missing with the mean of the column.
# 'median'        : Replace missing with the median (robust to outliers).
# 'most_frequent' : Replace missing with the mode.
# 'constant'      : Replace missing with a specific filler value.
# 'knn'           : Use K-Nearest Neighbors to impute based on similar rows.
NUMERIC_IMPUTER_STRATEGY = 'median'

# Categorical Imputation Options:
# 'most_frequent' : Replace missing with the most common category.
# 'constant'      : Replace missing with a specific filler string (e.g., "Unknown").
CATEGORICAL_IMPUTER_STRATEGY = 'most_frequent'

# Feature Scaling Options (Crucial for distance-based models or regularized models):
# 'standard'      : StandardScaler (mean 0, variance 1).
# 'minmax'        : MinMaxScaler (scales strictly between 0 and 1).
# 'robust'        : RobustScaler (uses quartiles, resilient to outliers).
# 'none'          : Skip scaling.
SCALING_STRATEGY = 'standard'

# --- Feature Selection ---
# Selection Method Options:
# 'none'          : Use all available features.
# 'embedded_rf'   : Select based on Random Forest feature importances.
# 'embedded_l1'   : Select using Lasso (L1 regularization) penalty.
# 'rfe'           : Recursive Feature Elimination (requires defining RFE_N_FEATURES).
# 'rfecv'         : RFE with Cross-Validation (automatically finds optimal feature count).
# 'select_k_best' : Select top K features based on statistical tests (e.g., ANOVA).
FEATURE_SELECTOR = 'rfecv' 

# Selector Hyperparameters
SELECTOR_N_ESTIMATORS = 100
RFE_N_FEATURES = None  # Set integer for 'rfe', leave as None for 'rfecv'

# --- Training / Model ---
# Model Type Options:
# 'random_forest'       : Great baseline, robust to overfitting, handles unscaled data well.
# 'gradient_boosting'   : sklearn's default GBM. High performance but prone to overfitting.
# 'xgboost'             : Extreme Gradient Boosting. Often wins Kaggle tabular competitions.
# 'lightgbm'            : Fast, memory-efficient, great for larger datasets.
# 'catboost'            : Excellent out-of-the-box handling of categorical features.
# 'logistic_regression' : Simple, interpretable linear baseline.
MODEL_TYPE = 'gradient_boosting'

# Validation Strategy Options:
# 'holdout'          : Single train/test split.
# 'cross_validation' : K-Fold cross-validation.
# 'stratified_cv'    : K-Fold that preserves the percentage of samples for each class.
VALIDATION_STRATEGY = 'cross_validation'
NUM_FOLDS = 10 
TEST_SIZE = 0.2
USE_STRATIFY = True 

# Class Imbalance Handling Options:
# None                 : All classes carry equal weight.
# 'balanced'           : Automatically adjusts weights inversely proportional to class frequencies.
# 'balanced_subsample' : (RF only) Computes weights based on the bootstrap sample for each tree.
CLASS_WEIGHT = 'balanced' 

# --- Modular Hyperparameter Dictionary ---
# By using a dictionary, your training script can just call: 
# params = MODEL_PARAMS[MODEL_TYPE]

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'class_weight': CLASS_WEIGHT,
        'random_state': SEED
    },
    'gradient_boosting': {
        'learning_rate': 0.03,
        'n_estimators': 500,
        'max_depth': 3,
        'subsample': 0.8,
        'min_samples_split': 4,
        'random_state': SEED
    },
    'xgboost': {
        'learning_rate': 0.05,
        'n_estimators': 300,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED
    },
    'logistic_regression': {
        'penalty': 'l2',
        'C': 1.0,
        'class_weight': CLASS_WEIGHT,
        'max_iter': 1000,
        'random_state': SEED
    }
}

# --- Output ---
RESULTS_DIR = 'yokummy/runs'
# Dynamically name the output file based on the model chosen
MODEL_FILENAME = f'{MODEL_TYPE}_model.pkl'