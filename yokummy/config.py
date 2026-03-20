# Configuration for Boy Or Girl Prediction Pipeline

# Random Seed for Reproducibility
SEED = 42

# --- Data Loading ---
DATA_PATH = 'data/raw/train.csv'

# --- Preprocessing ---
# Thresholds for outlier detection
HEIGHT_THRESHOLD = 200
WEIGHT_THRESHOLD = 150

# Imputation Strategies
# Numeric: 'mean', 'median', 'most_frequent', 'constant'
NUMERIC_IMPUTER_STRATEGY = 'median'
# Categorical: 'most_frequent', 'constant'
CATEGORICAL_IMPUTER_STRATEGY = 'most_frequent'

# --- Feature Selection ---
# Method: 'embedded_rf', 'none'
FEATURE_SELECTOR = 'embedded_rf' 
# For embedded_rf, n_estimators for the selector model
SELECTOR_N_ESTIMATORS = 100

# --- Training / Model ---
# Validation Strategy
# 'holdout' for single split, 'cross_validation' for k-fold
VALIDATION_STRATEGY = 'cross_validation'
NUM_FOLDS = 10 # Used if VALIDATION_STRATEGY is 'cross_validation'
# Split Strategy
TEST_SIZE = 0.2
USE_STRATIFY = True  # Set to True to use stratified split

# Class Imbalance Handling
# Options: None, 'balanced', 'balanced_subsample'
CLASS_WEIGHT = 'balanced' 

# Model Hyperparameters (Random Forest)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2

# --- Output ---
RESULTS_DIR = 'yokummy/runs'
MODEL_FILENAME = 'rf_model.pkl'
