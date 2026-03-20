"""
Data loading, cleaning, and splitting functions
Corresponds to Phase 1 of workflow.md
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from typing import Tuple, List


def load_data(train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV (optional)

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean data: remove duplicates, handle outliers

    Args:
        df: Input dataframe
        config: Configuration dict from config.yaml

    Returns:
        Cleaned dataframe
    """
    df = df.copy()

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")

    # Handle outliers (if enabled)
    if config['preprocessing'].get('outlier_detection', False):
        # Example: remove extreme height/weight values
        if 'Height' in df.columns:
            threshold = config['preprocessing'].get('height_threshold', 200)
            df = df[df['Height'] < threshold]

        if 'Weight' in df.columns:
            threshold = config['preprocessing'].get('weight_threshold', 120)
            df = df[df['Weight'] < threshold]

    return df


def impute_missing_values(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values using strategy from config
    IMPORTANT: Fit on train, transform on validation

    Args:
        X_train: Training features
        X_val: Validation features
        config: Configuration dict

    Returns:
        Tuple of (X_train_imputed, X_val_imputed)
    """
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()

    # Numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        strategy = config['preprocessing'].get('numeric_imputer', 'median')
        imputer = SimpleImputer(strategy=strategy)
        X_train_imputed[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_val_imputed[numeric_cols] = imputer.transform(X_val[numeric_cols])

    # Categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        strategy = config['preprocessing'].get('categorical_imputer', 'most_frequent')
        imputer = SimpleImputer(strategy=strategy)
        X_train_imputed[categorical_cols] = imputer.fit_transform(X_train[categorical_cols])
        X_val_imputed[categorical_cols] = imputer.transform(X_val[categorical_cols])

    return X_train_imputed, X_val_imputed


def create_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    stratified: bool = True,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits

    Args:
        X: Features
        y: Target
        n_folds: Number of folds
        stratified: Use stratified split
        random_state: Random seed

    Returns:
        List of (train_idx, val_idx) tuples
    """
    if stratified:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits = list(kfold.split(X, y))
    return splits
