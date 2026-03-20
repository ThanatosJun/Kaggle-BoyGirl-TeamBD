"""
Feature engineering functions
Corresponds to Phase 2 of workflow.md
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Tuple, List


def encode_categorical(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical features

    Args:
        X_train: Training features
        X_val: Validation features

    Returns:
        Tuple of (X_train_encoded, X_val_encoded)
    """
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    if len(categorical_cols) == 0:
        return X_train, X_val

    # Use pd.get_dummies with consistent columns
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)

    # Align columns (in case validation has different categories)
    X_train_encoded, X_val_encoded = X_train_encoded.align(X_val_encoded, join='left', axis=1, fill_value=0)

    return X_train_encoded, X_val_encoded


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric features
    IMPORTANT: Fit on train, transform on validation

    Args:
        X_train: Training features
        X_val: Validation features
        method: Scaling method ('standard', 'minmax', 'robust', 'none')

    Returns:
        Tuple of (X_train_scaled, X_val_scaled)
    """
    if method == 'none':
        return X_train, X_val

    # Select scaler
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scalers.get(method, StandardScaler())

    # Scale
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    return X_train_scaled, X_val_scaled


def apply_pca(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    n_components: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply PCA for dimensionality reduction

    Args:
        X_train: Training features
        X_val: Validation features
        n_components: Number of components (None = auto based on variance)

    Returns:
        Tuple of (X_train_pca, X_val_pca)
    """
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    # Convert back to DataFrame
    n_comp = X_train_pca.shape[1]
    columns = [f'PC{i+1}' for i in range(n_comp)]

    X_train_pca = pd.DataFrame(X_train_pca, columns=columns, index=X_train.index)
    X_val_pca = pd.DataFrame(X_val_pca, columns=columns, index=X_val.index)

    print(f"PCA: Reduced from {X_train.shape[1]} to {n_comp} features")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    return X_train_pca, X_val_pca


def create_polynomial_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    degree: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create polynomial and interaction features

    Args:
        X_train: Training features
        X_val: Validation features
        degree: Polynomial degree

    Returns:
        Tuple of (X_train_poly, X_val_poly)
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Get feature names
    feature_names = poly.get_feature_names_out(X_train.columns)

    X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
    X_val_poly = pd.DataFrame(X_val_poly, columns=feature_names, index=X_val.index)

    return X_train_poly, X_val_poly


def engineer_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply all feature engineering steps according to config

    Args:
        X_train: Training features
        X_val: Validation features
        config: Configuration dict

    Returns:
        Tuple of (X_train_engineered, X_val_engineered)
    """
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()

    # 1. Encoding
    if config['features']['transformation']['enable']:
        methods = config['features']['transformation']['methods']

        if 'one_hot_encoding' in methods:
            X_train_proc, X_val_proc = encode_categorical(X_train_proc, X_val_proc)

    # 2. Scaling
    scaling_method = config['preprocessing'].get('scaling', 'standard')
    X_train_proc, X_val_proc = scale_features(X_train_proc, X_val_proc, scaling_method)

    # 3. PCA (if enabled)
    if config['features']['extraction']['enable']:
        if config['features']['extraction']['method'] == 'pca':
            n_components = config['features']['extraction'].get('n_components')
            X_train_proc, X_val_proc = apply_pca(X_train_proc, X_val_proc, n_components)

    return X_train_proc, X_val_proc
