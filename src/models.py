"""
Model training, evaluation, and prediction functions
Corresponds to Phase 3-4 of workflow.md
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, Tuple
import pickle


def create_model(model_type: str, config: dict):
    """
    Create model instance based on config

    Args:
        model_type: Model type ('random_forest', 'xgboost', 'logistic_regression')
        config: Configuration dict

    Returns:
        Model instance
    """
    model_config = config['model'][model_type]

    if model_type == 'random_forest':
        model = RandomForestClassifier(**model_config, random_state=config['experiment']['seed'])

    elif model_type == 'logistic_regression':
        model = LogisticRegression(**model_config, random_state=config['experiment']['seed'])

    elif model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(**model_config, random_state=config['experiment']['seed'])
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

    elif model_type == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**model_config, random_state=config['experiment']['seed'])
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a single model

    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """
    Evaluate model on validation set

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1_score': f1_score(y_val, y_pred, zero_division=0),
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)

    return metrics


def cross_validate(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    config: dict,
    logger
) -> Dict[str, Any]:
    """
    Perform cross-validation training

    Args:
        model_type: Type of model to train
        X: Full feature set
        y: Full labels
        cv_splits: List of (train_idx, val_idx) tuples
        config: Configuration dict
        logger: Logger instance

    Returns:
        Dictionary containing fold results and summary statistics
    """
    from src.data import impute_missing_values
    from src.features import engineer_features

    fold_results = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold_idx + 1}/{len(cv_splits)}")
        logger.info(f"{'='*60}")

        # Split data
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()

        # Remove target column if present
        target_col = config['data'].get('target_column', 'Gender')
        if target_col in X_train_fold.columns:
            X_train_fold = X_train_fold.drop(columns=[target_col])
            X_val_fold = X_val_fold.drop(columns=[target_col])

        # Impute missing values (fit on train, transform on val)
        X_train_fold, X_val_fold = impute_missing_values(X_train_fold, X_val_fold, config)

        # Feature engineering (fit on train, transform on val)
        X_train_fold, X_val_fold = engineer_features(X_train_fold, X_val_fold, config)

        # Create and train model
        model = create_model(model_type, config)
        model = train_model(model, X_train_fold, y_train_fold)

        # Evaluate
        metrics = evaluate_model(model, X_val_fold, y_val_fold)

        logger.info(f"Fold {fold_idx + 1} Results:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        fold_results.append(metrics)
        fold_models.append(model)

    # Aggregate results
    cv_results = aggregate_cv_results(fold_results, logger)
    cv_results['fold_results'] = fold_results
    cv_results['models'] = fold_models

    return cv_results


def aggregate_cv_results(fold_results: list, logger) -> Dict[str, Any]:
    """
    Aggregate cross-validation results

    Args:
        fold_results: List of metric dicts from each fold
        logger: Logger instance

    Returns:
        Dictionary with mean and std for each metric
    """
    metrics = {}
    metric_names = fold_results[0].keys()

    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'='*60}")

    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)

        metrics[f'{metric_name}_mean'] = mean_val
        metrics[f'{metric_name}_std'] = std_val

        logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    return metrics


def save_model(model, filepath: str):
    """Save model to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str):
    """Load model from file"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_test_set(models: list, X_test: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Make predictions on test set using ensemble of CV models

    Args:
        models: List of trained models from each fold
        X_test: Test features
        config: Configuration dict

    Returns:
        Array of predictions
    """
    predictions = []

    for model in models:
        pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        predictions.append(pred)

    # Ensemble: average predictions
    avg_predictions = np.mean(predictions, axis=0)
    final_predictions = (avg_predictions > 0.5).astype(int)

    return final_predictions
