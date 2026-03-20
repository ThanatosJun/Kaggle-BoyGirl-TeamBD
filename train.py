"""
Main training script for Boy or Girl Classification
Follows the workflow defined in workflow.md

Usage:
    python train.py --config config/train_config.yaml --exp_name baseline_v1
"""

import argparse
from pathlib import Path
from datetime import datetime

from src.utils import load_config, save_config, save_metrics, setup_logger, create_experiment_dir
from src.data import load_data, clean_data, create_cv_splits
from src.models import cross_validate, save_model, predict_test_set


def main(config_path: str, exp_name: str = None):
    """
    Main training pipeline

    Args:
        config_path: Path to configuration YAML file
        exp_name: Experiment name (optional, uses config if not provided)
    """
    # Load configuration
    config = load_config(config_path)

    # Setup experiment name
    if exp_name is None:
        exp_name = config['experiment']['name']

    # Create experiment directory
    exp_dir = create_experiment_dir(config['output']['results_dir'], exp_name)

    # Setup logger
    log_file = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Path('logs').mkdir(exist_ok=True)
    logger = setup_logger(log_file)

    logger.info(f"{'='*60}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"{'='*60}")

    # Save experiment config
    config_save_path = exp_dir / "config.yaml"
    save_config(config, str(config_save_path))
    logger.info(f"Configuration saved to: {config_save_path}")

    # ========== Phase 1: Data Loading & Cleaning ==========
    logger.info("\n" + "="*60)
    logger.info("Phase 1: Data Loading & Cleaning")
    logger.info("="*60)

    train_df, test_df = load_data(
        config['data']['train_path'],
        config['data'].get('test_path')
    )
    logger.info(f"Loaded training data: {train_df.shape}")
    if test_df is not None:
        logger.info(f"Loaded test data: {test_df.shape}")

    # Clean data
    train_df = clean_data(train_df, config)
    logger.info(f"After cleaning: {train_df.shape}")

    # Separate features and target
    target_col = config['data']['target_column']
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Remove ID column if present
    id_columns = ['Id', 'ID', 'id']
    for id_col in id_columns:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])
            logger.info(f"Removed ID column: {id_col}")

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    # ========== Phase 2: Cross-Validation Setup ==========
    logger.info("\n" + "="*60)
    logger.info("Phase 2: Cross-Validation Setup")
    logger.info("="*60)

    n_folds = config['training']['n_folds']
    stratified = config['training']['stratified']

    cv_splits = create_cv_splits(
        X, y,
        n_folds=n_folds,
        stratified=stratified,
        random_state=config['experiment']['seed']
    )
    logger.info(f"Created {n_folds}-fold {'stratified ' if stratified else ''}CV splits")

    # ========== Phase 3: Model Training ==========
    logger.info("\n" + "="*60)
    logger.info("Phase 3: Model Training with Cross-Validation")
    logger.info("="*60)

    model_type = config['model']['type']
    logger.info(f"Model type: {model_type}")

    cv_results = cross_validate(
        model_type=model_type,
        X=X,
        y=y,
        cv_splits=cv_splits,
        config=config,
        logger=logger
    )

    # ========== Phase 4: Save Results ==========
    logger.info("\n" + "="*60)
    logger.info("Phase 4: Saving Results")
    logger.info("="*60)

    # Save metrics
    metrics_save_path = exp_dir / "metrics.json"
    metrics_to_save = {k: v for k, v in cv_results.items() if k not in ['fold_results', 'models']}
    metrics_to_save['fold_results'] = cv_results['fold_results']  # Include individual fold results
    save_metrics(metrics_to_save, str(metrics_save_path))
    logger.info(f"Metrics saved to: {metrics_save_path}")

    # Save models
    if config['output']['save_model']:
        models_dir = exp_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for fold_idx, model in enumerate(cv_results['models']):
            model_path = models_dir / f"fold_{fold_idx}.pkl"
            save_model(model, str(model_path))

        logger.info(f"Models saved to: {models_dir}")

    # ========== Phase 5: Test Set Prediction (if available) ==========
    if test_df is not None:
        logger.info("\n" + "="*60)
        logger.info("Phase 5: Test Set Prediction")
        logger.info("="*60)

        # Preprocess test set (same steps as training, but no target)
        X_test = test_df.copy()
        test_ids = X_test[id_columns[0]] if any(col in X_test.columns for col in id_columns) else None

        for id_col in id_columns:
            if id_col in X_test.columns:
                X_test = X_test.drop(columns=[id_col])

        # Make predictions using ensemble of CV models
        predictions = predict_test_set(cv_results['models'], X_test, config)

        # Create submission file
        if config['output']['save_predictions']:
            submission_df = pd.DataFrame({
                'Id': test_ids if test_ids is not None else range(len(predictions)),
                target_col: predictions
            })

            submission_path = exp_dir / "submission.csv"
            submission_df.to_csv(submission_path, index=False)
            logger.info(f"Submission file saved to: {submission_path}")

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Boy or Girl classification model")
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name (overrides config)'
    )

    args = parser.parse_args()
    main(args.config, args.exp_name)
