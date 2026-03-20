"""
Utility functions for configuration, logging, and metrics
"""

import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def load_config(config_path: str = "config/train_config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """
    Save metrics to JSON file

    Args:
        metrics: Metrics dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger('BoyGirl')
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_experiment_dir(base_dir: str, exp_name: str) -> Path:
    """
    Create experiment directory with timestamp

    Args:
        base_dir: Base results directory
        exp_name: Experiment name

    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
