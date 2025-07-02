"""
Configuration Management Module
==============================

This module provides centralized configuration management for the credit default prediction application.
It handles app settings, feature configurations, model parameters, and logging setup.

Functions extracted from chunks 1-2:
- get_app_config(): Main application configuration
- load_feature_config(): Feature engineering settings
- get_model_parameters(): ML model parameters
- setup_logging_config(): Logging configuration
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import yaml

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration class"""
    app_name: str = "Credit Default Prediction"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "localhost"
    port: int = 8501
    max_file_size: int = 200  # MB
    allowed_file_types: List[str] = None
    session_timeout: int = 3600  # seconds

    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.csv', '.xlsx', '.xls', '.json']


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Numerical features
    numerical_features: List[str] = None
    categorical_features: List[str] = None

    # Feature engineering parameters
    outlier_threshold: float = 3.0
    missing_value_strategy: str = "median"  # median, mean, mode, drop
    scaling_method: str = "standard"  # standard, minmax, robust

    # Feature selection
    feature_selection_method: str = "correlation"  # correlation, mutual_info, chi2
    max_features: int = 50
    correlation_threshold: float = 0.95

    # Encoding parameters
    categorical_encoding: str = "onehot"  # onehot, label, target
    handle_unknown: str = "ignore"

    def __post_init__(self):
        if self.numerical_features is None:
            self.numerical_features = [
                'loan_amount', 'annual_income', 'debt_to_income_ratio',
                'credit_score', 'employment_length', 'loan_term',
                'interest_rate', 'monthly_payment', 'total_debt'
            ]
        if self.categorical_features is None:
            self.categorical_features = [
                'loan_purpose', 'home_ownership', 'employment_status',
                'loan_grade', 'verification_status', 'state'
            ]


@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    # Model selection
    primary_model: str = "xgboost"  # xgboost, lightgbm, random_forest, logistic
    ensemble_models: List[str] = None

    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # XGBoost parameters
    xgb_params: Dict[str, Any] = None

    # LightGBM parameters
    lgb_params: Dict[str, Any] = None

    # Random Forest parameters
    rf_params: Dict[str, Any] = None

    # Logistic Regression parameters
    lr_params: Dict[str, Any] = None

    # Model evaluation
    scoring_metrics: List[str] = None
    threshold_optimization: str = "f1"  # f1, precision, recall, roc_auc

    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ["xgboost", "lightgbm", "random_forest"]

        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }

        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }

        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }

        if self.lr_params is None:
            self.lr_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'random_state': 42,
                'max_iter': 1000
            }

        if self.scoring_metrics is None:
            self.scoring_metrics = [
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                'precision_recall_auc', 'log_loss'
            ]


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: str = "credit_default_app.log"
    max_file_size: int = 10  # MB
    backup_count: int = 5
    console_logging: bool = True
    file_logging: bool = True


def get_app_config() -> AppConfig:
    """
    Get main application configuration.

    Returns:
        AppConfig: Application configuration object
    """
    # Load from environment variables if available
    config = AppConfig()

    # Override with environment variables
    config.debug = os.getenv('DEBUG', 'False').lower() == 'true'
    config.host = os.getenv('HOST', config.host)
    config.port = int(os.getenv('PORT', config.port))
    config.max_file_size = int(os.getenv('MAX_FILE_SIZE', config.max_file_size))
    config.session_timeout = int(os.getenv('SESSION_TIMEOUT', config.session_timeout))

    # Load from config file if exists
    config_file = PROJECT_ROOT / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                app_config = file_config.get('app', {})

                for key, value in app_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}")

    return config


def load_feature_config() -> FeatureConfig:
    """
    Load feature engineering configuration.

    Returns:
        FeatureConfig: Feature configuration object
    """
    config = FeatureConfig()

    # Load from config file if exists
    config_file = PROJECT_ROOT / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                feature_config = file_config.get('features', {})

                for key, value in feature_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logging.warning(f"Failed to load feature config: {e}")

    # Override with environment variables
    config.outlier_threshold = float(os.getenv('OUTLIER_THRESHOLD', config.outlier_threshold))
    config.missing_value_strategy = os.getenv('MISSING_VALUE_STRATEGY', config.missing_value_strategy)
    config.scaling_method = os.getenv('SCALING_METHOD', config.scaling_method)
    config.max_features = int(os.getenv('MAX_FEATURES', config.max_features))

    return config


def get_model_parameters() -> ModelConfig:
    """
    Get machine learning model parameters.

    Returns:
        ModelConfig: Model configuration object
    """
    config = ModelConfig()

    # Load from config file if exists
    config_file = PROJECT_ROOT / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                model_config = file_config.get('models', {})

                # Update primary model
                config.primary_model = model_config.get('primary_model', config.primary_model)
                config.ensemble_models = model_config.get('ensemble_models', config.ensemble_models)

                # Update training parameters
                training_params = model_config.get('training', {})
                for key, value in training_params.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

                # Update model-specific parameters
                if 'xgboost' in model_config:
                    config.xgb_params.update(model_config['xgboost'])
                if 'lightgbm' in model_config:
                    config.lgb_params.update(model_config['lightgbm'])
                if 'random_forest' in model_config:
                    config.rf_params.update(model_config['random_forest'])
                if 'logistic_regression' in model_config:
                    config.lr_params.update(model_config['logistic_regression'])

        except Exception as e:
            logging.warning(f"Failed to load model config: {e}")

    # Override with environment variables
    config.primary_model = os.getenv('PRIMARY_MODEL', config.primary_model)
    config.test_size = float(os.getenv('TEST_SIZE', config.test_size))
    config.random_state = int(os.getenv('RANDOM_STATE', config.random_state))

    return config


def setup_logging_config() -> LoggingConfig:
    """
    Setup logging configuration.

    Returns:
        LoggingConfig: Logging configuration object
    """
    config = LoggingConfig()

    # Load from config file if exists
    config_file = PROJECT_ROOT / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                logging_config = file_config.get('logging', {})

                for key, value in logging_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"Failed to load logging config: {e}")

    # Override with environment variables
    config.level = os.getenv('LOG_LEVEL', config.level)
    config.log_file = os.getenv('LOG_FILE', config.log_file)
    config.console_logging = os.getenv('CONSOLE_LOGGING', 'True').lower() == 'true'
    config.file_logging = os.getenv('FILE_LOGGING', 'True').lower() == 'true'

    # Configure logging
    configure_logging(config)

    return config


def configure_logging(config: LoggingConfig):
    """
    Configure the logging system based on the provided configuration.

    Args:
        config (LoggingConfig): Logging configuration object
    """
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Set logging level
    level = getattr(logging, config.level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt=config.format,
        datefmt=config.date_format
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    if config.console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler
    if config.file_logging:
        from logging.handlers import RotatingFileHandler

        log_file_path = LOGS_DIR / config.log_file
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=config.max_file_size * 1024 * 1024,  # Convert MB to bytes
            backupCount=config.backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def save_config_template():
    """
    Save a template configuration file for reference.
    """
    template_config = {
        'app': {
            'app_name': 'Credit Default Prediction',
            'version': '1.0.0',
            'debug': False,
            'host': 'localhost',
            'port': 8501,
            'max_file_size': 200,
            'session_timeout': 3600
        },
        'features': {
            'outlier_threshold': 3.0,
            'missing_value_strategy': 'median',
            'scaling_method': 'standard',
            'feature_selection_method': 'correlation',
            'max_features': 50,
            'correlation_threshold': 0.95,
            'categorical_encoding': 'onehot'
        },
        'models': {
            'primary_model': 'xgboost',
            'ensemble_models': ['xgboost', 'lightgbm', 'random_forest'],
            'training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            },
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'credit_default_app.log',
            'max_file_size': 10,
            'backup_count': 5,
            'console_logging': True,
            'file_logging': True
        }
    }

    config_file = PROJECT_ROOT / "config_template.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(template_config, f, default_flow_style=False, indent=2)

    return config_file


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration (if needed for future extensions).

    Returns:
        Dict[str, Any]: Database configuration
    """
    return {
        'type': 'sqlite',
        'path': str(DATA_DIR / 'credit_default.db'),
        'echo': False,
        'pool_size': 5,
        'max_overflow': 10
    }


def validate_config() -> bool:
    """
    Validate all configuration settings.

    Returns:
        bool: True if all configurations are valid
    """
    try:
        # Test all configuration functions
        app_config = get_app_config()
        feature_config = load_feature_config()
        model_config = get_model_parameters()
        logging_config = setup_logging_config()

        # Basic validation checks
        assert app_config.port > 0, "Port must be positive"
        assert app_config.max_file_size > 0, "Max file size must be positive"
        assert 0 < model_config.test_size < 1, "Test size must be between 0 and 1"
        assert feature_config.outlier_threshold > 0, "Outlier threshold must be positive"

        logging.info("Configuration validation successful")
        return True

    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False


# Initialize logging on module import
if __name__ != "__main__":
    setup_logging_config()


# Export main functions and classes
__all__ = [
    'AppConfig', 'FeatureConfig', 'ModelConfig', 'LoggingConfig',
    'get_app_config', 'load_feature_config', 'get_model_parameters', 
    'setup_logging_config', 'configure_logging', 'save_config_template',
    'get_database_config', 'validate_config'
]
