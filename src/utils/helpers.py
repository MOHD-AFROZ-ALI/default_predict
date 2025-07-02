"""
Utility Functions and Helper Classes
Provides data processing utilities, validation helpers, configuration loaders, and caching
"""

import pandas as pd
import numpy as np
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from functools import wraps
import time

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Loads and manages application configuration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.config = self._load_default_config()
        self._load_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'model': {
                'random_state': 42,
                'test_size': 0.2,
                'cv_folds': 5,
                'scoring': 'roc_auc'
            },
            'data': {
                'missing_threshold': 0.5,
                'outlier_method': 'iqr',
                'scaling_method': 'standard'
            },
            'compliance': {
                'disparate_impact_threshold': 0.8,
                'significance_level': 0.05
            },
            'cache': {
                'enabled': True,
                'ttl_hours': 24,
                'max_size_mb': 100
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

    def _load_config(self) -> None:
        """Load configuration from file if exists"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(self.config, file_config)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")

    def _merge_config(self, base: Dict, override: Dict) -> None:
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def save_config(self, filepath: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            path = filepath or self.config_path
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

class DataValidator:
    """Validates data quality and integrity"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        """Validate DataFrame structure and quality"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        try:
            # Basic structure validation
            if df.empty:
                validation_results['errors'].append("DataFrame is empty")
                validation_results['is_valid'] = False
                return validation_results

            # Required columns check
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    validation_results['errors'].append(f"Missing required columns: {missing_cols}")
                    validation_results['is_valid'] = False

            # Data quality statistics
            validation_results['stats'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }

            # Quality warnings
            missing_pct = (validation_results['stats']['missing_values'] / (len(df) * len(df.columns))) * 100
            if missing_pct > 10:
                validation_results['warnings'].append(f"High missing data percentage: {missing_pct:.1f}%")

            if validation_results['stats']['duplicate_rows'] > 0:
                validation_results['warnings'].append(f"Found {validation_results['stats']['duplicate_rows']} duplicate rows")

        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False

        return validation_results

    @staticmethod
    def validate_model_inputs(X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """Validate model input data"""
        validation_results = {'is_valid': True, 'errors': [], 'warnings': []}

        try:
            # Check for infinite values
            if np.isinf(X.select_dtypes(include=[np.number])).any().any():
                validation_results['errors'].append("Infinite values found in features")
                validation_results['is_valid'] = False

            # Check for all NaN columns
            nan_cols = X.columns[X.isnull().all()].tolist()
            if nan_cols:
                validation_results['errors'].append(f"Columns with all NaN values: {nan_cols}")
                validation_results['is_valid'] = False

            # Check target variable if provided
            if y is not None:
                if y.isnull().any():
                    validation_results['errors'].append("Target variable contains missing values")
                    validation_results['is_valid'] = False

                if len(y.unique()) < 2:
                    validation_results['errors'].append("Target variable has less than 2 unique values")
                    validation_results['is_valid'] = False

        except Exception as e:
            validation_results['errors'].append(f"Input validation error: {str(e)}")
            validation_results['is_valid'] = False

        return validation_results

class CacheManager:
    """Manages caching for expensive operations"""

    def __init__(self, cache_dir: str = "/tmp/credit_model_cache", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self._cleanup_old_cache()

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cleanup_old_cache(self) -> None:
        """Remove old cache files to maintain size limit"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files) / 1024**2

            if total_size > self.max_size_mb:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                for f in cache_files[:len(cache_files)//2]:  # Remove half
                    f.unlink()
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Check if cache is still valid (24 hours)
                if datetime.now() - cached_data['timestamp'] < timedelta(hours=24):
                    return cached_data['data']
                else:
                    cache_file.unlink()  # Remove expired cache
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")

        return None

    def set(self, key: str, value: Any) -> bool:
        """Set cached value"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            cached_data = {
                'data': value,
                'timestamp': datetime.now()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)

            return True
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            return False

def cached(cache_manager: CacheManager):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_manager._get_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)
            logger.info(f"Cached result for {func.__name__}")

            return result
        return wrapper
    return decorator

class DataProcessor:
    """Common data processing utilities"""

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        df_processed = df.copy()

        for column in df_processed.columns:
            if df_processed[column].isnull().any():
                if df_processed[column].dtype in ['int64', 'float64']:
                    if strategy == 'median':
                        fill_value = df_processed[column].median()
                    elif strategy == 'mean':
                        fill_value = df_processed[column].mean()
                    else:  # mode
                        fill_value = df_processed[column].mode().iloc[0] if not df_processed[column].mode().empty else 0
                else:
                    fill_value = df_processed[column].mode().iloc[0] if not df_processed[column].mode().empty else 'Unknown'

                df_processed[column].fillna(fill_value, inplace=True)

        return df_processed

    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Detect outliers in numerical columns"""
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        for column in df.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[column] = (df[column] < lower_bound) | (df[column] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outlier_mask[column] = z_scores > threshold

        return outlier_mask

    @staticmethod
    def encode_categorical(df: pd.DataFrame, encoding_type: str = 'onehot') -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables"""
        df_encoded = df.copy()
        encoding_info = {}

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        for column in categorical_columns:
            if encoding_type == 'onehot':
                dummies = pd.get_dummies(df[column], prefix=column)
                df_encoded = pd.concat([df_encoded.drop(column, axis=1), dummies], axis=1)
                encoding_info[column] = {'type': 'onehot', 'categories': df[column].unique().tolist()}

            elif encoding_type == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df[column].astype(str))
                encoding_info[column] = {'type': 'label', 'encoder': le}

        return df_encoded, encoding_info

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Setup application logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default

def format_currency(amount: float) -> str:
    """Format number as currency"""
    return f"${amount:,.2f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage"""
    return f"{value * 100:.{decimals}f}%"

def get_feature_importance_summary(feature_names: List[str], 
                                 importances: np.ndarray, 
                                 top_n: int = 10) -> Dict[str, float]:
    """Get top N most important features"""
    feature_importance = dict(zip(feature_names, importances))
    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
