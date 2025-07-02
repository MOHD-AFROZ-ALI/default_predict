import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading utilities for credit default prediction application."""

    def __init__(self, data_dir: str = "data"):
        """Initialize DataLoader with data directory path."""
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']

    def load_uci_credit_dataset(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load UCI credit default dataset from various file formats.

        Args:
            file_path: Path to the dataset file

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}. Supported: {self.supported_formats}")

        try:
            if file_ext == '.csv':
                df = self._load_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            elif file_ext == '.json':
                df = self._load_json(file_path)

            logger.info(f"Successfully loaded dataset: {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with proper encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"CSV loaded with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue

        # Fallback to default pandas behavior
        return pd.read_csv(file_path)

    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(file_path)

    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load JSON file and convert to DataFrame."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("JSON data must be a list or dictionary")

    def validate_credit_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate credit dataset structure and content.

        Args:
            df: DataFrame to validate

        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }

        # Check for empty dataset
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results

        # Check for expected UCI credit dataset columns
        expected_columns = [
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
            'default.payment.next.month'
        ]

        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            validation_results['warnings'].append(f"Missing expected columns: {missing_cols}")

        # Check for negative values in amount columns
        amount_cols = [col for col in df.columns if 'AMT' in col or 'LIMIT_BAL' in col]
        for col in amount_cols:
            if col in df.columns and (df[col] < 0).any():
                validation_results['warnings'].append(f"Negative values found in {col}")

        # Check target variable
        target_col = 'default.payment.next.month'
        if target_col in df.columns:
            unique_targets = df[target_col].unique()
            if not set(unique_targets).issubset({0, 1}):
                validation_results['errors'].append(f"Target variable should contain only 0 and 1, found: {unique_targets}")
                validation_results['is_valid'] = False

        # Check for excessive missing values
        missing_threshold = 0.5
        high_missing_cols = [col for col, missing_pct in validation_results['missing_values'].items() 
                           if missing_pct / len(df) > missing_threshold]
        if high_missing_cols:
            validation_results['warnings'].append(f"Columns with >50% missing values: {high_missing_cols}")

        return validation_results

    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict containing dataset information
        """
        info = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_types': df.dtypes.value_counts().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'missing_data_summary': {
                'total_missing': df.isnull().sum().sum(),
                'columns_with_missing': df.columns[df.isnull().any()].tolist(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            'basic_stats': df.describe().to_dict() if not df.empty else {}
        }

        return info

    def save_dataset(self, df: pd.DataFrame, file_path: Union[str, Path], 
                    format_type: str = 'csv') -> bool:
        """
        Save dataset to specified format.

        Args:
            df: DataFrame to save
            file_path: Output file path
            format_type: Format to save ('csv', 'excel', 'json')

        Returns:
            bool: Success status
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format_type.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif format_type.lower() == 'excel':
                df.to_excel(file_path, index=False)
            elif format_type.lower() == 'json':
                df.to_json(file_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            logger.info(f"Dataset saved to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            return False

# Utility functions
def load_credit_data(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to load and validate credit dataset.

    Args:
        file_path: Path to dataset file

    Returns:
        Tuple of (DataFrame, validation_results)
    """
    loader = DataLoader()
    df = loader.load_uci_credit_dataset(file_path)
    validation = loader.validate_credit_dataset(df)

    return df, validation

def get_sample_data() -> pd.DataFrame:
    """
    Generate sample credit dataset for testing purposes.

    Returns:
        pd.DataFrame: Sample dataset
    """
    np.random.seed(42)
    n_samples = 1000

    data = {
        'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),
        'SEX': np.random.choice([1, 2], n_samples),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples),
        'MARRIAGE': np.random.choice([1, 2, 3], n_samples),
        'AGE': np.random.randint(21, 80, n_samples),
        'PAY_0': np.random.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'PAY_2': np.random.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'BILL_AMT1': np.random.randint(0, 100000, n_samples),
        'PAY_AMT1': np.random.randint(0, 50000, n_samples),
        'default.payment.next.month': np.random.choice([0, 1], n_samples, p=[0.78, 0.22])
    }

    return pd.DataFrame(data)
