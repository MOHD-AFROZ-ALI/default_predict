"""
Credit Default Data Processing Module

This module contains the DataProcessor class responsible for loading, validating,
and preprocessing credit default data from the UCI dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data loading, validation, and preprocessing for credit default prediction.

    This class provides methods to:
    - Load UCI credit default dataset
    - Generate synthetic data if needed
    - Validate data structure and quality
    - Preprocess data for machine learning models
    """

    def __init__(self):
        """Initialize the DataProcessor with default configurations."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'default.payment.next.month'

    def load_uci_credit_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the UCI Credit Card Default dataset.

        Args:
            file_path (str, optional): Path to the dataset file. If None, generates synthetic data.

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If the specified file path doesn't exist
            ValueError: If the dataset structure is invalid
        """
        logger.info("Loading UCI Credit Card Default dataset...")

        if file_path and os.path.exists(file_path):
            try:
                # Try different file formats
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    # Assume CSV format
                    df = pd.read_csv(file_path)

                logger.info(f"Dataset loaded successfully from {file_path}")
                logger.info(f"Dataset shape: {df.shape}")

                # Validate the loaded dataset
                if self._validate_uci_structure(df):
                    return df
                else:
                    logger.warning("Dataset structure validation failed. Generating synthetic data...")
                    return self._generate_synthetic_uci_dataset()

            except Exception as e:
                logger.error(f"Error loading dataset from {file_path}: {str(e)}")
                logger.info("Generating synthetic dataset as fallback...")
                return self._generate_synthetic_uci_dataset()
        else:
            logger.info("No file path provided or file not found. Generating synthetic dataset...")
            return self._generate_synthetic_uci_dataset()

    def _generate_synthetic_uci_dataset(self, n_samples: int = 30000) -> pd.DataFrame:
        """
        Generate synthetic UCI credit default dataset with realistic distributions.

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            pd.DataFrame: Synthetic dataset matching UCI structure
        """
        logger.info(f"Generating synthetic UCI dataset with {n_samples} samples...")

        np.random.seed(42)  # For reproducibility

        # Generate synthetic data with realistic distributions
        data = {
            'ID': range(1, n_samples + 1),
            'LIMIT_BAL': np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples).astype(int),
            'SEX': np.random.choice([1, 2], size=n_samples, p=[0.4, 0.6]),  # 1=male, 2=female
            'EDUCATION': np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.37, 0.21, 0.07]),
            'MARRIAGE': np.random.choice([1, 2, 3], size=n_samples, p=[0.45, 0.53, 0.02]),
            'AGE': np.random.normal(35, 9, n_samples).clip(21, 75).astype(int),
        }

        # Payment status for 6 months (PAY_0 to PAY_6)
        for i in range(6):
            col_name = f'PAY_{i}' if i == 0 else f'PAY_{i}'
            # Payment status: -1=pay duly, 1=delay 1 month, 2=delay 2 months, etc.
            data[col_name] = np.random.choice(
                [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
                size=n_samples, 
                p=[0.6, 0.1, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005]
            )

        # Bill amounts for 6 months (BILL_AMT1 to BILL_AMT6)
        for i in range(1, 7):
            col_name = f'BILL_AMT{i}'
            # Generate bill amounts with some correlation to credit limit
            base_amount = data['LIMIT_BAL'] * np.random.uniform(0.1, 0.8, n_samples)
            noise = np.random.normal(0, base_amount * 0.2)
            data[col_name] = (base_amount + noise).clip(0, None).astype(int)

        # Payment amounts for 6 months (PAY_AMT1 to PAY_AMT6)
        for i in range(1, 7):
            col_name = f'PAY_AMT{i}'
            bill_col = f'BILL_AMT{i}'
            # Payment amounts typically less than bill amounts
            payment_ratio = np.random.beta(2, 3, n_samples)  # Skewed towards lower payments
            data[col_name] = (data[bill_col] * payment_ratio).clip(0, None).astype(int)

        # Generate target variable with realistic default rate (~22%)
        # Create some correlation with payment history and amounts
        risk_score = np.zeros(n_samples)

        # Higher risk for delayed payments
        for i in range(6):
            pay_col = f'PAY_{i}' if i == 0 else f'PAY_{i}'
            risk_score += np.where(data[pay_col] > 0, data[pay_col] * 0.1, 0)

        # Higher risk for high utilization
        utilization = np.mean([data[f'BILL_AMT{i}'] / data['LIMIT_BAL'] for i in range(1, 7)], axis=0)
        risk_score += utilization * 2

        # Add some randomness
        risk_score += np.random.normal(0, 0.5, n_samples)

        # Convert to binary default indicator
        default_threshold = np.percentile(risk_score, 78)  # ~22% default rate
        data['default.payment.next.month'] = (risk_score > default_threshold).astype(int)

        df = pd.DataFrame(data)

        logger.info(f"Synthetic dataset generated successfully")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Default rate: {df['default.payment.next.month'].mean():.2%}")

        return df

    def _validate_uci_structure(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataset has the expected UCI credit default structure.

        Args:
            df (pd.DataFrame): Dataset to validate

        Returns:
            bool: True if structure is valid, False otherwise
        """
        logger.info("Validating dataset structure...")

        # Expected columns for UCI credit default dataset
        expected_columns = [
            'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
            'default.payment.next.month'
        ]

        # Check if all expected columns are present
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            return False

        # Check data types and ranges
        validation_checks = [
            (df['SEX'].isin([1, 2]).all(), "SEX should be 1 or 2"),
            (df['EDUCATION'].isin([1, 2, 3, 4, 5, 6]).all(), "EDUCATION should be 1-6"),
            (df['MARRIAGE'].isin([0, 1, 2, 3]).all(), "MARRIAGE should be 0-3"),
            (df['AGE'].between(18, 100).all(), "AGE should be between 18-100"),
            (df['default.payment.next.month'].isin([0, 1]).all(), "Target should be 0 or 1"),
            (df['LIMIT_BAL'].min() > 0, "LIMIT_BAL should be positive"),
        ]

        for check, message in validation_checks:
            if not check:
                logger.warning(f"Validation failed: {message}")
                return False

        logger.info("Dataset structure validation passed")
        return True

    def perform_data_quality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.

        Args:
            df (pd.DataFrame): Dataset to assess

        Returns:
            Dict[str, Any]: Data quality report
        """
        logger.info("Performing data quality assessment...")

        quality_report = {
            'dataset_shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': {},
            'categorical_summary': {},
            'outliers': {},
            'data_quality_score': 0
        }

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_report['numeric_summary'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum()
            }

            # Detect outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            quality_report['outliers'][col] = outliers

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            quality_report['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].empty else None,
                'value_counts': df[col].value_counts().head().to_dict()
            }

        # Calculate overall data quality score (0-100)
        score = 100

        # Deduct points for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_ratio * 30

        # Deduct points for duplicates
        duplicate_ratio = quality_report['duplicate_rows'] / df.shape[0]
        score -= duplicate_ratio * 20

        # Deduct points for excessive outliers
        total_outliers = sum(quality_report['outliers'].values())
        outlier_ratio = total_outliers / (df.shape[0] * len(numeric_cols)) if numeric_cols.any() else 0
        score -= min(outlier_ratio * 25, 25)

        quality_report['data_quality_score'] = max(score, 0)

        logger.info(f"Data quality assessment completed. Score: {quality_report['data_quality_score']:.1f}/100")

        return quality_report

    def preprocess_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the dataset for machine learning.

        Args:
            df (pd.DataFrame): Raw dataset
            test_size (float): Proportion of dataset for testing
            random_state (int): Random state for reproducibility

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        logger.info("Starting data preprocessing...")

        # Create a copy to avoid modifying original data
        df_processed = df.copy()

        # Remove ID column if present
        if 'ID' in df_processed.columns:
            df_processed = df_processed.drop('ID', axis=1)

        # Separate features and target
        if self.target_column in df_processed.columns:
            X = df_processed.drop(self.target_column, axis=1)
            y = df_processed[self.target_column]
        else:
            logger.error(f"Target column '{self.target_column}' not found in dataset")
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].empty else 'Unknown')

        # Encode categorical variables if any
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X[col] = self.label_encoder.fit_transform(X[col].astype(str))

        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Data preprocessing completed")
        logger.info(f"Training set shape: {X_train_scaled.shape}")
        logger.info(f"Test set shape: {X_test_scaled.shape}")
        logger.info(f"Class distribution in training set: {np.bincount(y_train)}")

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    def get_feature_names(self) -> list:
        """
        Get the list of feature column names.

        Returns:
            list: Feature column names
        """
        return self.feature_columns if self.feature_columns else []

    def inverse_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.

        Args:
            X_scaled (np.ndarray): Scaled features

        Returns:
            np.ndarray: Features in original scale
        """
        return self.scaler.inverse_transform(X_scaled)
