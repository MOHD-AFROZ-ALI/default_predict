
from enum import Enum
from typing import Dict, List, Tuple, Any

# ============================================================================
# APPLICATION METADATA
# ============================================================================

APP_NAME = "Credit Default Prediction System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Machine Learning application for predicting credit default risk"
APP_AUTHOR = "Credit Risk Analytics Team"
APP_LICENSE = "MIT"

# ============================================================================
# FILE AND DATA CONSTANTS
# ============================================================================

# File handling
MAX_FILE_SIZE_MB = 200
SUPPORTED_FILE_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.feather']
DEFAULT_ENCODING = 'utf-8'
CHUNK_SIZE = 10000  # For processing large files
MAX_ROWS_PREVIEW = 1000  # Maximum rows to show in data preview

# Data quality thresholds
MIN_DATA_COMPLETENESS = 0.8  # 80% completeness required
MAX_DUPLICATE_RATE = 0.05    # 5% duplicates allowed
MAX_OUTLIER_RATE = 0.1       # 10% outliers allowed
MIN_UNIQUE_VALUES = 2        # Minimum unique values for categorical features

# ============================================================================
# MACHINE LEARNING CONSTANTS
# ============================================================================

# Model training defaults
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_SCORING = 'roc_auc'

# Feature engineering
OUTLIER_THRESHOLD = 3.0
CORRELATION_THRESHOLD = 0.95
MAX_FEATURES = 50
MIN_FEATURE_IMPORTANCE = 0.001
FEATURE_SELECTION_THRESHOLD = 0.01

# Model performance thresholds
MIN_ACCURACY_THRESHOLD = 0.75
MIN_PRECISION_THRESHOLD = 0.70
MIN_RECALL_THRESHOLD = 0.65
MIN_F1_THRESHOLD = 0.70
MIN_AUC_THRESHOLD = 0.80
MIN_PRECISION_RECALL_AUC = 0.75

# ============================================================================
# BUSINESS RULES AND RISK THRESHOLDS
# ============================================================================

# Risk probability thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4
LOW_RISK_THRESHOLD = 0.2
VERY_HIGH_RISK_THRESHOLD = 0.85

# Credit score ranges
CREDIT_SCORE_RANGES = {
    'excellent': (750, 850),
    'very_good': (700, 749),
    'good': (650, 699),
    'fair': (600, 649),
    'poor': (550, 599),
    'very_poor': (300, 549)
}

# Loan amount ranges (in thousands USD)
LOAN_AMOUNT_RANGES = {
    'micro': (0, 5),
    'small': (5, 15),
    'medium': (15, 35),
    'large': (35, 75),
    'very_large': (75, 150),
    'jumbo': (150, float('inf'))
}

# Annual income ranges (in thousands USD)
INCOME_RANGES = {
    'very_low': (0, 25),
    'low': (25, 40),
    'lower_middle': (40, 60),
    'middle': (60, 80),
    'upper_middle': (80, 120),
    'high': (120, 200),
    'very_high': (200, float('inf'))
}

# Debt-to-Income ratio categories
DTI_CATEGORIES = {
    'excellent': (0, 0.15),
    'very_good': (0.15, 0.25),
    'good': (0.25, 0.35),
    'acceptable': (0.35, 0.43),
    'risky': (0.43, 0.50),
    'very_risky': (0.50, float('inf'))
}

# Employment length categories (in years)
EMPLOYMENT_LENGTH_CATEGORIES = {
    'new': (0, 1),
    'short': (1, 3),
    'medium': (3, 7),
    'long': (7, 15),
    'very_long': (15, float('inf'))
}

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Core numerical features
NUMERICAL_FEATURES = [
    'loan_amount',
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'employment_length',
    'loan_term',
    'interest_rate',
    'monthly_payment',
    'total_debt',
    'credit_utilization',
    'number_of_accounts',
    'delinquency_2yrs',
    'inquiries_6m',
    'open_accounts',
    'public_records',
    'revolving_balance',
    'revolving_utilization',
    'total_accounts',
    'months_since_last_delinq',
    'months_since_last_record',
    'collections_12_mths_ex_med',
    'policy_code',
    'acc_now_delinq',
    'tot_coll_amt',
    'tot_cur_bal',
    'total_rev_hi_lim'
]

# Core categorical features
CATEGORICAL_FEATURES = [
    'loan_purpose',
    'home_ownership',
    'employment_status',
    'loan_grade',
    'loan_sub_grade',
    'verification_status',
    'state',
    'application_type',
    'joint_application',
    'initial_list_status',
    'hardship_flag',
    'disbursement_method',
    'debt_settlement_flag'
]

# Derived/engineered features
DERIVED_FEATURES = [
    'income_to_loan_ratio',
    'monthly_income',
    'debt_service_ratio',
    'credit_age_years',
    'account_utilization_ratio',
    'payment_to_income_ratio',
    'loan_to_value_ratio',
    'credit_mix_score',
    'payment_history_score',
    'credit_length_score',
    'new_credit_score',
    'total_credit_limit',
    'available_credit',
    'credit_usage_trend'
]

# High-importance features (based on typical model analysis)
HIGH_IMPORTANCE_FEATURES = [
    'credit_score',
    'debt_to_income_ratio',
    'annual_income',
    'loan_amount',
    'interest_rate',
    'loan_grade',
    'employment_length',
    'delinquency_2yrs',
    'revolving_utilization',
    'inquiries_6m'
]

# Features requiring special handling
FEATURES_WITH_MISSING_VALUES = [
    'employment_length',
    'months_since_last_delinq',
    'months_since_last_record',
    'revolving_utilization',
    'collections_12_mths_ex_med'
]

# ============================================================================
# ENUMERATION CLASSES
# ============================================================================

class LoanPurpose(Enum):
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    CAR = "car"
    VACATION = "vacation"
    WEDDING = "wedding"
    MOVING = "moving"
    HOUSE = "house"
    RENEWABLE_ENERGY = "renewable_energy"
    SMALL_BUSINESS = "small_business"
    EDUCATIONAL = "educational"
    OTHER = "other"


class HomeOwnership(Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"
    NONE = "NONE"
    ANY = "ANY"


class EmploymentStatus(Enum):
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"
    STUDENT = "student"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    OTHER = "other"


class LoanGrade(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class LoanSubGrade(Enum):
    A1 = "A1"; A2 = "A2"; A3 = "A3"; A4 = "A4"; A5 = "A5"
    B1 = "B1"; B2 = "B2"; B3 = "B3"; B4 = "B4"; B5 = "B5"
    C1 = "C1"; C2 = "C2"; C3 = "C3"; C4 = "C4"; C5 = "C5"
    D1 = "D1"; D2 = "D2"; D3 = "D3"; D4 = "D4"; D5 = "D5"
    E1 = "E1"; E2 = "E2"; E3 = "E3"; E4 = "E4"; E5 = "E5"
    F1 = "F1"; F2 = "F2"; F3 = "F3"; F4 = "F4"; F5 = "F5"
    G1 = "G1"; G2 = "G2"; G3 = "G3"; G4 = "G4"; G5 = "G5"


class VerificationStatus(Enum):
    VERIFIED = "Verified"
    SOURCE_VERIFIED = "Source Verified"
    NOT_VERIFIED = "Not Verified"


class ApplicationType(Enum):
    INDIVIDUAL = "Individual"
    JOINT = "Joint App"


class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class FeatureType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    TEXT = "text"
    DATETIME = "datetime"


# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# XGBoost parameters
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'random_state': DEFAULT_RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'random_state': DEFAULT_RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1
}

# Random Forest parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'oob_score': True,
    'random_state': DEFAULT_RANDOM_STATE,
    'n_jobs': -1,
    'verbose': 0
}

# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'liblinear',
    'random_state': DEFAULT_RANDOM_STATE,
    'max_iter': 1000,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'class_weight': None,
    'dual': False,
    'warm_start': False
}

# Gradient Boosting parameters
GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'subsample': 1.0,
    'max_features': None,
    'random_state': DEFAULT_RANDOM_STATE,
    'verbose': 0
}

# Neural Network parameters
NEURAL_NETWORK_PARAMS = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'max_iter': 200,
    'shuffle': True,
    'random_state': DEFAULT_RANDOM_STATE,
    'tol': 0.0001,
    'verbose': False,
    'warm_start': False,
    'momentum': 0.9,
    'nesterovs_momentum': True,
    'early_stopping': False,
    'validation_fraction': 0.1,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-08
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Classification metrics
CLASSIFICATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'precision_recall_auc',
    'log_loss',
    'matthews_corrcoef',
    'balanced_accuracy',
    'cohen_kappa',
    'hamming_loss',
    'jaccard_score'
]

# Business metrics
BUSINESS_METRICS = [
    'profit',
    'cost_savings',
    'approval_rate',
    'default_rate',
    'expected_loss',
    'return_on_investment',
    'risk_adjusted_return'
]

# Model monitoring metrics
MONITORING_METRICS = [
    'prediction_drift',
    'feature_drift',
    'model_performance_decay',
    'data_quality_score',
    'prediction_confidence'
]

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

# Plot styling
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_LARGE = (16, 10)
DPI = 300
FONT_SIZE = 12
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 10

# Color palettes
COLOR_PALETTE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]

# Risk-specific colors
RISK_COLORS = {
    'very_low': '#00ff00',    # Bright Green
    'low': '#2ca02c',         # Green
    'medium': '#ff7f0e',      # Orange
    'high': '#d62728',        # Red
    'very_high': '#8b0000'    # Dark Red
}

# Grade-specific colors
GRADE_COLORS = {
    'A': '#00ff00',  # Green
    'B': '#7fff00',  # Chartreuse
    'C': '#ffff00',  # Yellow
    'D': '#ffa500',  # Orange
    'E': '#ff4500',  # Red Orange
    'F': '#ff0000',  # Red
    'G': '#8b0000'   # Dark Red
}

# Feature importance colors
IMPORTANCE_COLORS = {
    'high': '#d62728',      # Red
    'medium': '#ff7f0e',    # Orange
    'low': '#2ca02c',       # Green
    'negligible': '#7f7f7f' # Gray
}

# ============================================================================
# UI AND INTERFACE CONSTANTS
# ============================================================================

# Streamlit UI settings
SIDEBAR_WIDTH = 300
MAIN_CONTENT_WIDTH = 800
CHART_HEIGHT = 400
CHART_WIDTH = 600
TABLE_PAGE_SIZE = 50
MAX_DISPLAY_ROWS = 100

# Progress indicators
PROGRESS_STEPS = {
    'data_loading': 10,
    'data_preprocessing': 30,
    'feature_engineering': 50,
    'model_training': 80,
    'model_evaluation': 90,
    'report_generation': 100
}

# Status messages
STATUS_MESSAGES = {
    'loading': '🔄 Loading data...',
    'processing': '⚙️ Processing...',
    'training': '🤖 Training model...',
    'predicting': '🔮 Making predictions...',
    'success': '✅ Operation completed successfully!',
    'error': '❌ An error occurred',
    'warning': '⚠️ Warning',
    'info': 'ℹ️ Information'
}

# ============================================================================
# API AND SYSTEM CONSTANTS
# ============================================================================

# API settings
API_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
REQUEST_RATE_LIMIT = 100  # requests per minute

# Cache settings
CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 100  # Maximum number of cached items
CACHE_CLEANUP_INTERVAL = 1800  # 30 minutes

# Logging constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Database constants (for future use)
DB_CONNECTION_TIMEOUT = 30
DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10
DB_ECHO = False

# ============================================================================
# BUSINESS IMPACT CONSTANTS
# ============================================================================

# Cost-benefit analysis
COST_OF_FALSE_POSITIVE = 1000  # Cost of rejecting a good loan
COST_OF_FALSE_NEGATIVE = 5000  # Cost of approving a bad loan
PROFIT_PER_GOOD_LOAN = 2000    # Profit from a good loan
AVERAGE_LOAN_PROCESSING_COST = 100  # Cost to process each loan

# Interest rates by grade
INTEREST_RATES_BY_GRADE = {
    'A': (5.32, 9.44),
    'B': (9.44, 12.29),
    'C': (12.29, 15.59),
    'D': (15.59, 18.94),
    'E': (18.94, 21.49),
    'F': (21.49, 25.89),
    'G': (25.89, 30.99)
}

# Default rates by grade (historical averages)
DEFAULT_RATES_BY_GRADE = {
    'A': 0.02,  # 2%
    'B': 0.04,  # 4%
    'C': 0.08,  # 8%
    'D': 0.13,  # 13%
    'E': 0.20,  # 20%
    'F': 0.28,  # 28%
    'G': 0.35   # 35%
}

# ============================================================================
# FEATURE IMPORTANCE THRESHOLDS
# ============================================================================

FEATURE_IMPORTANCE_THRESHOLDS = {
    'critical': 0.15,    # Features with >15% importance
    'high': 0.10,        # Features with 10-15% importance
    'medium': 0.05,      # Features with 5-10% importance
    'low': 0.01,         # Features with 1-5% importance
    'negligible': 0.001  # Features with <1% importance
}

# SHAP value interpretation thresholds
SHAP_THRESHOLDS = {
    'strong_positive': 0.5,
    'moderate_positive': 0.2,
    'weak_positive': 0.05,
    'neutral': 0.05,
    'weak_negative': -0.05,
    'moderate_negative': -0.2,
    'strong_negative': -0.5
}

# ============================================================================
# MODEL MONITORING CONSTANTS
# ============================================================================

# Drift detection thresholds
MODEL_DRIFT_THRESHOLD = 0.1
FEATURE_DRIFT_THRESHOLD = 0.05
PERFORMANCE_DEGRADATION_THRESHOLD = 0.05
DATA_QUALITY_THRESHOLD = 0.8

# Retraining triggers
RETRAINING_THRESHOLD = 0.1
MIN_SAMPLES_FOR_RETRAINING = 1000
RETRAINING_FREQUENCY_DAYS = 30

# Alert thresholds
ALERT_THRESHOLDS = {
    'performance_drop': 0.05,
    'drift_detected': 0.1,
    'data_quality_issue': 0.2,
    'prediction_anomaly': 0.15
}

# ============================================================================
# STATE AND REGION CONSTANTS
# ============================================================================

# US States (for geographic analysis)
US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

# High-risk states (based on historical data)
HIGH_RISK_STATES = ['NV', 'FL', 'CA', 'AZ', 'MI']

# Low-risk states
LOW_RISK_STATES = ['ND', 'SD', 'WY', 'VT', 'NH']

# ============================================================================
# EXPORT ALL CONSTANTS
# ============================================================================

__all__ = [
    # Application metadata
    'APP_NAME', 'APP_VERSION', 'APP_DESCRIPTION', 'APP_AUTHOR', 'APP_LICENSE',

    # File and data constants
    'MAX_FILE_SIZE_MB', 'SUPPORTED_FILE_FORMATS', 'DEFAULT_ENCODING', 'CHUNK_SIZE',
    'MIN_DATA_COMPLETENESS', 'MAX_DUPLICATE_RATE', 'MAX_OUTLIER_RATE',

    # ML constants
    'DEFAULT_RANDOM_STATE', 'DEFAULT_TEST_SIZE', 'DEFAULT_VALIDATION_SIZE',
    'DEFAULT_CV_FOLDS', 'OUTLIER_THRESHOLD', 'CORRELATION_THRESHOLD',

    # Thresholds
    'MIN_ACCURACY_THRESHOLD', 'MIN_PRECISION_THRESHOLD', 'MIN_RECALL_THRESHOLD',
    'HIGH_RISK_THRESHOLD', 'MEDIUM_RISK_THRESHOLD', 'LOW_RISK_THRESHOLD',

    # Ranges and categories
    'CREDIT_SCORE_RANGES', 'LOAN_AMOUNT_RANGES', 'INCOME_RANGES', 'DTI_CATEGORIES',
    'EMPLOYMENT_LENGTH_CATEGORIES',

    # Feature lists
    'NUMERICAL_FEATURES', 'CATEGORICAL_FEATURES', 'DERIVED_FEATURES',
    'HIGH_IMPORTANCE_FEATURES', 'FEATURES_WITH_MISSING_VALUES',

    # Enums
    'LoanPurpose', 'HomeOwnership', 'EmploymentStatus', 'LoanGrade', 'LoanSubGrade',
    'VerificationStatus', 'ApplicationType', 'RiskLevel', 'ModelType', 'FeatureType',

    # Model parameters
    'XGBOOST_PARAMS', 'LIGHTGBM_PARAMS', 'RANDOM_FOREST_PARAMS',
    'LOGISTIC_REGRESSION_PARAMS', 'GRADIENT_BOOSTING_PARAMS', 'NEURAL_NETWORK_PARAMS',

    # Metrics
    'CLASSIFICATION_METRICS', 'BUSINESS_METRICS', 'MONITORING_METRICS',

    # Visualization
    'PLOT_STYLE', 'FIGURE_SIZE', 'COLOR_PALETTE', 'RISK_COLORS', 'GRADE_COLORS',

    # UI constants
    'SIDEBAR_WIDTH', 'MAIN_CONTENT_WIDTH', 'CHART_HEIGHT', 'CHART_WIDTH',
    'TABLE_PAGE_SIZE', 'PROGRESS_STEPS', 'STATUS_MESSAGES',

    # System constants
    'API_TIMEOUT', 'MAX_RETRIES', 'CACHE_TTL', 'LOG_FORMAT',

    # Business constants
    'COST_OF_FALSE_POSITIVE', 'COST_OF_FALSE_NEGATIVE', 'PROFIT_PER_GOOD_LOAN',
    'INTEREST_RATES_BY_GRADE', 'DEFAULT_RATES_BY_GRADE',

    # Feature importance and monitoring
    'FEATURE_IMPORTANCE_THRESHOLDS', 'SHAP_THRESHOLDS', 'MODEL_DRIFT_THRESHOLD',
    'ALERT_THRESHOLDS',

    # Geographic constants
    'US_STATES', 'HIGH_RISK_STATES', 'LOW_RISK_STATES'
]
