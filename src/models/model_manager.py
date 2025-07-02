"""
Credit Default Model Management Module

This module contains the ModelManager class responsible for training, evaluating,
and managing multiple machine learning models for credit default prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages multiple machine learning models for credit default prediction.

    This class provides methods to:
    - Initialize and configure multiple ML models
    - Train models with hyperparameter tuning
    - Evaluate model performance with comprehensive metrics
    - Extract feature importance and interpretability
    - Make predictions on new data
    - Save and load trained models
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelManager with default model configurations.

        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_metrics = {}
        self.feature_names = None
        self.best_model_name = None
        self.best_model = None

        # Initialize models with default configurations
        self.initialize_models()

        logger.info("ModelManager initialized with default model configurations")

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize multiple machine learning models with optimized configurations.

        Returns:
            Dict[str, Any]: Dictionary of initialized models
        """
        logger.info("Initializing machine learning models...")

        # Define model configurations optimized for credit default prediction
        model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },

            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=self.random_state
                ),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },

            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    solver='liblinear'
                ),
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },

            'SVM': {
                'model': SVC(
                    random_state=self.random_state,
                    probability=True,
                    kernel='rbf'
                ),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },

            'KNeighbors': {
                'model': KNeighborsClassifier(
                    n_neighbors=5,
                    weights='uniform'
                ),
                'param_grid': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },

            'NaiveBayes': {
                'model': GaussianNB(),
                'param_grid': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                }
            },

            'DecisionTree': {
                'model': DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10,
                    min_samples_split=5
                ),
                'param_grid': {
                    'max_depth': [3, 5, 10, 15, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            }
        }

        # Store model configurations
        self.models = model_configs

        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")

        return self.models

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray,
                   feature_names: Optional[List[str]] = None,
                   use_grid_search: bool = True,
                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train all models and evaluate their performance.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            feature_names (List[str], optional): Names of features
            use_grid_search (bool): Whether to use hyperparameter tuning
            cv_folds (int): Number of cross-validation folds

        Returns:
            Dict[str, Any]: Training results and model performance metrics
        """
        logger.info("Starting model training process...")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Using grid search: {use_grid_search}")

        # Store feature names
        self.feature_names = feature_names

        training_results = {
            'models_trained': [],
            'training_time': {},
            'best_params': {},
            'cv_scores': {},
            'test_metrics': {}
        }

        # Train each model
        for model_name, model_config in self.models.items():
            logger.info(f"Training {model_name}...")

            try:
                start_time = datetime.now()

                if use_grid_search and len(model_config['param_grid']) > 0:
                    # Hyperparameter tuning with GridSearchCV
                    grid_search = GridSearchCV(
                        model_config['model'],
                        model_config['param_grid'],
                        cv=cv_folds,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=0
                    )

                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    cv_score = grid_search.best_score_

                    logger.info(f"{model_name} - Best CV Score: {cv_score:.4f}")
                    logger.info(f"{model_name} - Best Params: {best_params}")

                else:
                    # Train with default parameters
                    best_model = model_config['model']
                    best_model.fit(X_train, y_train)
                    best_params = {}

                    # Calculate cross-validation score
                    cv_scores = cross_val_score(
                        best_model, X_train, y_train, 
                        cv=cv_folds, scoring='roc_auc'
                    )
                    cv_score = cv_scores.mean()

                # Store trained model
                self.trained_models[model_name] = best_model

                # Calculate training time
                training_time = (datetime.now() - start_time).total_seconds()

                # Evaluate on test set
                test_metrics = self._calculate_metrics(
                    best_model, X_test, y_test, model_name
                )

                # Store results
                training_results['models_trained'].append(model_name)
                training_results['training_time'][model_name] = training_time
                training_results['best_params'][model_name] = best_params
                training_results['cv_scores'][model_name] = cv_score
                training_results['test_metrics'][model_name] = test_metrics

                # Store model metrics for comparison
                self.model_metrics[model_name] = test_metrics

                logger.info(f"{model_name} training completed in {training_time:.2f}s")
                logger.info(f"{model_name} - Test AUC: {test_metrics['roc_auc']:.4f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue

        # Identify best model based on ROC AUC
        if self.model_metrics:
            best_model_name = max(
                self.model_metrics.keys(),
                key=lambda x: self.model_metrics[x]['roc_auc']
            )
            self.best_model_name = best_model_name
            self.best_model = self.trained_models[best_model_name]

            logger.info(f"Best model: {best_model_name} (AUC: {self.model_metrics[best_model_name]['roc_auc']:.4f})")

        # Summary
        logger.info(f"Model training completed. Trained {len(training_results['models_trained'])} models")

        return training_results

    def _calculate_metrics(self, model: Any, X_test: np.ndarray, 
                          y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a trained model.

        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of the model

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate basic metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1_score': f1_score(y_test, y_pred, average='binary'),
                'specificity': 0.0,  # Will calculate below
                'roc_auc': 0.0  # Will calculate below if probabilities available
            }

            # Calculate confusion matrix for specificity
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Calculate ROC AUC if probabilities are available
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            else:
                # Use decision function for SVM without probability
                if hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_test, decision_scores)
                else:
                    metrics['roc_auc'] = 0.0

            # Additional metrics
            metrics['true_positives'] = int(cm[1, 1]) if cm.shape == (2, 2) else 0
            metrics['false_positives'] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
            metrics['true_negatives'] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
            metrics['false_negatives'] = int(cm[1, 0]) if cm.shape == (2, 2) else 0

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'specificity': 0.0, 'roc_auc': 0.0, 'true_positives': 0,
                'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0
            }

    def _get_feature_importance(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Extract feature importance from trained models.

        Args:
            model_name (str, optional): Specific model name. If None, uses best model.

        Returns:
            Dict[str, float]: Feature importance scores
        """
        # Use best model if no specific model requested
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)

        if model is None:
            logger.warning(f"Model {model_name} not found or not trained")
            return {}

        try:
            feature_importance = {}

            # Extract feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (RandomForest, GradientBoosting, DecisionTree)
                importances = model.feature_importances_

            elif hasattr(model, 'coef_'):
                # Linear models (LogisticRegression, SVM with linear kernel)
                importances = np.abs(model.coef_[0])

            else:
                logger.warning(f"Feature importance not available for {model_name}")
                return {}

            # Create feature importance dictionary
            if self.feature_names and len(self.feature_names) == len(importances):
                feature_importance = dict(zip(self.feature_names, importances))
            else:
                feature_importance = {f'feature_{i}': imp for i, imp in enumerate(importances)}

            # Sort by importance (descending)
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            logger.info(f"Feature importance extracted for {model_name}")
            logger.info(f"Top 5 features: {list(feature_importance.keys())[:5]}")

            return feature_importance

        except Exception as e:
            logger.error(f"Error extracting feature importance for {model_name}: {str(e)}")
            return {}

    def predict_single(self, features: Union[np.ndarray, List[float]], 
                      model_name: Optional[str] = None,
                      return_probability: bool = True) -> Dict[str, Any]:
        """
        Make prediction for a single instance.

        Args:
            features (Union[np.ndarray, List[float]]): Feature values for prediction
            model_name (str, optional): Specific model to use. If None, uses best model.
            return_probability (bool): Whether to return prediction probabilities

        Returns:
            Dict[str, Any]: Prediction results including class and probability
        """
        # Use best model if no specific model requested
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)

        if model is None:
            logger.error(f"Model {model_name} not found or not trained")
            return {'error': f'Model {model_name} not available'}

        try:
            # Ensure features are in correct format
            if isinstance(features, list):
                features = np.array(features)

            # Reshape for single prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Make prediction
            prediction = model.predict(features)[0]

            result = {
                'model_used': model_name,
                'prediction': int(prediction),
                'prediction_label': 'Default' if prediction == 1 else 'No Default'
            }

            # Add probability if requested and available
            if return_probability and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                result['probability_no_default'] = float(probabilities[0])
                result['probability_default'] = float(probabilities[1])
                result['confidence'] = float(max(probabilities))

            elif return_probability and hasattr(model, 'decision_function'):
                # For SVM without probability
                decision_score = model.decision_function(features)[0]
                result['decision_score'] = float(decision_score)
                result['confidence'] = float(abs(decision_score))

            logger.info(f"Prediction made using {model_name}: {result['prediction_label']}")

            return result

        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}'}

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison of all trained models' performance.

        Returns:
            pd.DataFrame: Model comparison table
        """
        if not self.model_metrics:
            logger.warning("No trained models available for comparison")
            return pd.DataFrame()

        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.model_metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by ROC AUC (descending)
        if 'roc_auc' in df.columns:
            df = df.sort_values('roc_auc', ascending=False)

        return df

    def save_models(self, save_path: str) -> Dict[str, str]:
        """
        Save all trained models to disk.

        Args:
            save_path (str): Directory path to save models

        Returns:
            Dict[str, str]: Dictionary of model names and their saved file paths
        """
        import os

        if not self.trained_models:
            logger.warning("No trained models to save")
            return {}

        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        saved_models = {}

        for model_name, model in self.trained_models.items():
            try:
                file_path = os.path.join(save_path, f"{model_name.lower()}_model.joblib")
                joblib.dump(model, file_path)
                saved_models[model_name] = file_path
                logger.info(f"Saved {model_name} to {file_path}")

            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")

        # Save model metadata
        metadata = {
            'best_model': self.best_model_name,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'save_timestamp': datetime.now().isoformat()
        }

        metadata_path = os.path.join(save_path, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved {len(saved_models)} models and metadata to {save_path}")

        return saved_models

    def load_models(self, load_path: str) -> bool:
        """
        Load previously saved models from disk.

        Args:
            load_path (str): Directory path containing saved models

        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        import os
        import json

        if not os.path.exists(load_path):
            logger.error(f"Load path {load_path} does not exist")
            return False

        try:
            # Load metadata
            metadata_path = os.path.join(load_path, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.best_model_name = metadata.get('best_model')
                self.feature_names = metadata.get('feature_names')
                self.model_metrics = metadata.get('model_metrics', {})

            # Load models
            loaded_models = {}
            for file_name in os.listdir(load_path):
                if file_name.endswith('_model.joblib'):
                    model_name = file_name.replace('_model.joblib', '').title()
                    file_path = os.path.join(load_path, file_name)

                    try:
                        model = joblib.load(file_path)
                        loaded_models[model_name] = model
                        logger.info(f"Loaded {model_name} from {file_path}")

                    except Exception as e:
                        logger.error(f"Error loading {model_name}: {str(e)}")

            self.trained_models = loaded_models

            # Set best model
            if self.best_model_name and self.best_model_name in self.trained_models:
                self.best_model = self.trained_models[self.best_model_name]

            logger.info(f"Successfully loaded {len(loaded_models)} models from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading models from {load_path}: {str(e)}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the ModelManager state.

        Returns:
            Dict[str, Any]: Summary of models, metrics, and configuration
        """
        summary = {
            'total_models_configured': len(self.models),
            'total_models_trained': len(self.trained_models),
            'best_model': self.best_model_name,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'models_available': list(self.models.keys()),
            'models_trained': list(self.trained_models.keys()),
            'performance_summary': {}
        }

        # Add performance summary
        if self.model_metrics:
            for model_name, metrics in self.model_metrics.items():
                summary['performance_summary'][model_name] = {
                    'accuracy': round(metrics.get('accuracy', 0), 4),
                    'precision': round(metrics.get('precision', 0), 4),
                    'recall': round(metrics.get('recall', 0), 4),
                    'f1_score': round(metrics.get('f1_score', 0), 4),
                    'roc_auc': round(metrics.get('roc_auc', 0), 4)
                }

        return summary
