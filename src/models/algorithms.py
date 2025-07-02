"""
Machine Learning Algorithms for Credit Default Prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class CreditDefaultAlgorithms:
    """ML algorithms for credit default prediction"""

    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}

    def initialize_models(self):
        """Initialize all ML models with default parameters"""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='minkowski'
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'NaiveBayes': GaussianNB()
        }

    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model

    def train_all_models(self, X_train, y_train):
        """Train all models"""
        self.initialize_models()
        for model_name in self.models:
            self.train_model(model_name, X_train, y_train)

    def predict(self, model_name, X_test):
        """Make predictions with a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        return predictions, probabilities

    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a trained model"""
        predictions, probabilities = self.predict(model_name, X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions)
        }

        if probabilities is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, probabilities)

        self.results[model_name] = metrics
        return metrics

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        for model_name in self.trained_models:
            results[model_name] = self.evaluate_model(model_name, X_test, y_test)
        return results

    def cross_validate_model(self, model_name, X, y, cv=5):
        """Perform cross-validation on a model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }

    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, cv=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }

    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]

        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None

    def get_model_comparison(self):
        """Get comparison of all model results"""
        if not self.results:
            return None

        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)
        return comparison_df.sort_values('accuracy', ascending=False)

    def save_model(self, model_name, filepath):
        """Save a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        import joblib
        joblib.dump(self.trained_models[model_name], filepath)

    def load_model(self, model_name, filepath):
        """Load a saved model"""
        import joblib
        self.trained_models[model_name] = joblib.load(filepath)


def get_default_hyperparameters():
    """Get default hyperparameter grids for tuning"""
    return {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }


def ensemble_predictions(predictions_dict, method='voting'):
    """Combine predictions from multiple models"""
    if method == 'voting':
        # Simple majority voting
        predictions_array = np.array(list(predictions_dict.values()))
        ensemble_pred = np.round(np.mean(predictions_array, axis=0))
    elif method == 'weighted':
        # Weighted average (weights can be based on model performance)
        weights = np.ones(len(predictions_dict)) / len(predictions_dict)
        predictions_array = np.array(list(predictions_dict.values()))
        ensemble_pred = np.round(np.average(predictions_array, axis=0, weights=weights))

    return ensemble_pred.astype(int)
