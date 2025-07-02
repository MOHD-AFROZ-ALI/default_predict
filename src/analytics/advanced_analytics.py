
"""
Advanced Analytics Module for Credit Default Prediction
Provides statistical analysis, model comparison, feature analysis, and performance evaluation
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Statistical analysis and hypothesis testing"""

    @staticmethod
    def perform_ks_test(data1, data2, feature_name=""):
        """Kolmogorov-Smirnov test for distribution comparison"""
        statistic, p_value = stats.ks_2samp(data1, data2)
        return {
            'feature': feature_name,
            'ks_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    @staticmethod
    def chi_square_test(observed, expected=None):
        """Chi-square test for independence"""
        if expected is None:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        else:
            chi2, p_value = stats.chisquare(observed, expected)
            dof = len(observed) - 1

        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05
        }

    @staticmethod
    def correlation_analysis(df, target_col, method='pearson'):
        """Comprehensive correlation analysis"""
        correlations = df.corr(method=method)[target_col].drop(target_col)

        return {
            'correlations': correlations.to_dict(),
            'strong_positive': correlations[correlations > 0.7].to_dict(),
            'strong_negative': correlations[correlations < -0.7].to_dict(),
            'weak_correlations': correlations[abs(correlations) < 0.3].to_dict()
        }

class ModelComparator:
    """Model comparison and benchmarking utilities"""

    def __init__(self):
        self.results = {}

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        self.results[model_name] = metrics
        return metrics

    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """Cross-validation evaluation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }

    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            return "No models evaluated yet"

        comparison_df = pd.DataFrame(self.results).T
        comparison_df['rank'] = comparison_df['f1_score'].rank(ascending=False)

        return comparison_df.sort_values('rank')

    def get_best_model(self, metric='f1_score'):
        """Get best performing model based on specified metric"""
        if not self.results:
            return None

        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1]

class FeatureAnalyzer:
    """Feature importance and analysis utilities"""

    @staticmethod
    def calculate_feature_importance(model, feature_names):
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = abs(model.coef_[0])
        else:
            return None

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return feature_importance

    @staticmethod
    def analyze_feature_distribution(df, feature_col, target_col):
        """Analyze feature distribution by target classes"""
        analysis = {}

        for target_value in df[target_col].unique():
            subset = df[df[target_col] == target_value][feature_col]
            analysis[f'target_{target_value}'] = {
                'mean': subset.mean(),
                'median': subset.median(),
                'std': subset.std(),
                'min': subset.min(),
                'max': subset.max(),
                'count': len(subset)
            }

        return analysis

    @staticmethod
    def feature_stability_index(expected, actual, bins=10):
        """Calculate Population Stability Index (PSI)"""
        def psi_calculation(expected_array, actual_array, buckets):
            expected_percents = np.histogram(expected_array, buckets)[0] / len(expected_array)
            actual_percents = np.histogram(actual_array, buckets)[0] / len(actual_array)

            # Avoid division by zero
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
            return psi

        breakpoints = np.linspace(min(expected.min(), actual.min()), 
                                max(expected.max(), actual.max()), bins + 1)

        psi_value = psi_calculation(expected, actual, breakpoints)

        stability_level = "Stable" if psi_value < 0.1 else "Moderate" if psi_value < 0.25 else "Unstable"

        return {
            'psi_value': psi_value,
            'stability_level': stability_level,
            'interpretation': f"PSI: {psi_value:.4f} - {stability_level}"
        }

class PerformanceEvaluator:
    """Advanced performance evaluation metrics"""

    @staticmethod
    def detailed_classification_report(y_true, y_pred, y_pred_proba=None):
        """Generate detailed classification performance report"""
        report = {
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        if y_pred_proba is not None:
            report['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            report['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            report['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}

        return report

    @staticmethod
    def calculate_lift_curve(y_true, y_pred_proba, bins=10):
        """Calculate lift curve for model performance"""
        df = pd.DataFrame({'actual': y_true, 'predicted_proba': y_pred_proba})
        df = df.sort_values('predicted_proba', ascending=False)

        df['decile'] = pd.qcut(df['predicted_proba'], bins, labels=False, duplicates='drop')

        lift_data = []
        for decile in range(bins):
            subset = df[df['decile'] == decile]
            if len(subset) > 0:
                actual_rate = subset['actual'].mean()
                baseline_rate = df['actual'].mean()
                lift = actual_rate / baseline_rate if baseline_rate > 0 else 0

                lift_data.append({
                    'decile': decile + 1,
                    'actual_rate': actual_rate,
                    'baseline_rate': baseline_rate,
                    'lift': lift,
                    'cumulative_lift': lift
                })

        return lift_data

    @staticmethod
    def model_stability_metrics(train_scores, test_scores):
        """Calculate model stability metrics"""
        train_mean, train_std = np.mean(train_scores), np.std(train_scores)
        test_mean, test_std = np.mean(test_scores), np.std(test_scores)

        stability_ratio = test_std / train_std if train_std > 0 else float('inf')
        performance_gap = abs(train_mean - test_mean)

        return {
            'train_performance': {'mean': train_mean, 'std': train_std},
            'test_performance': {'mean': test_mean, 'std': test_std},
            'stability_ratio': stability_ratio,
            'performance_gap': performance_gap,
            'is_stable': stability_ratio < 1.5 and performance_gap < 0.05
        }

# Utility functions for quick analysis
def quick_model_comparison(models_dict, X_test, y_test):
    """Quick comparison of multiple models"""
    comparator = ModelComparator()

    for name, model in models_dict.items():
        comparator.evaluate_model(model, X_test, y_test, name)

    return comparator.compare_models()

def analyze_dataset_drift(reference_data, current_data, features):
    """Analyze dataset drift using statistical tests"""
    analyzer = StatisticalAnalyzer()
    drift_results = {}

    for feature in features:
        if feature in reference_data.columns and feature in current_data.columns:
            result = analyzer.perform_ks_test(
                reference_data[feature].dropna(),
                current_data[feature].dropna(),
                feature
            )
            drift_results[feature] = result

    return drift_results

def comprehensive_model_evaluation(model, X_train, X_test, y_train, y_test, feature_names):
    """Comprehensive evaluation combining all analysis components"""
    evaluator = PerformanceEvaluator()
    analyzer = FeatureAnalyzer()

    # Model predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Performance evaluation
    performance = evaluator.detailed_classification_report(y_test, y_pred, y_pred_proba)

    # Feature importance
    feature_importance = analyzer.calculate_feature_importance(model, feature_names)

    # Model stability
    train_pred = model.predict(X_train)
    train_scores = [accuracy_score(y_train, train_pred)]
    test_scores = [accuracy_score(y_test, y_pred)]
    stability = evaluator.model_stability_metrics(train_scores, test_scores)

    return {
        'performance_metrics': performance,
        'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else None,
        'model_stability': stability
    }
