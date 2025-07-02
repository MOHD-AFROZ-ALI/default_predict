"""
Regulatory Compliance and Audit Functions
Handles fair lending analysis, model governance, and regulatory reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

class FairLendingAnalyzer:
    """Analyzes lending decisions for fair lending compliance"""

    def __init__(self):
        self.protected_classes = ['race', 'gender', 'age_group', 'ethnicity']
        self.thresholds = {
            'disparate_impact': 0.8,  # 80% rule
            'statistical_significance': 0.05
        }

    def calculate_disparate_impact(self, data: pd.DataFrame, 
                                 protected_class: str, 
                                 outcome_col: str = 'approved') -> Dict[str, float]:
        """Calculate disparate impact ratios for protected classes"""
        try:
            results = {}

            # Get approval rates by group
            approval_rates = data.groupby(protected_class)[outcome_col].mean()

            # Calculate disparate impact ratios
            reference_group = approval_rates.max()

            for group, rate in approval_rates.items():
                ratio = rate / reference_group if reference_group > 0 else 0
                results[f'{group}_ratio'] = ratio
                results[f'{group}_rate'] = rate
                results[f'{group}_compliant'] = ratio >= self.thresholds['disparate_impact']

            return results

        except Exception as e:
            logger.error(f"Error calculating disparate impact: {e}")
            return {}

    def statistical_parity_test(self, data: pd.DataFrame, 
                              protected_class: str,
                              outcome_col: str = 'approved') -> Dict[str, Any]:
        """Perform statistical tests for fair lending compliance"""
        try:
            groups = data[protected_class].unique()
            results = {}

            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    # Get outcomes for each group
                    group1_outcomes = data[data[protected_class] == group1][outcome_col]
                    group2_outcomes = data[data[protected_class] == group2][outcome_col]

                    # Perform chi-square test
                    contingency_table = pd.crosstab(
                        data[protected_class], 
                        data[outcome_col]
                    )

                    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

                    results[f'{group1}_vs_{group2}'] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'significant_difference': p_value < self.thresholds['statistical_significance']
                    }

            return results

        except Exception as e:
            logger.error(f"Error in statistical parity test: {e}")
            return {}

class ModelGovernance:
    """Handles model governance, validation, and monitoring"""

    def __init__(self):
        self.validation_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        self.drift_threshold = 0.1
        self.performance_threshold = 0.05

    def validate_model_performance(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Validate model performance metrics"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }

            if y_prob is not None:
                metrics['auc'] = roc_auc_score(y_true, y_prob)

            # Check if performance meets thresholds
            metrics['performance_acceptable'] = all(
                metrics[metric] >= 0.7 for metric in ['accuracy', 'precision', 'recall']
            )

            return metrics

        except Exception as e:
            logger.error(f"Error validating model performance: {e}")
            return {}

    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame,
                         features: List[str]) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        try:
            drift_results = {}

            for feature in features:
                if feature in reference_data.columns and feature in current_data.columns:
                    # Kolmogorov-Smirnov test for continuous variables
                    if pd.api.types.is_numeric_dtype(reference_data[feature]):
                        ks_stat, p_value = stats.ks_2samp(
                            reference_data[feature].dropna(),
                            current_data[feature].dropna()
                        )

                        drift_results[feature] = {
                            'test': 'ks_test',
                            'statistic': ks_stat,
                            'p_value': p_value,
                            'drift_detected': ks_stat > self.drift_threshold
                        }
                    else:
                        # Chi-square test for categorical variables
                        try:
                            ref_counts = reference_data[feature].value_counts()
                            curr_counts = current_data[feature].value_counts()

                            # Align categories
                            all_categories = set(ref_counts.index) | set(curr_counts.index)
                            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]

                            chi2, p_value, _, _ = stats.chi2_contingency([ref_aligned, curr_aligned])

                            drift_results[feature] = {
                                'test': 'chi2_test',
                                'statistic': chi2,
                                'p_value': p_value,
                                'drift_detected': p_value < 0.05
                            }
                        except:
                            drift_results[feature] = {'test': 'failed', 'drift_detected': False}

            return drift_results

        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            return {}

class RegulatoryReporter:
    """Generates regulatory compliance reports"""

    def __init__(self):
        self.report_templates = {
            'fair_lending': ['disparate_impact', 'statistical_tests', 'remediation_actions'],
            'model_validation': ['performance_metrics', 'drift_analysis', 'governance_status'],
            'audit_trail': ['model_changes', 'data_lineage', 'approval_workflow']
        }

    def generate_fair_lending_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fair lending compliance report"""
        try:
            report = {
                'report_date': datetime.now().isoformat(),
                'report_type': 'fair_lending_compliance',
                'executive_summary': {},
                'detailed_findings': analysis_results,
                'recommendations': [],
                'compliance_status': 'COMPLIANT'
            }

            # Analyze results for executive summary
            non_compliant_groups = []
            for key, value in analysis_results.items():
                if isinstance(value, dict) and 'compliant' in str(key).lower():
                    if not value:
                        non_compliant_groups.append(key)

            if non_compliant_groups:
                report['compliance_status'] = 'NON_COMPLIANT'
                report['recommendations'].append(
                    f"Address disparate impact in groups: {', '.join(non_compliant_groups)}"
                )

            report['executive_summary'] = {
                'total_groups_analyzed': len([k for k in analysis_results.keys() if 'ratio' in k]),
                'compliant_groups': len([k for k in analysis_results.keys() if 'compliant' in k and analysis_results[k]]),
                'non_compliant_groups': len(non_compliant_groups),
                'overall_status': report['compliance_status']
            }

            return report

        except Exception as e:
            logger.error(f"Error generating fair lending report: {e}")
            return {}

    def generate_model_governance_report(self, validation_results: Dict[str, Any],
                                       drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model governance and validation report"""
        try:
            report = {
                'report_date': datetime.now().isoformat(),
                'report_type': 'model_governance',
                'model_performance': validation_results,
                'data_drift_analysis': drift_results,
                'governance_status': 'APPROVED',
                'action_items': []
            }

            # Check for performance issues
            if not validation_results.get('performance_acceptable', True):
                report['governance_status'] = 'REQUIRES_ATTENTION'
                report['action_items'].append('Model performance below acceptable thresholds')

            # Check for data drift
            drift_detected = any(
                result.get('drift_detected', False) 
                for result in drift_results.values()
            )

            if drift_detected:
                report['governance_status'] = 'REQUIRES_ATTENTION'
                report['action_items'].append('Data drift detected in input features')

            return report

        except Exception as e:
            logger.error(f"Error generating model governance report: {e}")
            return {}

class AuditTrail:
    """Maintains audit trail for model decisions and changes"""

    def __init__(self):
        self.audit_log = []

    def log_model_decision(self, customer_id: str, features: Dict[str, Any],
                          prediction: int, probability: float,
                          model_version: str = "1.0") -> None:
        """Log individual model decisions for audit purposes"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'model_decision',
                'customer_id': customer_id,
                'model_version': model_version,
                'input_features': features,
                'prediction': prediction,
                'probability': probability,
                'decision_rationale': self._generate_decision_rationale(features, prediction)
            }

            self.audit_log.append(audit_entry)

        except Exception as e:
            logger.error(f"Error logging model decision: {e}")

    def log_model_change(self, change_type: str, description: str,
                        approver: str, model_version: str) -> None:
        """Log model changes and updates"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'model_change',
                'change_type': change_type,
                'description': description,
                'approver': approver,
                'model_version': model_version
            }

            self.audit_log.append(audit_entry)

        except Exception as e:
            logger.error(f"Error logging model change: {e}")

    def _generate_decision_rationale(self, features: Dict[str, Any], 
                                   prediction: int) -> str:
        """Generate human-readable decision rationale"""
        try:
            key_factors = []

            # Identify key risk factors
            if features.get('credit_score', 0) < 600:
                key_factors.append('Low credit score')
            if features.get('debt_to_income', 0) > 0.4:
                key_factors.append('High debt-to-income ratio')
            if features.get('employment_length', 0) < 2:
                key_factors.append('Short employment history')

            decision = 'APPROVED' if prediction == 0 else 'DENIED'

            if key_factors:
                rationale = f"Decision: {decision}. Key factors: {', '.join(key_factors)}"
            else:
                rationale = f"Decision: {decision}. Standard risk assessment applied."

            return rationale

        except Exception as e:
            logger.error(f"Error generating decision rationale: {e}")
            return f"Decision: {'APPROVED' if prediction == 0 else 'DENIED'}"

    def export_audit_log(self, filepath: str) -> bool:
        """Export audit log to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            return True

        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")
            return False
