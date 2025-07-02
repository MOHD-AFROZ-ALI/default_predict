"""
Credit Default SHAP Explainability Engine

This module provides core SHAP explainability functions for credit default prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPEngine:
    """
    Provides SHAP-based model explainability for credit default prediction.

    Core functionality:
    - Initialize SHAP explainers for different model types
    - Compute SHAP values for predictions
    - Generate summary and force plots
    - Provide business-friendly explanations
    """

    def __init__(self, model=None, X_train=None, feature_names=None):
        """Initialize the SHAP explainability engine."""
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or []
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

        # Configure matplotlib
        plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['figure.dpi'] = 100

        logger.info("SHAPEngine initialized")

    def initialize_explainer(self, model, X_train: np.ndarray, explainer_type: str = 'auto') -> bool:
        """
        Initialize SHAP explainer based on model type.

        Args:
            model: Trained ML model
            X_train: Training data for background samples
            explainer_type: Type of explainer ('auto', 'tree', 'linear', 'kernel')

        Returns:
            bool: True if successful
        """
        logger.info(f"Initializing SHAP explainer: {explainer_type}")

        try:
            self.model = model
            self.X_train = X_train

            # Auto-determine explainer type
            if explainer_type == 'auto':
                explainer_type = self._determine_explainer_type(model)

            # Initialize appropriate explainer
            if explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(model)
            elif explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(model, X_train)
            elif explainer_type == 'kernel':
                background_size = min(100, X_train.shape[0])
                background_data = shap.sample(X_train, background_size)
                self.explainer = shap.KernelExplainer(model.predict_proba, background_data)
            else:
                logger.error(f"Unknown explainer type: {explainer_type}")
                return False

            # Store expected value
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value
                if isinstance(self.expected_value, np.ndarray):
                    self.expected_value = self.expected_value[1]  # Binary classification

            logger.info("SHAP explainer initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            return False

    def _determine_explainer_type(self, model) -> str:
        """Determine the best SHAP explainer type for the model."""
        model_name = type(model).__name__.lower()

        # Tree-based models
        tree_models = ['randomforest', 'gradientboosting', 'xgboost', 'decisiontree']
        if any(tree_model in model_name for tree_model in tree_models):
            return 'tree'

        # Linear models
        linear_models = ['logisticregression', 'linearregression', 'ridge', 'lasso']
        if any(linear_model in model_name for linear_model in linear_models):
            return 'linear'

        # Default to kernel explainer
        return 'kernel'

    def compute_shap_values(self, X_test: np.ndarray, max_samples: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values for test data.

        Args:
            X_test: Test data to explain
            max_samples: Maximum samples to compute (for efficiency)

        Returns:
            Tuple of SHAP values and expected value
        """
        if self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return None, None

        logger.info(f"Computing SHAP values for {X_test.shape[0]} samples")

        try:
            # Limit samples for efficiency
            if max_samples and X_test.shape[0] > max_samples:
                indices = np.random.choice(X_test.shape[0], max_samples, replace=False)
                X_explain = X_test[indices]
            else:
                X_explain = X_test

            # Compute SHAP values
            shap_values = self.explainer.shap_values(X_explain)

            # Handle binary classification
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Use positive class

            # Store computed values
            self.shap_values = shap_values
            self.X_explain = X_explain

            logger.info(f"SHAP values computed. Shape: {shap_values.shape}")
            return shap_values, self.expected_value

        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            return None, None

    def generate_summary_plot(self, shap_values=None, X_data=None, feature_names=None, 
                            plot_type='dot', max_display=20, save_path=None) -> bool:
        """
        Generate SHAP summary plot.

        Args:
            shap_values: SHAP values to plot
            X_data: Feature data
            feature_names: Feature names
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum features to display
            save_path: Path to save plot

        Returns:
            bool: True if successful
        """
        logger.info(f"Generating SHAP summary plot: {plot_type}")

        try:
            # Use stored values if not provided
            if shap_values is None:
                shap_values = self.shap_values
            if X_data is None:
                X_data = getattr(self, 'X_explain', None)
            if feature_names is None:
                feature_names = self.feature_names

            if shap_values is None:
                logger.error("No SHAP values available")
                return False

            # Create plot
            plt.figure(figsize=(12, 8))

            if plot_type == 'bar':
                shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                                plot_type='bar', max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                                max_display=max_display, show=False)

            plt.title('SHAP Summary - Credit Default Prediction', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")

            plt.show()
            return True

        except Exception as e:
            logger.error(f"Error generating summary plot: {str(e)}")
            return False

    def generate_force_plot(self, instance_index=0, shap_values=None, X_data=None, 
                          feature_names=None, save_path=None) -> bool:
        """
        Generate force plot for single prediction explanation.

        Args:
            instance_index: Index of instance to explain
            shap_values: SHAP values
            X_data: Feature data
            feature_names: Feature names
            save_path: Path to save plot

        Returns:
            bool: True if successful
        """
        logger.info(f"Generating force plot for instance {instance_index}")

        try:
            # Use stored values if not provided
            if shap_values is None:
                shap_values = self.shap_values
            if X_data is None:
                X_data = getattr(self, 'X_explain', None)
            if feature_names is None:
                feature_names = self.feature_names

            if shap_values is None or X_data is None:
                logger.error("SHAP values or data not available")
                return False

            # Create custom force plot
            instance_shap = shap_values[instance_index]
            instance_features = X_data[instance_index]

            # Prepare data for visualization
            feature_data = []
            for i, (shap_val, feature_val) in enumerate(zip(instance_shap, instance_features)):
                feature_name = feature_names[i] if i < len(feature_names) else f'Feature_{i}'
                feature_data.append({
                    'feature': feature_name,
                    'shap_value': shap_val,
                    'feature_value': feature_val,
                    'abs_shap': abs(shap_val)
                })

            # Sort and take top features
            feature_df = pd.DataFrame(feature_data)
            top_features = feature_df.nlargest(15, 'abs_shap')

            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in top_features['shap_value']]

            ax.barh(range(len(top_features)), top_features['shap_value'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([f"{row['feature']} = {row['feature_value']:.2f}" 
                               for _, row in top_features.iterrows()])
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title(f'Feature Impact - Credit Default Prediction (Instance {instance_index})', 
                        fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Add legend
            ax.text(0.02, 0.98, 'Blue: Increases risk\nRed: Decreases risk', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Force plot saved to {save_path}")

            plt.show()
            return True

        except Exception as e:
            logger.error(f"Error generating force plot: {str(e)}")
            return False

    def generate_business_explanation(self, instance_index=0, shap_values=None, X_data=None, 
                                    feature_names=None, prediction_proba=None) -> Dict[str, Any]:
        """
        Generate business-friendly explanation of model prediction.

        Args:
            instance_index: Index of instance to explain
            shap_values: SHAP values
            X_data: Feature data
            feature_names: Feature names
            prediction_proba: Prediction probability

        Returns:
            Dict with business explanation
        """
        logger.info(f"Generating business explanation for instance {instance_index}")

        try:
            # Use stored values if not provided
            if shap_values is None:
                shap_values = self.shap_values
            if X_data is None:
                X_data = getattr(self, 'X_explain', None)
            if feature_names is None:
                feature_names = self.feature_names

            if shap_values is None or X_data is None:
                return {}

            # Get instance data
            instance_shap = shap_values[instance_index]
            instance_features = X_data[instance_index]

            # Analyze features
            feature_analysis = []
            for i, (shap_val, feature_val) in enumerate(zip(instance_shap, instance_features)):
                feature_name = feature_names[i] if i < len(feature_names) else f'Feature_{i}'
                feature_analysis.append({
                    'feature': feature_name,
                    'value': float(feature_val),
                    'shap_value': float(shap_val),
                    'impact': 'increases' if shap_val > 0 else 'decreases',
                    'magnitude': abs(float(shap_val))
                })

            # Sort by impact magnitude
            feature_analysis.sort(key=lambda x: x['magnitude'], reverse=True)

            # Get top contributors
            positive_contributors = [f for f in feature_analysis if f['shap_value'] > 0][:3]
            negative_contributors = [f for f in feature_analysis if f['shap_value'] < 0][:3]

            # Calculate risk assessment
            net_impact = sum(f['shap_value'] for f in feature_analysis)

            if prediction_proba is not None:
                risk_probability = prediction_proba
            else:
                base_value = self.expected_value if self.expected_value is not None else 0.5
                risk_probability = max(0, min(1, base_value + net_impact))

            # Risk categorization
            if risk_probability >= 0.7:
                risk_level = "HIGH"
                risk_description = "Strong indicators suggest high default risk"
            elif risk_probability >= 0.4:
                risk_level = "MEDIUM"
                risk_description = "Mixed indicators suggest moderate default risk"
            else:
                risk_level = "LOW"
                risk_description = "Indicators suggest low default risk"

            # Create explanation
            explanation = {
                'instance_index': instance_index,
                'risk_assessment': {
                    'risk_level': risk_level,
                    'risk_probability': round(risk_probability, 4),
                    'risk_description': risk_description
                },
                'key_factors': {
                    'top_risk_increasing': [
                        {
                            'feature': f['feature'],
                            'value': f['value'],
                            'impact_score': round(f['shap_value'], 4)
                        }
                        for f in positive_contributors
                    ],
                    'top_risk_decreasing': [
                        {
                            'feature': f['feature'],
                            'value': f['value'],
                            'impact_score': round(f['shap_value'], 4)
                        }
                        for f in negative_contributors
                    ]
                },
                'summary': f"Risk Level: {risk_level} ({risk_probability:.1%} probability)"
            }

            return explanation

        except Exception as e:
            logger.error(f"Error generating business explanation: {str(e)}")
            return {}

    def get_feature_importance(self, model_name=None) -> Dict[str, float]:
        """
        Extract feature importance from SHAP values.

        Returns:
            Dict of feature importance scores
        """
        if self.shap_values is None:
            logger.warning("No SHAP values available")
            return {}

        try:
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(self.shap_values), axis=0)

            # Create feature importance dictionary
            if self.feature_names and len(self.feature_names) == len(mean_shap):
                feature_importance = dict(zip(self.feature_names, mean_shap))
            else:
                feature_importance = {f'feature_{i}': imp for i, imp in enumerate(mean_shap)}

            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return feature_importance

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return {}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of SHAP engine state."""
        return {
            'explainer_initialized': self.explainer is not None,
            'shap_values_computed': self.shap_values is not None,
            'feature_count': len(self.feature_names),
            'samples_explained': self.shap_values.shape[0] if self.shap_values is not None else 0,
            'expected_value': self.expected_value
        }
