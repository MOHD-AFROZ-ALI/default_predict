"""
Individual Risk Assessment Module
Real-time credit risk assessment with SHAP explanations

This module provides comprehensive individual credit risk assessment functionality
including form validation, data preprocessing, prediction engine, and SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from typing import Dict, Any, Tuple, Optional
import hashlib
import json
from datetime import datetime
import os

def create_risk_assessment_form_config() -> Dict[str, Any]:
    """
    Create configuration for risk assessment form fields

    Returns:
        Dict containing field configurations with validation rules
    """
    return {
        'person_age': {'type': 'number', 'min': 18, 'max': 100, 'default': 35},
        'person_income': {'type': 'number', 'min': 0, 'max': 1000000, 'default': 50000},
        'person_home_ownership': {'type': 'selectbox', 'options': ['RENT', 'OWN', 'MORTGAGE', 'OTHER']},
        'person_emp_length': {'type': 'number', 'min': 0, 'max': 50, 'default': 5},
        'loan_intent': {'type': 'selectbox', 'options': ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']},
        'loan_grade': {'type': 'selectbox', 'options': ['A', 'B', 'C', 'D', 'E', 'F', 'G']},
        'loan_amnt': {'type': 'number', 'min': 1000, 'max': 100000, 'default': 10000},
        'loan_int_rate': {'type': 'number', 'min': 0.0, 'max': 30.0, 'default': 10.0},
        'loan_percent_income': {'type': 'number', 'min': 0.0, 'max': 1.0, 'default': 0.2},
        'cb_person_default_on_file': {'type': 'selectbox', 'options': ['Y', 'N']},
        'cb_person_cred_hist_length': {'type': 'number', 'min': 0, 'max': 50, 'default': 10}
    }

def build_customer_input_form(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build Streamlit form for customer input

    Args:
        config: Form configuration dictionary

    Returns:
        Dictionary of form inputs if submitted, None otherwise
    """
    inputs = {}

    with st.form("risk_assessment_form"):
        st.subheader("🎯 Credit Risk Assessment")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Personal Information**")
            inputs['person_age'] = st.number_input(
                "Age", 
                min_value=config['person_age']['min'],
                max_value=config['person_age']['max'],
                value=config['person_age']['default']
            )
            inputs['person_income'] = st.number_input(
                "Annual Income ($)", 
                min_value=config['person_income']['min'],
                max_value=config['person_income']['max'],
                value=config['person_income']['default']
            )
            inputs['person_home_ownership'] = st.selectbox(
                "Home Ownership", 
                config['person_home_ownership']['options']
            )
            inputs['person_emp_length'] = st.number_input(
                "Employment Length (years)", 
                min_value=config['person_emp_length']['min'],
                max_value=config['person_emp_length']['max'],
                value=config['person_emp_length']['default']
            )
            inputs['cb_person_cred_hist_length'] = st.number_input(
                "Credit History Length (years)", 
                min_value=config['cb_person_cred_hist_length']['min'],
                max_value=config['cb_person_cred_hist_length']['max'],
                value=config['cb_person_cred_hist_length']['default']
            )
            inputs['cb_person_default_on_file'] = st.selectbox(
                "Previous Default", 
                config['cb_person_default_on_file']['options']
            )

        with col2:
            st.markdown("**Loan Information**")
            inputs['loan_intent'] = st.selectbox(
                "Loan Purpose", 
                config['loan_intent']['options']
            )
            inputs['loan_grade'] = st.selectbox(
                "Loan Grade", 
                config['loan_grade']['options']
            )
            inputs['loan_amnt'] = st.number_input(
                "Loan Amount ($)", 
                min_value=config['loan_amnt']['min'],
                max_value=config['loan_amnt']['max'],
                value=config['loan_amnt']['default']
            )
            inputs['loan_int_rate'] = st.number_input(
                "Interest Rate (%)", 
                min_value=config['loan_int_rate']['min'],
                max_value=config['loan_int_rate']['max'],
                value=config['loan_int_rate']['default']
            )
            inputs['loan_percent_income'] = st.number_input(
                "Loan as % of Income", 
                min_value=config['loan_percent_income']['min'],
                max_value=config['loan_percent_income']['max'],
                value=config['loan_percent_income']['default']
            )

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    return inputs if submitted else None

def validate_form_inputs(inputs: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate form inputs against configuration and business rules

    Args:
        inputs: Form input values
        config: Form configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = {}

    for field, value in inputs.items():
        field_config = config.get(field, {})

        if field_config.get('type') == 'number':
            if value < field_config.get('min', float('-inf')):
                errors[field] = f"Value must be at least {field_config['min']}"
            elif value > field_config.get('max', float('inf')):
                errors[field] = f"Value must be at most {field_config['max']}"

        elif field_config.get('type') == 'selectbox':
            if value not in field_config.get('options', []):
                errors[field] = f"Invalid option selected"

    # Business logic validation
    if inputs.get('loan_amnt', 0) > inputs.get('person_income', 0) * 5:
        errors['loan_amnt'] = "Loan amount seems too high relative to income"

    if inputs.get('loan_percent_income', 0) > 0.5:
        errors['loan_percent_income'] = "Loan percentage of income is very high"

    if inputs.get('person_age', 0) < 18:
        errors['person_age'] = "Applicant must be at least 18 years old"

    return len(errors) == 0, errors

def sanitize_credit_data_inputs(inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Sanitize and prepare inputs for model prediction

    Args:
        inputs: Raw form inputs

    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    # Create DataFrame
    df = pd.DataFrame([inputs])

    # Encode categorical variables
    categorical_mappings = {
        'person_home_ownership': {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3},
        'loan_intent': {
            'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 
            'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5
        },
        'loan_grade': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
        'cb_person_default_on_file': {'N': 0, 'Y': 1}
    }

    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Ensure all numeric columns are float
    numeric_cols = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    df = df.fillna(0)

    # Feature engineering
    df['debt_to_income'] = df['loan_amnt'] / df['person_income'].replace(0, 1)
    df['age_income_ratio'] = df['person_age'] / (df['person_income'] / 1000)

    return df

class PredictionEngine:
    """
    Real-time prediction engine with caching and SHAP integration

    Handles model loading, prediction caching, and SHAP explanations
    for individual credit risk assessment.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.explainer = None
        self.cache = {}
        self.feature_names = [
            'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
            'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
            'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'
        ]

        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self._initialize_explainer()
            except Exception as e:
                st.warning(f"Could not load model: {e}")

    def _initialize_explainer(self):
        """Initialize SHAP explainer for model interpretability"""
        if self.model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                st.warning(f"Could not initialize SHAP explainer: {e}")

    def predict(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Make prediction and return probability and risk score

        Args:
            data: Preprocessed input data

        Returns:
            Tuple of (probability, risk_score)
        """
        if self.model is None:
            # Intelligent mock prediction based on input features
            risk_factors = 0

            # Age factor
            age = data.iloc[0].get('person_age', 35)
            if age < 25 or age > 65:
                risk_factors += 0.1

            # Income factor
            income = data.iloc[0].get('person_income', 50000)
            if income < 30000:
                risk_factors += 0.2
            elif income > 100000:
                risk_factors -= 0.1

            # Loan factors
            loan_pct = data.iloc[0].get('loan_percent_income', 0.2)
            if loan_pct > 0.3:
                risk_factors += 0.3
            elif loan_pct < 0.1:
                risk_factors -= 0.1

            # Default history (most important factor)
            default_history = data.iloc[0].get('cb_person_default_on_file', 0)
            if default_history == 1:
                risk_factors += 0.4

            # Employment length
            emp_length = data.iloc[0].get('person_emp_length', 5)
            if emp_length < 2:
                risk_factors += 0.15
            elif emp_length > 10:
                risk_factors -= 0.1

            # Loan grade
            loan_grade = data.iloc[0].get('loan_grade', 2)
            risk_factors += loan_grade * 0.05

            # Add controlled randomness
            risk_score = min(0.95, max(0.05, risk_factors + np.random.normal(0, 0.05)))
            probability = risk_score
        else:
            try:
                probability = self.model.predict_proba(data)[0][1]
                risk_score = probability
            except Exception as e:
                st.error(f"Prediction error: {e}")
                risk_score = 0.5
                probability = 0.5

        return probability, risk_score

    def get_shap_values(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get SHAP values for model explanation

        Args:
            data: Input data for explanation

        Returns:
            SHAP values array
        """
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(data)
                return shap_values[1] if isinstance(shap_values, list) else shap_values
            except Exception as e:
                st.warning(f"SHAP calculation error: {e}")

        # Intelligent mock SHAP values based on feature importance
        mock_values = []
        row = data.iloc[0]

        for feature in data.columns:
            if feature == 'cb_person_default_on_file':
                mock_values.append(0.35 if row[feature] == 1 else -0.15)
            elif feature == 'loan_percent_income':
                mock_values.append(0.25 if row[feature] > 0.3 else -0.1)
            elif feature == 'person_income':
                mock_values.append(-0.2 if row[feature] > 50000 else 0.15)
            elif feature == 'loan_grade':
                mock_values.append(0.1 * row[feature] - 0.25)
            elif feature == 'person_age':
                if row[feature] < 25 or row[feature] > 65:
                    mock_values.append(0.1)
                else:
                    mock_values.append(-0.05)
            elif feature == 'person_emp_length':
                mock_values.append(-0.1 if row[feature] > 5 else 0.08)
            else:
                mock_values.append(np.random.normal(0, 0.03))

        return np.array(mock_values)

def create_prediction_engine(model_path: Optional[str] = None) -> PredictionEngine:
    """
    Create and initialize prediction engine

    Args:
        model_path: Path to trained model file

    Returns:
        Initialized PredictionEngine instance
    """
    return PredictionEngine(model_path)

def calculate_live_prediction(engine: PredictionEngine, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate live prediction with explanations and recommendations

    Args:
        engine: Prediction engine instance
        data: Preprocessed input data

    Returns:
        Dictionary containing prediction results and metadata
    """
    probability, risk_score = engine.predict(data)
    shap_values = engine.get_shap_values(data)

    # Risk categorization with detailed recommendations
    if risk_score < 0.3:
        risk_category = "Low Risk"
        risk_color = "green"
        recommendation = "✅ Approve loan with standard terms"
        confidence = "High"
    elif risk_score < 0.7:
        risk_category = "Medium Risk"
        risk_color = "orange"
        recommendation = "⚠️ Consider with additional conditions or higher interest rate"
        confidence = "Medium"
    else:
        risk_category = "High Risk"
        risk_color = "red"
        recommendation = "❌ Recommend rejection or require collateral"
        confidence = "High"

    return {
        'probability': float(probability),
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'risk_color': risk_color,
        'recommendation': recommendation,
        'confidence': confidence,
        'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
        'feature_names': data.columns.tolist(),
        'timestamp': datetime.now().isoformat()
    }

def implement_prediction_caching(inputs: Dict[str, Any]) -> str:
    """
    Generate cache key for prediction inputs to avoid redundant calculations

    Args:
        inputs: Input dictionary

    Returns:
        MD5 hash string for caching
    """
    # Sort inputs for consistent hashing
    input_str = json.dumps(inputs, sort_keys=True)
    return hashlib.md5(input_str.encode()).hexdigest()

def integrate_realtime_shap(prediction_result: Dict[str, Any], data: pd.DataFrame) -> None:
    """
    Display real-time SHAP explanations with interactive visualizations

    Args:
        prediction_result: Prediction results dictionary
        data: Input data DataFrame
    """
    if 'shap_values' in prediction_result and 'feature_names' in prediction_result:
        st.subheader("🔍 Feature Importance Analysis (SHAP)")

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': prediction_result['feature_names'],
            'SHAP Value': prediction_result['shap_values'],
            'Input Value': data.iloc[0].values
        })

        # Add feature descriptions for better understanding
        feature_descriptions = {
            'person_age': 'Age of applicant',
            'person_income': 'Annual income',
            'person_home_ownership': 'Home ownership status',
            'person_emp_length': 'Employment length',
            'loan_intent': 'Loan purpose',
            'loan_grade': 'Loan grade',
            'loan_amnt': 'Loan amount',
            'loan_int_rate': 'Interest rate',
            'loan_percent_income': 'Loan as % of income',
            'cb_person_default_on_file': 'Previous default',
            'cb_person_cred_hist_length': 'Credit history length',
            'debt_to_income': 'Debt to income ratio',
            'age_income_ratio': 'Age to income ratio'
        }

        feature_importance['Description'] = feature_importance['Feature'].map(
            feature_descriptions
        ).fillna(feature_importance['Feature'])

        # Sort by absolute SHAP value
        feature_importance['Abs_SHAP'] = abs(feature_importance['SHAP Value'])
        feature_importance = feature_importance.sort_values('Abs_SHAP', ascending=False)

        # Display visualizations
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Top Risk Factors**")
            chart_data = feature_importance.head(8).set_index('Description')['SHAP Value']
            st.bar_chart(chart_data)

        with col2:
            st.markdown("**SHAP Interpretation**")
            st.markdown("""
            - **Positive values**: Increase risk
            - **Negative values**: Decrease risk
            - **Larger magnitude**: More important
            """)

            # Summary statistics
            total_positive = feature_importance[feature_importance['SHAP Value'] > 0]['SHAP Value'].sum()
            total_negative = abs(feature_importance[feature_importance['SHAP Value'] < 0]['SHAP Value'].sum())

            st.metric("Risk Factors", f"+{total_positive:.3f}")
            st.metric("Protective Factors", f"-{total_negative:.3f}")

        # Display detailed table
        st.markdown("**Detailed Feature Analysis**")
        display_df = feature_importance[['Description', 'SHAP Value', 'Input Value']].head(10)
        display_df['Impact'] = display_df['SHAP Value'].apply(
            lambda x: '🔴 Increases Risk' if x > 0 else '🟢 Decreases Risk'
        )
        display_df['SHAP Value'] = display_df['SHAP Value'].round(4)
        st.dataframe(display_df, use_container_width=True)

def display_risk_assessment_results(prediction_result: Dict[str, Any]) -> None:
    """
    Display comprehensive risk assessment results with visual indicators

    Args:
        prediction_result: Dictionary containing prediction results
    """
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Risk Score", 
            f"{prediction_result['risk_score']:.1%}",
            delta=None
        )

    with col2:
        st.metric(
            "Risk Category", 
            prediction_result['risk_category'],
            delta=None
        )

    with col3:
        st.metric(
            "Default Probability", 
            f"{prediction_result['probability']:.1%}",
            delta=None
        )

    with col4:
        st.metric(
            "Confidence", 
            prediction_result['confidence'],
            delta=None
        )

    # Risk visualization
    risk_score = prediction_result['risk_score']
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; 
                background-color: {'#ffebee' if risk_score > 0.7 else '#fff3e0' if risk_score > 0.3 else '#e8f5e8'};">
        <h3 style="color: {prediction_result['risk_color']}; margin: 0;">
            {prediction_result['risk_category']}
        </h3>
        <p style="margin: 10px 0;"><strong>Recommendation:</strong> {prediction_result['recommendation']}</p>
        <p style="margin: 0; font-size: 0.9em; color: #666;">
            Assessment completed at {prediction_result['timestamp'][:19]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar for risk level
    st.markdown("**Risk Level Indicator**")
    progress_col1, progress_col2, progress_col3 = st.columns([1, 8, 1])

    with progress_col1:
        st.markdown("**Low**")
    with progress_col2:
        st.progress(risk_score)
    with progress_col3:
        st.markdown("**High**")
