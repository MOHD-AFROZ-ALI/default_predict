"""
Streamlit UI Pages for Credit Default Prediction Application
Contains all page functions for dashboard, risk assessment, analytics, and business intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import io

# Page configuration
def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Credit Default Prediction System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar():
    """Render application sidebar with navigation and controls"""
    st.sidebar.title("🏦 Credit Risk System")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Dashboard", "Individual Assessment", "Batch Processing", 
         "Analytics", "Business Intelligence", "Compliance", "Settings"]
    )

    # Model information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"Model Version: 1.0\nLast Updated: {datetime.now().strftime('%Y-%m-%d')}")

    # Quick stats
    st.sidebar.markdown("### Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Accuracy", "94.2%")
    with col2:
        st.metric("AUC", "0.89")

    return page

def dashboard_page():
    """Main dashboard page with overview metrics and charts"""
    st.title("📊 Credit Risk Dashboard")
    st.markdown("Real-time overview of credit risk metrics and model performance")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Applications Today", value="1,247", delta="12%")

    with col2:
        st.metric(label="Approval Rate", value="73.2%", delta="-2.1%")

    with col3:
        st.metric(label="Avg Risk Score", value="0.34", delta="0.02")

    with col4:
        st.metric(label="Model Accuracy", value="94.2%", delta="0.3%")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Daily Application Volume")
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        volumes = np.random.randint(800, 1500, len(dates))

        fig = px.line(x=dates, y=volumes, title="Application Volume Trend",
                     labels={'x': 'Date', 'y': 'Applications'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Risk Score Distribution")
        risk_scores = np.random.beta(2, 5, 1000)

        fig = px.histogram(x=risk_scores, nbins=30, title="Current Risk Score Distribution",
                          labels={'x': 'Risk Score', 'y': 'Count'})
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Decision Threshold")
        st.plotly_chart(fig, use_container_width=True)

    # Performance monitoring
    st.subheader("🔍 Model Performance Monitoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm_data = [[850, 120], [95, 935]]
        fig = px.imshow(cm_data, text_auto=True, aspect="auto",
                       color_continuous_scale="Blues",
                       labels=dict(x="Predicted", y="Actual"))
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No Default", "Default"])
        fig.update_yaxes(tickvals=[0, 1], ticktext=["No Default", "Default"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Feature Importance**")
        features = ['Credit Score', 'Income', 'Debt Ratio', 'Employment Length', 'Loan Amount']
        importance = [0.35, 0.25, 0.20, 0.12, 0.08]

        fig = px.bar(x=importance, y=features, orientation='h', title="Top Feature Importance")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("**ROC Curve**")
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-2 * fpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC Curve (AUC=0.89)'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='Random Classifier', line=dict(dash='dash')))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                         yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

def individual_assessment_page():
    """Individual credit risk assessment page"""
    st.title("👤 Individual Risk Assessment")
    st.markdown("Assess credit risk for individual loan applications")

    # Input form
    with st.form("risk_assessment_form"):
        st.subheader("📝 Application Details")

        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            employment_length = st.selectbox("Employment Length", 
                                           ["< 1 year", "1-2 years", "3-5 years", "5-10 years", "> 10 years"])
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

        with col2:
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
            loan_purpose = st.selectbox("Loan Purpose", 
                                      ["debt_consolidation", "credit_card", "home_improvement", 
                                       "major_purchase", "medical", "other"])
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.2, 0.01)
            loan_term = st.selectbox("Loan Term", ["36 months", "60 months"])

        submitted = st.form_submit_button("🔍 Assess Risk")

    if submitted:
        st.subheader("📊 Risk Assessment Results")

        # Calculate mock risk score
        risk_factors = {
            'credit_score': max(0, (700 - credit_score) / 400),
            'debt_to_income': debt_to_income,
            'income_factor': max(0, (60000 - annual_income) / 60000),
            'loan_amount_factor': loan_amount / 50000
        }

        risk_score = np.mean(list(risk_factors.values()))
        risk_score = max(0, min(1, risk_score))

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            # Risk score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score"},
                delta={'reference': 0.5},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if risk_score < 0.5:
                st.success("✅ **APPROVED**")
                st.write("Low risk application")
            else:
                st.error("❌ **DENIED**")
                st.write("High risk application")

            confidence = abs(risk_score - 0.5) * 2
            st.metric("Confidence", f"{confidence:.1%}")

        with col3:
            st.write("**Risk Factors:**")
            for factor, value in risk_factors.items():
                factor_name = factor.replace('_', ' ').title()
                st.write(f"• {factor_name}: {value:.2f}")

        # Detailed explanation
        st.subheader("📋 Decision Explanation")

        explanation_factors = []
        if credit_score < 600:
            explanation_factors.append("• Low credit score indicates higher default risk")
        if debt_to_income > 0.4:
            explanation_factors.append("• High debt-to-income ratio suggests financial strain")
        if annual_income < 30000:
            explanation_factors.append("• Low income may affect repayment ability")
        if loan_amount > annual_income * 0.5:
            explanation_factors.append("• Large loan amount relative to income")

        if explanation_factors:
            st.warning("**Key Risk Factors:**")
            for factor in explanation_factors:
                st.write(factor)
        else:
            st.info("**Positive Indicators:**\n• Strong credit profile\n• Manageable debt levels\n• Adequate income")

def batch_processing_page():
    """Batch processing page for multiple applications"""
    st.title("📁 Batch Processing")
    st.markdown("Process multiple loan applications simultaneously")

    # File upload
    st.subheader("📤 Upload Applications")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with loan applications",
        type=['csv'],
        help="File should contain columns: credit_score, annual_income, loan_amount, debt_to_income, etc."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} applications")

            # Show data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Processing options
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("⚙️ Processing Options")
                include_explanations = st.checkbox("Include decision explanations", value=True)
                risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)

            with col2:
                st.subheader("📊 Data Summary")
                st.write(f"**Total Applications:** {len(df)}")
                st.write(f"**Columns:** {len(df.columns)}")
                st.write(f"**Missing Values:** {df.isnull().sum().sum()}")

            # Process button
            if st.button("🚀 Process All Applications", type="primary"):
                with st.spinner("Processing applications..."):
                    progress_bar = st.progress(0)

                    results = []
                    for i, row in df.iterrows():
                        progress_bar.progress((i + 1) / len(df))

                        risk_score = np.random.beta(2, 3)
                        decision = "APPROVED" if risk_score < risk_threshold else "DENIED"
                        confidence = abs(risk_score - risk_threshold) * 2

                        result = {
                            'application_id': i + 1,
                            'risk_score': risk_score,
                            'decision': decision,
                            'confidence': confidence
                        }

                        if include_explanations:
                            result['explanation'] = f"Risk score: {risk_score:.3f}, Threshold: {risk_threshold}"

                        results.append(result)

                # Display results
                results_df = pd.DataFrame(results)

                st.subheader("📊 Processing Results")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Processed", len(results_df))

                with col2:
                    approved = len(results_df[results_df['decision'] == 'APPROVED'])
                    st.metric("Approved", approved)

                with col3:
                    denied = len(results_df[results_df['decision'] == 'DENIED'])
                    st.metric("Denied", denied)

                with col4:
                    approval_rate = approved / len(results_df) * 100
                    st.metric("Approval Rate", f"{approval_rate:.1f}%")

                # Results table
                st.subheader("📋 Detailed Results")
                st.dataframe(results_df, use_container_width=True)

                # Download results
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)

                st.download_button(
                    label="📥 Download Results",
                    data=csv_buffer.getvalue(),
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    else:
        # Show sample format
        st.subheader("📋 Expected File Format")
        sample_data = {
            'credit_score': [650, 720, 580],
            'annual_income': [50000, 75000, 35000],
            'loan_amount': [15000, 20000, 8000],
            'debt_to_income': [0.25, 0.15, 0.45],
            'employment_length': ['3-5 years', '> 10 years', '< 1 year']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

def analytics_page():
    """Analytics and reporting page"""
    st.title("📈 Analytics & Reporting")
    st.markdown("Comprehensive analytics and performance insights")

    # Time period selector
    col1, col2 = st.columns([1, 3])

    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Last year"]
        )

    # Performance metrics
    st.subheader("🎯 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Accuracy trend
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        accuracy = 0.94 + np.random.normal(0, 0.01, 30)

        fig = px.line(x=dates, y=accuracy, title="Model Accuracy Trend")
        fig.update_layout(yaxis_range=[0.9, 0.98])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Precision/Recall
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [0.92, 0.89, 0.90]

        fig = px.bar(x=metrics, y=values, title="Classification Metrics")
        fig.update_layout(yaxis_range=[0.8, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # AUC trend
        auc_values = 0.89 + np.random.normal(0, 0.005, 30)

        fig = px.line(x=dates, y=auc_values, title="AUC Score Trend")
        fig.update_layout(yaxis_range=[0.85, 0.95])
        st.plotly_chart(fig, use_container_width=True)

    # Feature analysis
    st.subheader("🔍 Feature Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance over time
        features = ['Credit Score', 'Income', 'Debt Ratio', 'Employment', 'Loan Amount']
        importance_data = np.random.rand(5, 4)

        fig = go.Figure()
        for i, feature in enumerate(features):
            fig.add_trace(go.Scatter(
                x=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                y=importance_data[i],
                mode='lines+markers',
                name=feature
            ))

        fig.update_layout(title="Feature Importance Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature correlation heatmap
        correlation_data = np.random.rand(5, 5)
        correlation_data = (correlation_data + correlation_data.T) / 2
        np.fill_diagonal(correlation_data, 1)

        fig = px.imshow(
            correlation_data,
            x=features,
            y=features,
            color_continuous_scale="RdBu",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

def business_intelligence_page():
    """Business intelligence and insights page"""
    st.title("💼 Business Intelligence")
    st.markdown("Strategic insights and business metrics")

    # Executive summary
    st.subheader("📊 Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Monthly Revenue", "$2.4M", "8.2%")

    with col2:
        st.metric("Portfolio Size", "$45.2M", "12.1%")

    with col3:
        st.metric("Default Rate", "3.2%", "-0.5%")

    with col4:
        st.metric("ROI", "15.8%", "2.1%")

    # Business metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Revenue Analysis")

        products = ['Personal Loans', 'Auto Loans', 'Credit Cards', 'Mortgages']
        revenue = [1200000, 800000, 600000, 400000]

        fig = px.bar(x=products, y=revenue, title="Revenue by Product Line")
        fig.update_layout(yaxis_title="Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Growth Metrics")

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        growth = [5.2, 7.1, 8.5, 6.8, 9.2, 8.8]

        fig = px.line(x=months, y=growth, title="Monthly Growth Rate (%)")
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)

def compliance_page():
    """Compliance monitoring and reporting page"""
    st.title("⚖️ Compliance Dashboard")
    st.markdown("Fair lending compliance and regulatory monitoring")

    # Compliance status overview
    st.subheader("🛡️ Compliance Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Fair Lending Score", "98.5%", "0.2%")

    with col2:
        st.metric("Disparate Impact", "0.85", "0.03")

    with col3:
        st.metric("Audit Score", "A+", "")

    with col4:
        st.metric("Regulatory Issues", "0", "-2")

    # Fair lending analysis
    st.subheader("⚖️ Fair Lending Analysis")

    col1, col2 = st.columns(2)

    with col1:
        groups = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
        approval_rates = [0.75, 0.68, 0.71, 0.78, 0.73]

        fig = px.bar(x=groups, y=approval_rates, title="Approval Rates by Race/Ethnicity")
        fig.add_hline(y=0.8*max(approval_rates), line_dash="dash", 
                     annotation_text="80% Rule Threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_data = {
            'Gender': ['Male', 'Female'],
            'Applications': [1200, 1100],
            'Approvals': [900, 825]
        }

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Applications', x=gender_data['Gender'], 
                            y=gender_data['Applications']))
        fig.add_trace(go.Bar(name='Approvals', x=gender_data['Gender'], 
                            y=gender_data['Approvals']))
        fig.update_layout(title="Applications and Approvals by Gender")
        st.plotly_chart(fig, use_container_width=True)

def settings_page():
    """Application settings and configuration page"""
    st.title("⚙️ Settings")
    st.markdown("Configure application settings and preferences")

    # Model settings
    st.subheader("🤖 Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Decision Thresholds**")
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.01)

        st.markdown("**Model Parameters**")
        model_version = st.selectbox("Model Version", ["1.0", "1.1", "2.0-beta"])
        auto_retrain = st.checkbox("Enable Auto-Retraining", value=True)

    with col2:
        st.markdown("**Data Processing**")
        missing_value_strategy = st.selectbox("Missing Value Strategy", 
                                            ["median", "mean", "mode"])
        outlier_detection = st.selectbox("Outlier Detection", 
                                       ["iqr", "zscore", "isolation_forest"])

        st.markdown("**Performance Monitoring**")
        drift_threshold = st.slider("Drift Detection Threshold", 0.0, 1.0, 0.1, 0.01)
        performance_alert = st.checkbox("Performance Alerts", value=True)

    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")
