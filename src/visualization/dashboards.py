"""
Credit Default Prediction - Streamlit Dashboard Module
This module contains the main dashboard interface for the credit default prediction app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import yaml
import os
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.model_trainer import ModelTrainer
from models.predictor import CreditPredictor
from visualization.charts import CreditVisualizationCharts

class CreditDefaultDashboard:
    """Main dashboard class for credit default prediction"""

    def __init__(self):
        self.config = self.load_config()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.predictor = CreditPredictor()
        self.viz = CreditVisualizationCharts(self.config)

        # Initialize session state
        self.init_session_state()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'app': {
                    'title': 'Credit Default Prediction Dashboard',
                    'description': 'ML-powered credit risk assessment tool'
                },
                'data': {
                    'features': ['age', 'income', 'debt_to_income_ratio', 'credit_score', 
                               'employment_length', 'loan_amount', 'loan_purpose', 'home_ownership'],
                    'target': 'default'
                },
                'models': {
                    'available_models': ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
                }
            }

    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'model_metrics' not in st.session_state:
            st.session_state.model_metrics = {}
        if 'preprocessor_fitted' not in st.session_state:
            st.session_state.preprocessor_fitted = None

    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title=self.config['app']['title'],
            page_icon="💳",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title(self.config['app']['title'])
        st.markdown(f"*{self.config['app']['description']}*")

        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Predictions", "📈 Model Comparison"]
        )

        if page == "🏠 Home":
            self.show_home_page()
        elif page == "📊 Data Analysis":
            self.show_data_analysis_page()
        elif page == "🤖 Model Training":
            self.show_model_training_page()
        elif page == "🔮 Predictions":
            self.show_prediction_page()
        elif page == "📈 Model Comparison":
            self.show_model_comparison_page()

    def show_home_page(self):
        """Display home page"""
        st.header("Welcome to Credit Default Prediction Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Purpose")
            st.write("""
            This dashboard helps financial institutions assess credit default risk using 
            machine learning models. It provides:

            - **Data Analysis**: Explore credit data patterns
            - **Model Training**: Train and compare ML models
            - **Predictions**: Make real-time credit decisions
            - **Insights**: Understand risk factors
            """)

            st.subheader("📋 Features")
            features = self.config['data']['features']
            for i, feature in enumerate(features, 1):
                st.write(f"{i}. {feature.replace('_', ' ').title()}")

        with col2:
            st.subheader("🚀 Quick Start")

            if st.button("Load Sample Data", type="primary"):
                with st.spinner("Loading data..."):
                    data = self.data_loader.load_data()
                    st.session_state.current_data = data
                    st.session_state.data_loaded = True
                st.success("✅ Sample data loaded successfully!")
                st.rerun()

            if st.session_state.data_loaded:
                st.info(f"📊 Data loaded: {st.session_state.current_data.shape[0]} records")

            st.subheader("📊 Data Overview")
            if st.session_state.data_loaded:
                data = st.session_state.current_data

                # Quick stats
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Records", len(data))
                with col_b:
                    default_rate = data['default'].mean() if 'default' in data.columns else 0
                    st.metric("Default Rate", f"{default_rate:.1%}")
                with col_c:
                    st.metric("Features", len(data.columns) - 1)

    def show_data_analysis_page(self):
        """Display data analysis page"""
        st.header("📊 Data Analysis")

        # Load data if not already loaded
        if not st.session_state.data_loaded:
            if st.button("Load Data"):
                with st.spinner("Loading data..."):
                    data = self.data_loader.load_data()
                    st.session_state.current_data = data
                    st.session_state.data_loaded = True
                st.rerun()
            return

        data = st.session_state.current_data

        # Data overview
        st.subheader("Data Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 1)
        with col3:
            default_rate = data['default'].mean() if 'default' in data.columns else 0
            st.metric("Default Rate", f"{default_rate:.1%}")
        with col4:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1%}")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10), use_container_width=True)

        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)

        # Visualizations
        st.subheader("Data Visualizations")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "🎯 Target Analysis", "📊 Feature Analysis"])

        with tab1:
            # Feature distributions
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'default' in numeric_cols:
                numeric_cols.remove('default')

            if numeric_cols:
                selected_feature = st.selectbox("Select feature for distribution:", numeric_cols)

                if selected_feature:
                    fig = self.viz.create_distribution_plot(
                        data[selected_feature], 
                        title=f"Distribution of {selected_feature.replace('_', ' ').title()}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show statistics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Mean", f"{data[selected_feature].mean():.2f}")
                    with col_b:
                        st.metric("Median", f"{data[selected_feature].median():.2f}")
                    with col_c:
                        st.metric("Std Dev", f"{data[selected_feature].std():.2f}")
                    with col_d:
                        st.metric("Missing", f"{data[selected_feature].isnull().sum()}")

        with tab2:
            # Correlation heatmap
            st.write("**Feature Correlation Matrix**")
            fig = self.viz.create_correlation_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)

            # Top correlations with target
            if 'default' in data.columns:
                numeric_data = data.select_dtypes(include=[np.number])
                correlations = numeric_data.corr()['default'].abs().sort_values(ascending=False)
                correlations = correlations.drop('default')  # Remove self-correlation

                st.write("**Top Features Correlated with Default:**")
                for i, (feature, corr) in enumerate(correlations.head(5).items(), 1):
                    st.write(f"{i}. {feature.replace('_', ' ').title()}: {corr:.3f}")

        with tab3:
            # Target variable analysis
            if 'default' in data.columns:
                # Default distribution
                default_counts = data['default'].value_counts()

                fig = go.Figure(data=[
                    go.Bar(
                        x=['No Default', 'Default'],
                        y=default_counts.values,
                        marker_color=['#2ca02c', '#d62728'],
                        text=default_counts.values,
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title='Default Distribution',
                    xaxis_title='Outcome',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Default rate by categorical features
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    selected_cat_feature = st.selectbox("Analyze default rate by:", categorical_cols)

                    if selected_cat_feature:
                        fig = self.viz.create_default_rate_by_feature(data, selected_cat_feature)
                        st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Feature analysis
            st.write("**Feature Information:**")

            feature_info = []
            for col in data.columns:
                if col != 'default':
                    info = {
                        'Feature': col.replace('_', ' ').title(),
                        'Type': str(data[col].dtype),
                        'Missing': data[col].isnull().sum(),
                        'Unique Values': data[col].nunique(),
                        'Sample Values': str(data[col].dropna().head(3).tolist())
                    }
                    feature_info.append(info)

            feature_df = pd.DataFrame(feature_info)
            st.dataframe(feature_df, use_container_width=True)

    def show_model_training_page(self):
        """Display model training page"""
        st.header("🤖 Model Training")

        # Check if data is loaded
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first from the Data Analysis page.")
            if st.button("Load Data Now"):
                with st.spinner("Loading data..."):
                    data = self.data_loader.load_data()
                    st.session_state.current_data = data
                    st.session_state.data_loaded = True
                st.rerun()
            return

        data = st.session_state.current_data

        # Training configuration
        st.subheader("Training Configuration")

        col1, col2 = st.columns(2)

        with col1:
            # Model selection
            available_models = self.config['models']['available_models']
            selected_models = st.multiselect(
                "Select models to train:",
                available_models,
                default=available_models[:2]
            )

            # Test size
            test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)

        with col2:
            # Cross-validation
            cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)

            # Random state
            random_state = st.number_input("Random state:", value=42, min_value=0)

        # Training button
        if st.button("🚀 Train Models", type="primary"):
            if not selected_models:
                st.error("Please select at least one model to train.")
                return

            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Prepare data
                    X, y = self.preprocessor.fit_transform(data)
                    st.session_state.preprocessor_fitted = self.preprocessor

                    # Split data
                    X_train, X_test, y_train, y_test = self.preprocessor.split_data(
                        X, y, test_size=test_size, random_state=random_state
                    )

                    # Train selected models
                    training_results = {}
                    progress_bar = st.progress(0)

                    for i, model_name in enumerate(selected_models):
                        st.write(f"Training {model_name}...")

                        result = self.model_trainer.train_model(
                            model_name, X_train, y_train, X_test, y_test
                        )
                        training_results[model_name] = result

                        progress_bar.progress((i + 1) / len(selected_models))

                    # Store results in session state
                    st.session_state.trained_models = self.model_trainer.trained_models
                    st.session_state.model_metrics = self.model_trainer.model_metrics
                    st.session_state.models_trained = True
                    st.session_state.training_results = training_results
                    st.session_state.test_data = (X_test, y_test)

                    st.success("✅ Models trained successfully!")

                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    return

        # Display training results
        if st.session_state.models_trained:
            st.subheader("Training Results")

            # Model comparison table
            comparison_df = self.model_trainer.get_model_comparison()
            st.dataframe(comparison_df, use_container_width=True)

            # Best model
            best_model_name, best_model = self.model_trainer.get_best_model()
            st.success(f"🏆 Best performing model: **{best_model_name}** (AUC: {st.session_state.model_metrics[best_model_name]['auc']:.4f})")

            # Model comparison chart
            fig = self.viz.create_model_comparison_chart(st.session_state.model_metrics)
            st.plotly_chart(fig, use_container_width=True)

            # Individual model details
            st.subheader("Model Details")

            selected_model_for_details = st.selectbox(
                "Select model for detailed analysis:",
                list(st.session_state.trained_models.keys())
            )

            if selected_model_for_details:
                model_metrics = st.session_state.model_metrics[selected_model_for_details]

                # Metrics display
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{model_metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{model_metrics['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{model_metrics['f1']:.4f}")
                with col5:
                    st.metric("AUC", f"{model_metrics['auc']:.4f}")

                # Feature importance
                if hasattr(st.session_state.trained_models[selected_model_for_details], 'feature_importances_') or                    hasattr(st.session_state.trained_models[selected_model_for_details], 'coef_'):

                    feature_names = self.preprocessor.get_feature_names()
                    if feature_names:
                        feature_importance = self.model_trainer.get_feature_importance(
                            selected_model_for_details, feature_names
                        )

                        if feature_importance:
                            st.subheader(f"Feature Importance - {selected_model_for_details}")
                            fig = self.viz.create_feature_importance_chart(feature_importance)
                            st.plotly_chart(fig, use_container_width=True)

                # ROC Curve and Confusion Matrix
                if 'training_results' in st.session_state and selected_model_for_details in st.session_state.training_results:
                    training_result = st.session_state.training_results[selected_model_for_details]
                    X_test, y_test = st.session_state.test_data

                    col_a, col_b = st.columns(2)

                    with col_a:
                        # ROC Curve
                        fig_roc = self.viz.create_roc_curve(
                            y_test, 
                            training_result['probabilities'], 
                            selected_model_for_details
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)

                    with col_b:
                        # Confusion Matrix
                        fig_cm = self.viz.create_confusion_matrix(
                            y_test, 
                            training_result['predictions']
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

        # Model export
        if st.session_state.models_trained:
            st.subheader("Export Models")

            export_model = st.selectbox(
                "Select model to export:",
                list(st.session_state.trained_models.keys())
            )

            if st.button("💾 Export Model"):
                try:
                    # Create models directory
                    models_dir = Path("models")
                    models_dir.mkdir(exist_ok=True)

                    # Save model
                    model_path = models_dir / f"{export_model}_model.joblib"
                    self.model_trainer.save_model(export_model, str(model_path))

                    # Save preprocessor
                    preprocessor_path = models_dir / f"{export_model}_preprocessor.joblib"
                    import joblib
                    joblib.dump(st.session_state.preprocessor_fitted, preprocessor_path)

                    st.success(f"✅ Model and preprocessor exported successfully!")
                    st.info(f"📁 Files saved to: {model_path} and {preprocessor_path}")

                except Exception as e:
                    st.error(f"❌ Error exporting model: {str(e)}")
