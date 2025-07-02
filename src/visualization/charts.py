"""
Credit Default Prediction - Chart Generation Module
Plotly-based visualization functions for model performance and risk analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def create_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Create interactive confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Default', 'Predicted Default'],
        y=['Actual No Default', 'Actual Default'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=400
    )

    return fig

def create_roc_curve(y_true, y_prob, title="ROC Curve"):
    """Create ROC curve with AUC score"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        showlegend=True
    )

    return fig

def create_risk_gauge(risk_score, title="Default Risk Score"):
    """Create risk gauge meter"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))

    fig.update_layout(width=400, height=400)
    return fig

def create_feature_importance(feature_names, importance_scores, title="Feature Importance"):
    """Create horizontal bar chart for feature importance"""
    df = pd.DataFrame({
        'features': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['features'],
        orientation='h',
        marker_color='steelblue'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=max(400, len(feature_names) * 25),
        width=700
    )

    return fig

def create_performance_metrics(metrics_dict, title="Model Performance Metrics"):
    """Create bar chart for performance metrics"""
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig = go.Figure(go.Bar(
        x=metrics,
        y=values,
        marker_color=['green', 'blue', 'orange', 'red', 'purple'][:len(metrics)],
        text=[f'{v:.3f}' for v in values],
        textposition='auto'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Metrics',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        width=600,
        height=400
    )

    return fig

def create_prediction_distribution(predictions, title="Prediction Score Distribution"):
    """Create histogram of prediction scores"""
    fig = go.Figure(go.Histogram(
        x=predictions,
        nbinsx=30,
        marker_color='lightblue',
        opacity=0.7
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Prediction Score',
        yaxis_title='Frequency',
        width=600,
        height=400
    )

    return fig

def create_risk_distribution_by_segment(data, segment_col, risk_col, title="Risk Distribution by Segment"):
    """Create box plot showing risk distribution across segments"""
    fig = go.Figure()

    segments = data[segment_col].unique()
    for segment in segments:
        segment_data = data[data[segment_col] == segment][risk_col]
        fig.add_trace(go.Box(
            y=segment_data,
            name=str(segment),
            boxpoints='outliers'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=segment_col,
        yaxis_title=risk_col,
        width=700,
        height=500
    )

    return fig

def create_correlation_heatmap(correlation_matrix, title="Feature Correlation Matrix"):
    """Create correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))

    fig.update_layout(
        title=title,
        width=800,
        height=600
    )

    return fig

def create_threshold_analysis(y_true, y_prob, title="Threshold Analysis"):
    """Create precision-recall curve with threshold analysis"""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Precision-Recall Curve', 'Precision & Recall vs Threshold'),
        vertical_spacing=0.1
    )

    # Precision-Recall curve
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'),
        row=1, col=1
    )

    # Precision and Recall vs Threshold
    fig.add_trace(
        go.Scatter(x=thresholds, y=precision[:-1], mode='lines', name='Precision'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=thresholds, y=recall[:-1], mode='lines', name='Recall'),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        height=700,
        width=700
    )

    return fig

def create_model_comparison(model_results, title="Model Performance Comparison"):
    """Create grouped bar chart comparing multiple models"""
    models = list(model_results.keys())
    metrics = list(model_results[models[0]].keys())

    fig = go.Figure()

    for metric in metrics:
        values = [model_results[model][metric] for model in models]
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        width=800,
        height=500
    )

    return fig
