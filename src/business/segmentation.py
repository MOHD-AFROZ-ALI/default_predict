import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentationEngine:
    """Advanced customer segmentation using multiple clustering techniques"""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.segment_profiles = {}
        self.feature_importance = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for segmentation"""
        features = df.copy()

        # Financial ratios
        if 'AMT_INCOME_TOTAL' in features.columns and 'AMT_CREDIT' in features.columns:
            features['income_credit_ratio'] = features['AMT_INCOME_TOTAL'] / (features['AMT_CREDIT'] + 1)
            features['credit_income_ratio'] = features['AMT_CREDIT'] / (features['AMT_INCOME_TOTAL'] + 1)

        # Age groups
        if 'DAYS_BIRTH' in features.columns:
            features['age'] = -features['DAYS_BIRTH'] / 365.25
            features['age_group'] = pd.cut(features['age'], bins=[0, 30, 45, 60, 100], labels=[1, 2, 3, 4])

        # Employment stability
        if 'DAYS_EMPLOYED' in features.columns:
            features['employment_years'] = -features['DAYS_EMPLOYED'] / 365.25
            features['employment_stability'] = features['employment_years'].apply(
                lambda x: 1 if x > 5 else (2 if x > 2 else 3)
            )

        # Select numeric features for clustering
        numeric_features = features.select_dtypes(include=[np.number]).columns
        return features[numeric_features].fillna(0)

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Fit segmentation model and predict segments"""
        features = self.prepare_features(df)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit clustering
        segments = self.kmeans.fit_predict(features_scaled)

        # Calculate segment profiles
        self._calculate_segment_profiles(df, segments)

        return segments

    def _calculate_segment_profiles(self, df: pd.DataFrame, segments: np.ndarray):
        """Calculate detailed profiles for each segment"""
        df_with_segments = df.copy()
        df_with_segments['segment'] = segments

        for segment_id in range(self.n_clusters):
            segment_data = df_with_segments[df_with_segments['segment'] == segment_id]

            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100,
                'avg_income': segment_data.get('AMT_INCOME_TOTAL', pd.Series([0])).mean(),
                'avg_credit': segment_data.get('AMT_CREDIT', pd.Series([0])).mean(),
                'avg_age': -segment_data.get('DAYS_BIRTH', pd.Series([0])).mean() / 365.25,
                'default_rate': segment_data.get('TARGET', pd.Series([0])).mean() if 'TARGET' in segment_data.columns else 0
            }

            self.segment_profiles[segment_id] = profile

class RiskSegmentAnalyzer:
    """Analyze risk patterns across customer segments"""

    def __init__(self):
        self.risk_metrics = {}
        self.segment_risk_profiles = {}

    def analyze_segment_risk(self, df: pd.DataFrame, segments: np.ndarray) -> Dict:
        """Comprehensive risk analysis by segment"""
        df_analysis = df.copy()
        df_analysis['segment'] = segments

        risk_analysis = {}

        for segment_id in np.unique(segments):
            segment_data = df_analysis[df_analysis['segment'] == segment_id]

            # Basic risk metrics
            default_rate = segment_data.get('TARGET', pd.Series([0])).mean()

            # Financial stress indicators
            income_credit_ratio = (segment_data.get('AMT_INCOME_TOTAL', pd.Series([1])) / 
                                 segment_data.get('AMT_CREDIT', pd.Series([1]))).mean()

            # Demographic risk factors
            avg_age = -segment_data.get('DAYS_BIRTH', pd.Series([0])).mean() / 365.25
            employment_years = -segment_data.get('DAYS_EMPLOYED', pd.Series([0])).mean() / 365.25

            # Risk score calculation
            risk_score = self._calculate_risk_score(default_rate, income_credit_ratio, avg_age, employment_years)

            risk_analysis[segment_id] = {
                'segment_size': len(segment_data),
                'default_rate': default_rate,
                'income_credit_ratio': income_credit_ratio,
                'avg_age': avg_age,
                'employment_years': employment_years,
                'risk_score': risk_score,
                'risk_level': self._categorize_risk(risk_score)
            }

        self.segment_risk_profiles = risk_analysis
        return risk_analysis

    def _calculate_risk_score(self, default_rate: float, income_ratio: float, 
                            age: float, employment: float) -> float:
        """Calculate composite risk score"""
        # Normalize and weight factors
        risk_score = (
            default_rate * 0.4 +  # Historical default rate (40%)
            (1 / max(income_ratio, 0.1)) * 0.3 +  # Income adequacy (30%)
            (1 / max(age, 18)) * 0.15 +  # Age factor (15%)
            (1 / max(employment, 0.1)) * 0.15  # Employment stability (15%)
        )

        return min(max(risk_score, 0), 1)  # Normalize to 0-1

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score < 0.3:
            return 'Low'
        elif risk_score < 0.6:
            return 'Medium'
        else:
            return 'High'

class BusinessIntelligenceEngine:
    """Comprehensive business intelligence and insights generation"""

    def __init__(self):
        self.insights = {}
        self.recommendations = {}

    def generate_insights(self, df: pd.DataFrame, segments: np.ndarray, 
                         risk_analysis: Dict) -> Dict:
        """Generate comprehensive business insights"""
        insights = {
            'portfolio_overview': self._analyze_portfolio(df, segments),
            'segment_insights': self._analyze_segments(risk_analysis),
            'risk_insights': self._analyze_risk_patterns(risk_analysis),
            'business_recommendations': self._generate_recommendations(risk_analysis)
        }

        self.insights = insights
        return insights

    def _analyze_portfolio(self, df: pd.DataFrame, segments: np.ndarray) -> Dict:
        """Analyze overall portfolio characteristics"""
        return {
            'total_customers': len(df),
            'total_segments': len(np.unique(segments)),
            'avg_credit_amount': df.get('AMT_CREDIT', pd.Series([0])).mean(),
            'avg_income': df.get('AMT_INCOME_TOTAL', pd.Series([0])).mean(),
            'overall_default_rate': df.get('TARGET', pd.Series([0])).mean()
        }

    def _analyze_segments(self, risk_analysis: Dict) -> List[Dict]:
        """Analyze individual segment characteristics"""
        segment_insights = []

        for segment_id, metrics in risk_analysis.items():
            insights = {
                'segment_id': segment_id,
                'size_percentage': (metrics['segment_size'] / 
                                  sum(m['segment_size'] for m in risk_analysis.values())) * 100,
                'risk_profile': metrics['risk_level'],
                'key_characteristics': self._identify_segment_characteristics(metrics)
            }
            segment_insights.append(insights)

        return segment_insights

    def _identify_segment_characteristics(self, metrics: Dict) -> List[str]:
        """Identify key characteristics of a segment"""
        characteristics = []

        if metrics['default_rate'] > 0.15:
            characteristics.append('High default risk')
        elif metrics['default_rate'] < 0.05:
            characteristics.append('Low default risk')

        if metrics['income_credit_ratio'] > 2:
            characteristics.append('Conservative borrowing')
        elif metrics['income_credit_ratio'] < 1:
            characteristics.append('High leverage')

        if metrics['avg_age'] > 50:
            characteristics.append('Mature customers')
        elif metrics['avg_age'] < 35:
            characteristics.append('Young customers')

        return characteristics

    def _analyze_risk_patterns(self, risk_analysis: Dict) -> Dict:
        """Analyze risk patterns across segments"""
        risk_levels = [metrics['risk_level'] for metrics in risk_analysis.values()]

        return {
            'high_risk_segments': sum(1 for level in risk_levels if level == 'High'),
            'medium_risk_segments': sum(1 for level in risk_levels if level == 'Medium'),
            'low_risk_segments': sum(1 for level in risk_levels if level == 'Low'),
            'avg_portfolio_risk': np.mean([metrics['risk_score'] for metrics in risk_analysis.values()])
        }

    def _generate_recommendations(self, risk_analysis: Dict) -> List[str]:
        """Generate actionable business recommendations"""
        recommendations = []

        high_risk_segments = [sid for sid, metrics in risk_analysis.items() 
                            if metrics['risk_level'] == 'High']

        if high_risk_segments:
            recommendations.append(f"Implement enhanced monitoring for {len(high_risk_segments)} high-risk segments")
            recommendations.append("Consider tightening credit policies for high-risk segments")

        low_risk_segments = [sid for sid, metrics in risk_analysis.items() 
                           if metrics['risk_level'] == 'Low']

        if low_risk_segments:
            recommendations.append(f"Expand marketing efforts to {len(low_risk_segments)} low-risk segments")
            recommendations.append("Consider preferential rates for low-risk customer segments")

        return recommendations

def create_segmentation_report(df: pd.DataFrame) -> Dict:
    """Create comprehensive segmentation and risk analysis report"""
    # Initialize engines
    segmentation_engine = CustomerSegmentationEngine(n_clusters=5)
    risk_analyzer = RiskSegmentAnalyzer()
    bi_engine = BusinessIntelligenceEngine()

    # Perform segmentation
    segments = segmentation_engine.fit_predict(df)

    # Analyze risks
    risk_analysis = risk_analyzer.analyze_segment_risk(df, segments)

    # Generate insights
    insights = bi_engine.generate_insights(df, segments, risk_analysis)

    # Compile comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'segment_profiles': segmentation_engine.segment_profiles,
        'risk_analysis': risk_analysis,
        'business_insights': insights,
        'segments': segments.tolist()
    }

    return report
