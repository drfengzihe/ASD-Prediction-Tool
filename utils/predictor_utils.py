#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for the ASD prediction tool
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelSimulator:
    """Simulates APLR model behavior for demonstration"""

    def __init__(self):
        # L3-4 model coefficients (simplified)
        self.l34_coefficients = {
            'L3-4 EBQ': 0.35,
            'L3-4 pfirrmann grade': 0.25,
            'L3-4 foraminal stenosis': 0.20,
            'L3-4 osteoarthritis of facet joints': 0.15,
            'Age': 0.02,
            'L3-4 local lordosis angle': -0.08,
            'intercept': -2.5
        }

        # L5-S1 model coefficients (simplified)
        self.l5s1_coefficients = {
            'L5-S1 osteoarthritis of facet joints': 0.40,
            'L3-4 EBQ': 0.25,
            'L5-S1 local lordosis angle': -0.12,
            'L5-S1 modic change': 0.18,
            'L5-S1 foraminal stenosis': 0.15,
            'Age': 0.01,
            'intercept': -1.8
        }

    def predict(self, features, dataset_name):
        """Simulate APLR prediction"""
        if dataset_name == "L34 Result":
            coeffs = self.l34_coefficients
        else:
            coeffs = self.l5s1_coefficients

        # Calculate linear combination
        score = coeffs['intercept']
        for feature, coeff in coeffs.items():
            if feature != 'intercept' and feature in features:
                score += coeff * features[feature]

        # Apply sigmoid for probability
        probability = 1 / (1 + np.exp(-score))
        prediction = 1 if probability > 0.5 else 0

        return {
            'prediction': prediction,
            'probability': [1 - probability, probability],
            'confidence': max(probability, 1 - probability),
            'raw_score': score
        }

    def get_feature_importance(self, dataset_name):
        """Get feature importance for visualization"""
        if dataset_name == "L34 Result":
            coeffs = self.l34_coefficients
        else:
            coeffs = self.l5s1_coefficients

        # Remove intercept and get absolute values
        importance = {k: abs(v) for k, v in coeffs.items() if k != 'intercept'}

        # Normalize
        total = sum(importance.values())
        importance = {k: v / total for k, v in importance.items()}

        return importance


class AnchorSimulator:
    """Simulates anchor explanations"""

    def __init__(self):
        # Pre-defined anchor rules
        self.l34_anchors = [
            {
                'condition': 'L3-4 EBQ > 3.6 AND L3-4 Pfirrmann Grade ≥ 4',
                'features': ['L3-4 EBQ', 'L3-4 pfirrmann grade'],
                'precision': 0.95,
                'coverage': 0.12,
                'check': lambda f: f.get('L3-4 EBQ', 0) > 3.6 and f.get('L3-4 pfirrmann grade', 0) >= 4
            },
            {
                'condition': 'L3-4 EBQ > 4.0 AND L3-4 Foraminal Stenosis ≥ 2',
                'features': ['L3-4 EBQ', 'L3-4 foraminal stenosis'],
                'precision': 0.92,
                'coverage': 0.08,
                'check': lambda f: f.get('L3-4 EBQ', 0) > 4.0 and f.get('L3-4 foraminal stenosis', 0) >= 2
            }
        ]

        self.l5s1_anchors = [
            {
                'condition': 'L5-S1 Facet Joint OA ≥ Moderate AND L3-4 EBQ > 3.0',
                'features': ['L5-S1 osteoarthritis of facet joints', 'L3-4 EBQ'],
                'precision': 0.93,
                'coverage': 0.15,
                'check': lambda f: f.get('L5-S1 osteoarthritis of facet joints', 0) >= 2 and f.get('L3-4 EBQ', 0) > 3.0
            },
            {
                'condition': 'L5-S1 Modic Changes ≥ Type 2 AND L5-S1 Foraminal Stenosis ≥ 2',
                'features': ['L5-S1 modic change', 'L5-S1 foraminal stenosis'],
                'precision': 0.89,
                'coverage': 0.10,
                'check': lambda f: f.get('L5-S1 modic change', 0) >= 2 and f.get('L5-S1 foraminal stenosis', 0) >= 2
            }
        ]

    def find_anchor(self, features, dataset_name, prediction):
        """Find applicable anchor for given features"""
        if prediction == 0:  # Low risk, no anchors needed
            return None

        anchors = self.l34_anchors if dataset_name == "L34 Result" else self.l5s1_anchors

        for anchor in anchors:
            if anchor['check'](features):
                return {
                    'anchor_features': anchor['features'],
                    'precision': anchor['precision'],
                    'coverage': anchor['coverage'],
                    'description': anchor['condition']
                }

        return None


class CounterfactualSimulator:
    """Simulates counterfactual analysis"""

    def __init__(self):
        self.interventional_features = ['Cage height', 'Operative time ', 'Blood loss']
        self.feature_constraints = {
            'Cage height': (8.0, 16.0),
            'Operative time ': (60.0, 300.0),
            'Blood loss': (50.0, 500.0)
        }

    def generate_counterfactual(self, features, dataset_name, current_prediction):
        """Generate counterfactual explanation"""
        target_class = 1 - current_prediction  # Flip the prediction

        if current_prediction == 1:  # High risk -> Low risk
            return self._generate_risk_reduction_cf(features, dataset_name)
        else:  # Low risk -> High risk
            return self._generate_risk_increase_cf(features, dataset_name)

    def _generate_risk_reduction_cf(self, features, dataset_name):
        """Generate counterfactual for risk reduction"""
        changes = {}

        if dataset_name == "L34 Result":
            # For L3-4, focus on reducing EBQ or improving surgical parameters
            if features.get('L3-4 EBQ', 3.0) > 3.0:
                changes['L3-4 EBQ'] = {
                    'original': features.get('L3-4 EBQ', 3.5),
                    'counterfactual': 2.8,
                    'change': 2.8 - features.get('L3-4 EBQ', 3.5)
                }

            # Surgical optimization
            if features.get('Operative time ', 150) > 120:
                changes['Operative time '] = {
                    'original': features.get('Operative time ', 150),
                    'counterfactual': 110,
                    'change': 110 - features.get('Operative time ', 150)
                }

        else:  # L5S1 Result
            # For L5-S1, focus on facet joint OA and other factors
            if features.get('L5-S1 osteoarthritis of facet joints', 2) >= 2:
                changes['L5-S1 osteoarthritis of facet joints'] = {
                    'original': features.get('L5-S1 osteoarthritis of facet joints', 2),
                    'counterfactual': 1,
                    'change': 1 - features.get('L5-S1 osteoarthritis of facet joints', 2)
                }

        return {
            'success': bool(changes),
            'target_class': "Low Risk",
            'changes': changes
        }

    def _generate_risk_increase_cf(self, features, dataset_name):
        """Generate counterfactual for risk increase (educational purposes)"""
        changes = {}

        # General surgical risk factors
        if features.get('Operative time ', 120) < 180:
            changes['Operative time '] = {
                'original': features.get('Operative time ', 120),
                'counterfactual': 200,
                'change': 200 - features.get('Operative time ', 120)
            }

        if features.get('Blood loss', 150) < 250:
            changes['Blood loss'] = {
                'original': features.get('Blood loss', 150),
                'counterfactual': 280,
                'change': 280 - features.get('Blood loss', 150)
            }

        return {
            'success': bool(changes),
            'target_class': "High Risk",
            'changes': changes
        }


def create_probability_chart(probabilities, title="Prediction Probabilities"):
    """Create probability visualization"""
    prob_data = pd.DataFrame({
        'Outcome': ['No ASD', 'ASD'],
        'Probability': probabilities
    })

    fig = px.bar(
        prob_data, x='Outcome', y='Probability',
        title=title,
        color='Outcome',
        color_discrete_map={'No ASD': '#81c784', 'ASD': '#e57373'}
    )

    fig.update_layout(
        height=300,
        showlegend=False,
        yaxis=dict(range=[0, 1])
    )

    return fig


def create_feature_importance_chart(importance_dict, title="Feature Importance"):
    """Create feature importance visualization"""
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())

    fig = px.bar(
        x=importances,
        y=features,
        orientation='h',
        title=title,
        labels={'x': 'Importance', 'y': 'Features'}
    )

    fig.update_layout(height=400)
    return fig


def create_anchor_quality_chart(precision, coverage):
    """Create anchor quality visualization"""
    fig = go.Figure()

    # Add current anchor point
    fig.add_trace(go.Scatter(
        x=[coverage],
        y=[precision],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Current Anchor',
        text=[f'Precision: {precision:.3f}<br>Coverage: {coverage:.3f}'],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add quality regions
    fig.add_shape(
        type="rect", x0=0.1, y0=0.9, x1=1, y1=1,
        fillcolor="lightgreen", opacity=0.3, line_width=0,
    )

    fig.add_shape(
        type="rect", x0=0.05, y0=0.8, x1=1, y1=0.9,
        fillcolor="lightyellow", opacity=0.3, line_width=0,
    )

    fig.update_layout(
        title='Anchor Quality Assessment',
        xaxis_title='Coverage',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )

    # Add quality annotations
    fig.add_annotation(x=0.55, y=0.95, text="High Quality", showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=0.5, y=0.85, text="Good Quality", showarrow=False, font=dict(color="orange"))

    return fig


def validate_feature_input(feature_values, feature_ranges):
    """Validate feature inputs"""
    errors = []
    warnings = []

    for feature, value in feature_values.items():
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]

            if value < min_val or value > max_val:
                errors.append(f"{feature}: {value} is outside valid range ({min_val}-{max_val})")
            elif value < min_val * 1.1 or value > max_val * 0.9:
                warnings.append(f"{feature}: {value} is near the boundary of typical range")

    return errors, warnings


def format_clinical_recommendation(changes, prediction):
    """Format clinical recommendations based on counterfactual analysis"""
    recommendations = []

    if prediction == 1:  # High risk
        recommendations.append("Patient classified as HIGH RISK for ASD development")
        recommendations.append("Consider enhanced monitoring and preventive strategies")

        for feature, change_info in changes.items():
            if feature == 'Operative time ':
                if change_info['change'] < 0:
                    recommendations.append(
                        f"Optimize surgical technique to reduce operative time by {abs(change_info['change']):.0f} minutes")
            elif feature == 'Blood loss':
                if change_info['change'] < 0:
                    recommendations.append(
                        f"Implement enhanced hemostatic measures to reduce blood loss by {abs(change_info['change']):.0f}ml")
            elif feature == 'Cage height':
                if change_info['change'] != 0:
                    direction = "increase" if change_info['change'] > 0 else "decrease"
                    recommendations.append(f"Consider {direction} in cage height by {abs(change_info['change']):.1f}mm")

    else:  # Low risk
        recommendations.append("Patient classified as LOW RISK for ASD development")
        recommendations.append("Standard postoperative monitoring recommended")

    return recommendations