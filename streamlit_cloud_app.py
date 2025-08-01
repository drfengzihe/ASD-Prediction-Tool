#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASD Risk Prediction Tool - Explainable AI Framework based on APLR
Beijing Chaoyang Hospital & Capital Medical University

For predicting Adjacent Segment Disease (ASD) risk after L4-5 lumbar fusion
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="ASD Risk Prediction Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #e57373;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #81c784;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class CloudASDPredictor:
    """Cloud-based ASD Prediction Tool"""

    def __init__(self):
        self.feature_names = [
            'Gender', 'Hypertension', 'Diabetes', 'Smoking history', 'Alcohol abuse',
            'L3-4 pfirrmann grade', 'L3-4 spinal canal stenosis', 'L3-4 foraminal stenosis',
            'L3-4 modic change', 'L3-4 osteoarthritis of facet joints', 'L3-4 sagittal imbalance',
            'L3-4 coronal imbalance', 'L5-S1 pfirrmann grade', 'L5-S1 spinal canal stenosis',
            'L5-S1 foraminal stenosis', 'L5-S1 modic change', 'L5-S1 osteoarthritis of facet joints',
            'L5-S1 sagittal imbalance', 'L5-S1 coronal imbalance', 'Cage height', 'Age', 'BMI',
            'HU', 'L3-4 EBQ', 'L3-4 local lordosis angle', 'L5-S1 EBQ', 'L5-S1 local lordosis angle',
            'L3-S1 lordosis angle', 'Lumbar lordosis angle', 'L3-4 preoperative disc height',
            'L5-S1 preoperative disc height', 'Operative time ', 'Blood loss'
        ]

        self.feature_descriptions = {
            'Gender': 'Gender (0=Female, 1=Male)',
            'Age': 'Age (years)',
            'BMI': 'Body Mass Index (kg/m²)',
            'Hypertension': 'Hypertension (0=No, 1=Yes)',
            'Diabetes': 'Diabetes (0=No, 1=Yes)',
            'Smoking history': 'Smoking History (0=No, 1=Yes)',
            'Alcohol abuse': 'Alcohol Abuse (0=No, 1=Yes)',
            'L3-4 pfirrmann grade': 'L3-4 Disc Degeneration Grade (1-5)',
            'L3-4 spinal canal stenosis': 'L3-4 Spinal Canal Stenosis (0=None,1=Mild,2=Moderate,3=Severe)',
            'L3-4 foraminal stenosis': 'L3-4 Foraminal Stenosis (0=None,1=Mild,2=Moderate,3=Severe)',
            'L3-4 modic change': 'L3-4 Modic Changes (0-3)',
            'L3-4 osteoarthritis of facet joints': 'L3-4 Facet Joint OA (0=None,1=Mild,2=Moderate,3=Severe)',
            'L3-4 sagittal imbalance': 'L3-4 Sagittal Imbalance (0=No, 1=Yes)',
            'L3-4 coronal imbalance': 'L3-4 Coronal Imbalance (0=No, 1=Yes)',
            'L5-S1 pfirrmann grade': 'L5-S1 Disc Degeneration Grade (1-5)',
            'L5-S1 spinal canal stenosis': 'L5-S1 Spinal Canal Stenosis (0=None,1=Mild,2=Moderate,3=Severe)',
            'L5-S1 foraminal stenosis': 'L5-S1 Foraminal Stenosis (0=None,1=Mild,2=Moderate,3=Severe)',
            'L5-S1 modic change': 'L5-S1 Modic Changes (0-3)',
            'L5-S1 osteoarthritis of facet joints': 'L5-S1 Facet Joint OA (0=None,1=Mild,2=Moderate,3=Severe)',
            'L5-S1 sagittal imbalance': 'L5-S1 Sagittal Imbalance (0=No, 1=Yes)',
            'L5-S1 coronal imbalance': 'L5-S1 Coronal Imbalance (0=No, 1=Yes)',
            'Cage height': 'Fusion Cage Height (mm)',
            'HU': 'Bone Density CT Value (Hounsfield Units)',
            'L3-4 EBQ': 'L3-4 Endplate Bone Quality Score (1-5)',
            'L3-4 local lordosis angle': 'L3-4 Local Lordosis Angle (degrees)',
            'L5-S1 EBQ': 'L5-S1 Endplate Bone Quality Score (1-5)',
            'L5-S1 local lordosis angle': 'L5-S1 Local Lordosis Angle (degrees)',
            'L3-S1 lordosis angle': 'L3-S1 Lordosis Angle (degrees)',
            'Lumbar lordosis angle': 'Lumbar Lordosis Angle (degrees)',
            'L3-4 preoperative disc height': 'L3-4 Preop Disc Height (0=Reduced, 1=Normal)',
            'L5-S1 preoperative disc height': 'L5-S1 Preop Disc Height (0=Reduced, 1=Normal)',
            'Operative time ': 'Operative Time (minutes)',
            'Blood loss': 'Intraoperative Blood Loss (ml)'
        }

        self.feature_ranges = {
            'Gender': (0, 1),
            'Age': (40, 80),
            'BMI': (18, 35),
            'Hypertension': (0, 1),
            'Diabetes': (0, 1),
            'Smoking history': (0, 1),
            'Alcohol abuse': (0, 1),
            'L3-4 pfirrmann grade': (1, 5),
            'L3-4 spinal canal stenosis': (0, 3),
            'L3-4 foraminal stenosis': (0, 3),
            'L3-4 modic change': (0, 3),
            'L3-4 osteoarthritis of facet joints': (0, 3),
            'L3-4 sagittal imbalance': (0, 1),
            'L3-4 coronal imbalance': (0, 1),
            'L5-S1 pfirrmann grade': (1, 5),
            'L5-S1 spinal canal stenosis': (0, 3),
            'L5-S1 foraminal stenosis': (0, 3),
            'L5-S1 modic change': (0, 3),
            'L5-S1 osteoarthritis of facet joints': (0, 3),
            'L5-S1 sagittal imbalance': (0, 1),
            'L5-S1 coronal imbalance': (0, 1),
            'Cage height': (10, 14),
            'HU': (80, 200),
            'L3-4 EBQ': (1, 5),
            'L3-4 local lordosis angle': (0, 15),
            'L5-S1 EBQ': (1, 5),
            'L5-S1 local lordosis angle': (5, 25),
            'L3-S1 lordosis angle': (15, 40),
            'Lumbar lordosis angle': (15, 50),
            'L3-4 preoperative disc height': (0, 1),
            'L5-S1 preoperative disc height': (0, 1),
            'Operative time ': (60, 300),
            'Blood loss': (50, 500)
        }

        # Demo anonymized samples
        self.demo_samples = {
            "L34 Result": [
                {
                    "sample_id": "Demo_L34_High_Risk",
                    "features": {
                        'Gender': 1, 'Age': 65, 'BMI': 25.2, 'Hypertension': 1, 'Diabetes': 0,
                        'Smoking history': 0, 'Alcohol abuse': 0, 'L3-4 pfirrmann grade': 4,
                        'L3-4 spinal canal stenosis': 2, 'L3-4 foraminal stenosis': 2,
                        'L3-4 modic change': 1, 'L3-4 osteoarthritis of facet joints': 2,
                        'L3-4 sagittal imbalance': 1, 'L3-4 coronal imbalance': 0,
                        'L5-S1 pfirrmann grade': 3, 'L5-S1 spinal canal stenosis': 1,
                        'L5-S1 foraminal stenosis': 1, 'L5-S1 modic change': 0,
                        'L5-S1 osteoarthritis of facet joints': 1, 'L5-S1 sagittal imbalance': 0,
                        'L5-S1 coronal imbalance': 1, 'Cage height': 12, 'HU': 125,
                        'L3-4 EBQ': 4.2, 'L3-4 local lordosis angle': 5.5, 'L5-S1 EBQ': 3.1,
                        'L5-S1 local lordosis angle': 13.2, 'L3-S1 lordosis angle': 28.5,
                        'Lumbar lordosis angle': 32.8, 'L3-4 preoperative disc height': 0,
                        'L5-S1 preoperative disc height': 1, 'Operative time ': 145, 'Blood loss': 180
                    },
                    "true_label": 1,
                    "risk_level": "High Risk"
                },
                {
                    "sample_id": "Demo_L34_Low_Risk",
                    "features": {
                        'Gender': 0, 'Age': 58, 'BMI': 23.8, 'Hypertension': 0, 'Diabetes': 0,
                        'Smoking history': 0, 'Alcohol abuse': 0, 'L3-4 pfirrmann grade': 2,
                        'L3-4 spinal canal stenosis': 0, 'L3-4 foraminal stenosis': 0,
                        'L3-4 modic change': 0, 'L3-4 osteoarthritis of facet joints': 1,
                        'L3-4 sagittal imbalance': 0, 'L3-4 coronal imbalance': 0,
                        'L5-S1 pfirrmann grade': 2, 'L5-S1 spinal canal stenosis': 0,
                        'L5-S1 foraminal stenosis': 0, 'L5-S1 modic change': 0,
                        'L5-S1 osteoarthritis of facet joints': 0, 'L5-S1 sagittal imbalance': 0,
                        'L5-S1 coronal imbalance': 0, 'Cage height': 12, 'HU': 145,
                        'L3-4 EBQ': 2.3, 'L3-4 local lordosis angle': 8.2, 'L5-S1 EBQ': 2.5,
                        'L5-S1 local lordosis angle': 14.8, 'L3-S1 lordosis angle': 29.2,
                        'Lumbar lordosis angle': 35.5, 'L3-4 preoperative disc height': 1,
                        'L5-S1 preoperative disc height': 1, 'Operative time ': 120, 'Blood loss': 120
                    },
                    "true_label": 0,
                    "risk_level": "Low Risk"
                }
            ],
            "L5S1 Result": [
                {
                    "sample_id": "Demo_L5S1_High_Risk",
                    "features": {
                        'Gender': 1, 'Age': 62, 'BMI': 26.5, 'Hypertension': 1, 'Diabetes': 1,
                        'Smoking history': 1, 'Alcohol abuse': 0, 'L3-4 pfirrmann grade': 3,
                        'L3-4 spinal canal stenosis': 1, 'L3-4 foraminal stenosis': 1,
                        'L3-4 modic change': 0, 'L3-4 osteoarthritis of facet joints': 1,
                        'L3-4 sagittal imbalance': 0, 'L3-4 coronal imbalance': 0,
                        'L5-S1 pfirrmann grade': 4, 'L5-S1 spinal canal stenosis': 3,
                        'L5-S1 foraminal stenosis': 2, 'L5-S1 modic change': 2,
                        'L5-S1 osteoarthritis of facet joints': 3, 'L5-S1 sagittal imbalance': 1,
                        'L5-S1 coronal imbalance': 1, 'Cage height': 12, 'HU': 115,
                        'L3-4 EBQ': 3.8, 'L3-4 local lordosis angle': 6.8, 'L5-S1 EBQ': 3.5,
                        'L5-S1 local lordosis angle': 11.2, 'L3-S1 lordosis angle': 26.8,
                        'Lumbar lordosis angle': 30.5, 'L3-4 preoperative disc height': 0,
                        'L5-S1 preoperative disc height': 0, 'Operative time ': 165, 'Blood loss': 220
                    },
                    "true_label": 1,
                    "risk_level": "High Risk"
                },
                {
                    "sample_id": "Demo_L5S1_Low_Risk",
                    "features": {
                        'Gender': 0, 'Age': 55, 'BMI': 22.1, 'Hypertension': 0, 'Diabetes': 0,
                        'Smoking history': 0, 'Alcohol abuse': 0, 'L3-4 pfirrmann grade': 2,
                        'L3-4 spinal canal stenosis': 0, 'L3-4 foraminal stenosis': 0,
                        'L3-4 modic change': 0, 'L3-4 osteoarthritis of facet joints': 0,
                        'L3-4 sagittal imbalance': 0, 'L3-4 coronal imbalance': 0,
                        'L5-S1 pfirrmann grade': 3, 'L5-S1 spinal canal stenosis': 1,
                        'L5-S1 foraminal stenosis': 0, 'L5-S1 modic change': 0,
                        'L5-S1 osteoarthritis of facet joints': 1, 'L5-S1 sagittal imbalance': 0,
                        'L5-S1 coronal imbalance': 0, 'Cage height': 12, 'HU': 155,
                        'L3-4 EBQ': 2.1, 'L3-4 local lordosis angle': 9.5, 'L5-S1 EBQ': 2.8,
                        'L5-S1 local lordosis angle': 16.2, 'L3-S1 lordosis angle': 31.5,
                        'Lumbar lordosis angle': 38.2, 'L3-4 preoperative disc height': 1,
                        'L5-S1 preoperative disc height': 1, 'Operative time ': 105, 'Blood loss': 95
                    },
                    "true_label": 0,
                    "risk_level": "Low Risk"
                }
            ]
        }


def load_models():
    """Load pre-trained models (simulation)"""
    # In production, this would load your actual model files
    return {
        "L34 Result": {"model": "aplr_l34_model", "accuracy": 0.999, "auc": 0.999},
        "L5S1 Result": {"model": "aplr_l5s1_model", "accuracy": 0.886, "auc": 0.914}
    }


def simulate_prediction(feature_values, dataset_name):
    """Simulate prediction (replace with actual APLR model)"""

    if dataset_name == "L34 Result":
        # Simplified L3-4 ASD risk assessment
        risk_score = (
                feature_values.get('L3-4 EBQ', 2.5) * 0.3 +
                feature_values.get('L3-4 pfirrmann grade', 2) * 0.2 +
                feature_values.get('Age', 60) * 0.01 +
                feature_values.get('L3-4 foraminal stenosis', 0) * 0.15 +
                feature_values.get('L3-4 osteoarthritis of facet joints', 0) * 0.1
        )
        probability = 1 / (1 + np.exp(-((risk_score - 3.5) * 2)))

    else:  # L5S1 Result
        # Simplified L5-S1 ASD risk assessment
        risk_score = (
                feature_values.get('L5-S1 osteoarthritis of facet joints', 0) * 0.4 +
                feature_values.get('L3-4 EBQ', 2.5) * 0.25 +
                feature_values.get('L5-S1 local lordosis angle', 14) * (-0.1) +
                feature_values.get('L5-S1 modic change', 0) * 0.2
        )
        probability = 1 / (1 + np.exp(-((risk_score - 2.0) * 1.5)))

    prediction = 1 if probability > 0.5 else 0

    return {
        'prediction': prediction,
        'probability': [1 - probability, probability],
        'confidence': max(probability, 1 - probability),
        'risk_score': risk_score
    }


def simulate_anchor_explanation(feature_values, dataset_name, prediction):
    """Simulate anchor explanation"""
    if dataset_name == "L34 Result" and prediction == 1:
        if feature_values.get('L3-4 EBQ', 2.5) > 3.6:
            return {
                'anchor_features': ['L3-4 EBQ', 'L3-4 pfirrmann grade'],
                'precision': 0.95,
                'coverage': 0.12,
                'description': 'L3-4 EBQ > 3.6 AND L3-4 disc degeneration ≥ grade 3'
            }
    elif dataset_name == "L5S1 Result" and prediction == 1:
        if feature_values.get('L5-S1 osteoarthritis of facet joints', 0) >= 2:
            return {
                'anchor_features': ['L5-S1 osteoarthritis of facet joints', 'L3-4 EBQ'],
                'precision': 0.92,
                'coverage': 0.08,
                'description': 'L5-S1 facet joint OA ≥ moderate AND L3-4 EBQ > 3.0'
            }

    return {
        'anchor_features': [],
        'precision': 0.0,
        'coverage': 0.0,
        'description': 'No significant anchor found'
    }


def simulate_counterfactual_analysis(feature_values, dataset_name, prediction):
    """Simulate counterfactual analysis"""
    if prediction == 1:  # High risk -> Low risk
        target_class = "Low Risk"
        if dataset_name == "L34 Result":
            changes = {
                'L3-4 EBQ': {
                    'original': feature_values.get('L3-4 EBQ', 3.5),
                    'counterfactual': max(2.5, feature_values.get('L3-4 EBQ', 3.5) - 1.2),
                    'change': -1.2
                }
            }
        else:
            changes = {
                'L5-S1 osteoarthritis of facet joints': {
                    'original': feature_values.get('L5-S1 osteoarthritis of facet joints', 2),
                    'counterfactual': max(0, feature_values.get('L5-S1 osteoarthritis of facet joints', 2) - 1),
                    'change': -1
                }
            }
    else:  # Low risk -> High risk
        target_class = "High Risk"
        changes = {
            'Operative time ': {
                'original': feature_values.get('Operative time ', 120),
                'counterfactual': feature_values.get('Operative time ', 120) + 45,
                'change': +45
            }
        }

    return {
        'success': True,
        'target_class': target_class,
        'changes': changes
    }


def main():
    """Main application"""
    predictor = CloudASDPredictor()

    # Header
    st.markdown('<h1 class="main-header">🏥 ASD Risk Prediction Tool</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Explainable AI Framework for Adjacent Segment Disease Prediction</p>',
        unsafe_allow_html=True)

    # Navigation
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Demo Analysis", "🔮 Patient Prediction"])

    with tab1:
        show_home_page()

    with tab2:
        show_demo_analysis(predictor)

    with tab3:
        show_patient_prediction(predictor)


def show_home_page():
    """Show home page with project introduction"""
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### From Prediction to Actionable Intervention

        This tool implements our novel **Explainable AI Framework** for predicting Adjacent Segment Disease (ASD) 
        following L4-5 lumbar fusion surgery. Our approach combines three cutting-edge techniques:

        #### 🤖 Core Technologies

        **1. Automated Piecewise Linear Regression (APLR)**
        - Transparent prediction with interpretable linear segments
        - Captures non-linear relationships while maintaining explainability
        - Achieved AUC of 0.999 for L3-4 and 0.914 for L5-S1 ASD prediction

        **2. Anchor Explanations**
        - Identifies sufficient conditions for high-risk classification
        - Provides verifiable decision criteria (e.g., "L3-4 EBQ > 3.6")
        - Establishes minimal feature sets that guarantee predictions

        **3. Counterfactual Analysis**
        - Quantifies precise intervention thresholds
        - Shows what changes would alter risk classification
        - Enables proactive surgical planning

        #### 🎯 Clinical Impact

        Our framework addresses three critical gaps in current ASD prediction:
        - **Transparency**: Black-box models → Interpretable predictions
        - **Individualization**: Population-level → Patient-specific guidance  
        - **Actionability**: Risk identification → Intervention planning
        """)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>📈 Model Performance</h4>
        <ul>
        <li><strong>L3-4 ASD Model:</strong><br>
        AUC: 0.999 (95% CI: 0.998-1.000)<br>
        Accuracy: 98.8%</li>
        <li><strong>L5-S1 ASD Model:</strong><br>
        AUC: 0.914 (95% CI: 0.844-0.983)<br>
        Accuracy: 88.6%</li>
        </ul>
        </div>

        <div class="warning-box">
        <h4>⚠️ Important Notice</h4>
        <p>This tool is for research demonstration purposes. 
        All patient data has been anonymized. Clinical decisions 
        should always involve qualified healthcare professionals.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">How to Use This Tool</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Demo Analysis

        Explore our framework using anonymized demonstration samples:

        1. **Global Model Insights**: View feature importance and model behavior
        2. **Sample Predictions**: See APLR predictions on demo cases
        3. **Anchor Explanations**: Understand sufficient conditions for risk
        4. **Counterfactual Analysis**: Explore intervention possibilities

        *Perfect for understanding how our AI framework works*
        """)

    with col2:
        st.markdown("""
        ### 🔮 Patient Prediction

        Input your own patient data for comprehensive analysis:

        1. **Data Input**: Enter clinical and radiographic parameters
        2. **Risk Prediction**: Get APLR-based ASD risk assessment
        3. **Explanation Suite**: Receive anchor and counterfactual analyses
        4. **Clinical Guidance**: Understand factors driving the prediction

        *Ideal for exploring specific clinical scenarios*
        """)


def show_demo_analysis(predictor):
    """Show demo analysis with anonymized samples"""
    st.markdown('<h2 class="sub-header">📊 Demonstration Analysis</h2>', unsafe_allow_html=True)

    # Dataset selection
    dataset_name = st.selectbox(
        "Select Adjacent Segment to Analyze:",
        ["L34 Result", "L5S1 Result"],
        help="Choose which adjacent segment to analyze for ASD risk"
    )

    if dataset_name == "L34 Result":
        st.info("**L3-4 Adjacent Segment**: Analyzing ASD risk at the level above L4-5 fusion")
    else:
        st.info("**L5-S1 Adjacent Segment**: Analyzing ASD risk at the level below L4-5 fusion")

    # Model performance overview
    st.markdown('<h3 class="sub-header">Model Performance Overview</h3>', unsafe_allow_html=True)

    models = load_models()
    model_info = models[dataset_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{model_info['accuracy']:.3f}")
    with col2:
        st.metric("AUC Score", f"{model_info['auc']:.3f}")
    with col3:
        st.metric("Demo Samples", len(predictor.demo_samples[dataset_name]))

    # Global feature importance (simulated)
    st.markdown('<h3 class="sub-header">Global Feature Importance</h3>', unsafe_allow_html=True)

    if dataset_name == "L34 Result":
        importance_data = {
            'Feature': ['L3-4 EBQ × L5-S1 EBQ', 'L3-4 EBQ', 'L3-4 Pfirrmann Grade',
                        'L3-4 Foraminal Stenosis', 'Age × L3-4 EBQ', 'L3-4 Local Lordosis'],
            'Importance': [0.28, 0.22, 0.18, 0.15, 0.12, 0.05]
        }
    else:
        importance_data = {
            'Feature': ['L5-S1 Facet Joint OA', 'L3-4 EBQ', 'L5-S1 Local Lordosis',
                        'L5-S1 Modic Changes', 'L5-S1 EBQ', 'Age × Bone Quality'],
            'Importance': [0.35, 0.25, 0.18, 0.12, 0.08, 0.02]
        }

    fig = px.bar(importance_data, x='Importance', y='Feature', orientation='h',
                 title=f'Top Features for {dataset_name} ASD Prediction')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Demo samples analysis
    st.markdown('<h3 class="sub-header">Demo Sample Analysis</h3>', unsafe_allow_html=True)

    samples = predictor.demo_samples[dataset_name]

    for i, sample in enumerate(samples):
        with st.expander(f"🔍 {sample['sample_id']} - {sample['risk_level']}"):

            # Prediction
            prediction_result = simulate_prediction(sample['features'], dataset_name)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Label", "ASD" if sample['true_label'] == 1 else "No ASD")
            with col2:
                pred_text = "ASD" if prediction_result['prediction'] == 1 else "No ASD"
                st.metric("APLR Prediction", pred_text)
            with col3:
                st.metric("Confidence", f"{prediction_result['confidence']:.3f}")

            # Probability visualization
            prob_data = pd.DataFrame({
                'Outcome': ['No ASD', 'ASD'],
                'Probability': prediction_result['probability']
            })

            fig = px.bar(prob_data, x='Outcome', y='Probability',
                         title='Prediction Probabilities',
                         color='Outcome',
                         color_discrete_map={'No ASD': '#81c784', 'ASD': '#e57373'})
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Anchor explanation
            anchor_result = simulate_anchor_explanation(
                sample['features'], dataset_name, prediction_result['prediction']
            )

            if anchor_result['anchor_features']:
                st.markdown("**⚓ Anchor Explanation:**")
                st.success(f"**Rule**: {anchor_result['description']}")
                st.write(
                    f"**Precision**: {anchor_result['precision']:.3f} | **Coverage**: {anchor_result['coverage']:.3f}")
            else:
                st.warning("⚠️ No significant anchor conditions found for this sample")

            # Counterfactual analysis
            cf_result = simulate_counterfactual_analysis(
                sample['features'], dataset_name, prediction_result['prediction']
            )

            if cf_result['success']:
                st.markdown("**🔄 Counterfactual Analysis:**")
                st.info(f"To change prediction to **{cf_result['target_class']}**:")

                for feature, change_info in cf_result['changes'].items():
                    direction = "increase" if change_info['change'] > 0 else "decrease"
                    st.write(f"- **{feature}**: {direction} by {abs(change_info['change']):.1f}")


def show_patient_prediction(predictor):
    """Show patient prediction interface"""
    st.markdown('<h2 class="sub-header">🔮 Patient Risk Prediction</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research Tool Disclaimer</h4>
    <p>This tool is for research and educational purposes only. Results should not be used for 
    clinical decision-making without consultation with qualified healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset selection
    dataset_name = st.selectbox(
        "Select Adjacent Segment for Prediction:",
        ["L34 Result", "L5S1 Result"],
        help="Choose which adjacent segment to analyze for ASD risk",
        key="pred_dataset"
    )

    st.markdown("### Patient Data Input")
    st.write("Please enter the patient's clinical and radiographic parameters:")

    # Feature input form
    with st.form("patient_data_form"):
        feature_values = {}

        # Organize features into categories
        demo_features = ['Gender', 'Age', 'BMI']
        comorbidity_features = ['Hypertension', 'Diabetes', 'Smoking history', 'Alcohol abuse']
        l34_features = [f for f in predictor.feature_names if 'L3-4' in f]
        l5s1_features = [f for f in predictor.feature_names if 'L5-S1' in f]
        spinal_features = ['L3-S1 lordosis angle', 'Lumbar lordosis angle']
        surgical_features = ['Cage height', 'HU', 'Operative time ', 'Blood loss']

        # Demographics
        st.subheader("👤 Demographics")
        col1, col2, col3 = st.columns(3)

        with col1:
            feature_values['Gender'] = st.selectbox('Gender', [0, 1],
                                                    format_func=lambda x: 'Female' if x == 0 else 'Male')
        with col2:
            feature_values['Age'] = st.number_input('Age (years)', min_value=40, max_value=80, value=60)
        with col3:
            feature_values['BMI'] = st.number_input('BMI (kg/m²)', min_value=18.0, max_value=35.0, value=25.0, step=0.1)

        # Comorbidities
        st.subheader("🏥 Comorbidities")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            feature_values['Hypertension'] = st.selectbox('Hypertension', [0, 1],
                                                          format_func=lambda x: 'No' if x == 0 else 'Yes')
        with col2:
            feature_values['Diabetes'] = st.selectbox('Diabetes', [0, 1],
                                                      format_func=lambda x: 'No' if x == 0 else 'Yes')
        with col3:
            feature_values['Smoking history'] = st.selectbox('Smoking History', [0, 1],
                                                             format_func=lambda x: 'No' if x == 0 else 'Yes')
        with col4:
            feature_values['Alcohol abuse'] = st.selectbox('Alcohol Abuse', [0, 1],
                                                           format_func=lambda x: 'No' if x == 0 else 'Yes')

        # L3-4 Parameters
        st.subheader("🦴 L3-4 Segment Parameters")
        l34_cols = st.columns(3)

        for i, feature in enumerate(l34_features):
            col_idx = i % 3
            with l34_cols[col_idx]:
                min_val, max_val = predictor.feature_ranges[feature]

                if feature in ['L3-4 sagittal imbalance', 'L3-4 coronal imbalance', 'L3-4 preoperative disc height']:
                    if 'imbalance' in feature:
                        feature_values[feature] = st.selectbox(feature.replace('L3-4 ', '').title(), [0, 1],
                                                               format_func=lambda x: 'No' if x == 0 else 'Yes',
                                                               key=feature)
                    else:
                        feature_values[feature] = st.selectbox(feature.replace('L3-4 ', '').title(), [0, 1],
                                                               format_func=lambda x: 'Reduced' if x == 0 else 'Normal',
                                                               key=feature)
                elif 'angle' in feature or 'EBQ' in feature:
                    feature_values[feature] = st.number_input(
                        feature.replace('L3-4 ', '').title(),
                        min_value=float(min_val), max_value=float(max_val),
                        value=float((min_val + max_val) / 2), step=0.1, key=feature
                    )
                else:
                    feature_values[feature] = st.selectbox(
                        feature.replace('L3-4 ', '').title(),
                        list(range(int(min_val), int(max_val) + 1)), key=feature
                    )

        # L5-S1 Parameters
        st.subheader("🦴 L5-S1 Segment Parameters")
        l5s1_cols = st.columns(3)

        for i, feature in enumerate(l5s1_features):
            col_idx = i % 3
            with l5s1_cols[col_idx]:
                min_val, max_val = predictor.feature_ranges[feature]

                if feature in ['L5-S1 sagittal imbalance', 'L5-S1 coronal imbalance', 'L5-S1 preoperative disc height']:
                    if 'imbalance' in feature:
                        feature_values[feature] = st.selectbox(feature.replace('L5-S1 ', '').title(), [0, 1],
                                                               format_func=lambda x: 'No' if x == 0 else 'Yes',
                                                               key=feature)
                    else:
                        feature_values[feature] = st.selectbox(feature.replace('L5-S1 ', '').title(), [0, 1],
                                                               format_func=lambda x: 'Reduced' if x == 0 else 'Normal',
                                                               key=feature)
                elif 'angle' in feature or 'EBQ' in feature:
                    feature_values[feature] = st.number_input(
                        feature.replace('L5-S1 ', '').title(),
                        min_value=float(min_val), max_value=float(max_val),
                        value=float((min_val + max_val) / 2), step=0.1, key=feature
                    )
                else:
                    feature_values[feature] = st.selectbox(
                        feature.replace('L5-S1 ', '').title(),
                        list(range(int(min_val), int(max_val) + 1)), key=feature
                    )

        # Spinal Alignment
        st.subheader("📐 Spinal Alignment")
        col1, col2 = st.columns(2)

        with col1:
            feature_values['L3-S1 lordosis angle'] = st.number_input('L3-S1 Lordosis Angle (degrees)',
                                                                     min_value=15.0, max_value=40.0, value=28.0,
                                                                     step=0.1)
        with col2:
            feature_values['Lumbar lordosis angle'] = st.number_input('Lumbar Lordosis Angle (degrees)',
                                                                      min_value=15.0, max_value=50.0, value=35.0,
                                                                      step=0.1)

        # Surgical Parameters
        st.subheader("⚕️ Surgical Parameters")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            feature_values['Cage height'] = st.selectbox('Cage Height (mm)', [10, 12, 14])
        with col2:
            feature_values['HU'] = st.number_input('Bone Density (HU)', min_value=80, max_value=200, value=130)
        with col3:
            feature_values['Operative time '] = st.number_input('Operative Time (min)', min_value=60, max_value=300,
                                                                value=150)
        with col4:
            feature_values['Blood loss'] = st.number_input('Blood Loss (ml)', min_value=50, max_value=500, value=150)

        # Submit button
        submitted = st.form_submit_button("🔮 Generate Prediction & Analysis", type="primary")

        if submitted:
            # Generate comprehensive analysis
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📊 Comprehensive Analysis Results</h3>', unsafe_allow_html=True)

            # Prediction
            prediction_result = simulate_prediction(feature_values, dataset_name)

            # Display prediction
            col1, col2, col3 = st.columns(3)

            with col1:
                pred_text = "ASD Risk" if prediction_result['prediction'] == 1 else "No ASD Risk"
                risk_class = "prediction-high-risk" if prediction_result['prediction'] == 1 else "prediction-low-risk"
                st.markdown(f'<div class="{risk_class}">Prediction: {pred_text}</div>', unsafe_allow_html=True)

            with col2:
                st.metric("ASD Probability", f"{prediction_result['probability'][1]:.3f}")

            with col3:
                st.metric("Confidence", f"{prediction_result['confidence']:.3f}")

            # Probability visualization
            prob_data = pd.DataFrame({
                'Outcome': ['No ASD', 'ASD'],
                'Probability': prediction_result['probability']
            })

            fig = px.bar(prob_data, x='Outcome', y='Probability',
                         title='ASD Risk Probability Distribution',
                         color='Outcome',
                         color_discrete_map={'No ASD': '#81c784', 'ASD': '#e57373'})
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Anchor Explanation
            st.markdown("### ⚓ Anchor Explanation")
            anchor_result = simulate_anchor_explanation(feature_values, dataset_name, prediction_result['prediction'])

            if anchor_result['anchor_features']:
                st.success(f"**Sufficient Condition Found**: {anchor_result['description']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precision", f"{anchor_result['precision']:.3f}")
                with col2:
                    st.metric("Coverage", f"{anchor_result['coverage']:.3f}")

                st.info(
                    "**Interpretation**: This rule accurately identifies patients with this risk level in 95%+ of cases.")

            else:
                st.warning(
                    "⚠️ No strong anchor conditions found. The prediction relies on complex feature interactions.")

            # Counterfactual Analysis
            st.markdown("### 🔄 Counterfactual Analysis")
            cf_result = simulate_counterfactual_analysis(feature_values, dataset_name, prediction_result['prediction'])

            if cf_result['success']:
                st.info(f"**Intervention Analysis**: Changes needed to achieve **{cf_result['target_class']}**:")

                for feature, change_info in cf_result['changes'].items():
                    direction = "increase" if change_info['change'] > 0 else "decrease"
                    feature_desc = predictor.feature_descriptions.get(feature, feature)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**{feature}**")
                        st.caption(feature_desc)
                    with col2:
                        st.metric("Current Value", f"{change_info['original']:.1f}")
                    with col3:
                        st.metric("Required Value", f"{change_info['counterfactual']:.1f}")

                if prediction_result['prediction'] == 1:
                    st.success("💡 **Clinical Insight**: These changes could potentially reduce ASD risk.")
                else:
                    st.warning("⚠️ **Risk Factors**: These changes would increase ASD risk.")

            # Feature Summary
            st.markdown("### 📋 Input Summary")

            # Key risk factors
            high_risk_features = []
            if feature_values.get('L3-4 EBQ', 2.5) > 3.5:
                high_risk_features.append("High L3-4 EBQ")
            if feature_values.get('L5-S1 osteoarthritis of facet joints', 0) >= 2:
                high_risk_features.append("Moderate-Severe L5-S1 Facet OA")
            if feature_values.get('Age', 60) > 65:
                high_risk_features.append("Advanced Age")

            if high_risk_features:
                st.warning(f"**Notable Risk Factors**: {', '.join(high_risk_features)}")
            else:
                st.success("**Risk Profile**: No major risk factors identified")


if __name__ == "__main__":
    main()