# 🏥 ASD Risk Prediction Tool

An explainable AI framework for predicting Adjacent Segment Disease (ASD) risk following L4-5 lumbar fusion surgery.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Research](https://img.shields.io/badge/License-Research-yellow.svg)](LICENSE)

## 🎯 Live Demo

**🚀 [Access the Live Tool Here](YOUR_STREAMLIT_URL_HERE)**

*The tool is deployed and ready to use for research and educational purposes.*

## 📚 Research Publication

**Title**: *"From Prediction to Actionable Intervention: An Explainable AI Framework for Preventing Adjacent Segment Disease Using Anchors and Counterfactuals"*

**Journal**: Clinical Orthopaedics and Related Research  
**Authors**: Feng Z, Zhao M, Zhang Y, Li K, Guo S, Liu Y, Hai Y  
**Institution**: Beijing Chaoyang Hospital, Capital Medical University  

## 🏥 Clinical Background

Adjacent Segment Disease (ASD) affects **up to 43%** of patients following lumbar fusion, significantly reducing quality of life and often requiring revision surgery. Traditional prediction models face three critical limitations:

1. **Lack of transparency** - Black-box models provide no insight into decision-making
2. **Inability to provide individualized guidance** - Population-level statistics don't help specific patients  
3. **Failure to translate predictions into interventions** - Knowing risk doesn't provide actionable guidance

Our framework solves these problems by providing transparent, individualized, and actionable predictions.

## 🤖 Technical Innovation

### Core AI Framework

Our approach combines **three cutting-edge explainable AI techniques**:

#### 1. 🔍 Automated Piecewise Linear Regression (APLR)
- **Transparent prediction** with interpretable linear segments
- Captures non-linear relationships while maintaining explainability
- **Performance**: AUC 0.999 for L3-4 and 0.914 for L5-S1 ASD prediction

#### 2. ⚓ Anchor Explanations  
- Identifies **sufficient conditions** for high-risk classification
- Provides verifiable decision criteria (e.g., "L3-4 EBQ > 3.6")
- Establishes minimal feature sets that guarantee predictions

#### 3. 🔄 Counterfactual Analysis
- Quantifies **precise intervention thresholds**
- Shows what changes would alter risk classification  
- Enables proactive surgical planning

### Model Performance

| Adjacent Segment | AUC | Accuracy | Precision | Recall | 95% CI |
|------------------|-----|----------|-----------|---------|---------|
| **L3-4 ASD** | 0.999 | 98.8% | 95.2% | 95.2% | 0.998-1.000 |
| **L5-S1 ASD** | 0.914 | 88.6% | 85.0% | 61.1% | 0.844-0.983 |

## 🚀 Tool Features

### 📊 Interactive Demo Analysis
- **Global Model Insights**: Feature importance and model behavior visualization
- **Anonymized Sample Predictions**: Real APLR predictions on demonstration cases
- **Anchor Explanations**: Interactive examples of sufficient risk conditions
- **Counterfactual Analysis**: Explore intervention possibilities safely

### 🔮 Patient Risk Calculator
- **Comprehensive Input Interface**: 33 clinical and radiographic parameters
- **Real-time Predictions**: Instant ASD risk assessment for both adjacent segments
- **Complete Explanation Suite**: APLR reasoning + anchor conditions + counterfactual analysis
- **Clinical Guidance**: Actionable insights for surgical planning

### 🎓 Educational Interface
- **Methodology Explanation**: Detailed description of our AI framework
- **Clinical Interpretation**: How to understand and use the results
- **Research Context**: Background on ASD prediction challenges

## 💻 Technical Specifications

### Built With
- **Frontend**: Streamlit (Python web framework)
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **AI Framework**: Custom APLR implementation + Alibi (Anchor/Counterfactual)
- **Deployment**: Streamlit Cloud

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 512MB RAM minimum
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)
- **Internet**: Required for cloud deployment access

## 🏃‍♂️ Quick Start

### Option 1: Use the Live Tool (Recommended)
Simply visit our deployed application: **[Live Tool URL](YOUR_STREAMLIT_URL_HERE)**

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ASD-Prediction-Tool.git
cd ASD-Prediction-Tool

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_cloud_app.py