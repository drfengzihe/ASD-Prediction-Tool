#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for ASD Prediction Tool
"""

import os

# Application settings
APP_TITLE = "ASD Risk Prediction Tool"
APP_VERSION = "1.0.0"
INSTITUTION = "Beijing Chaoyang Hospital & Capital Medical University"

# Model settings
MODEL_VERSION = "2025.1"
SUPPORTED_DATASETS = ["L34 Result", "L5S1 Result"]

# Feature configurations
INTERVENTIONAL_FEATURES = ['Cage height', 'Operative time ', 'Blood loss']

FEATURE_CONSTRAINTS = {
    'Cage height': (8.0, 16.0),
    'Operative time ': (0.0, 600.0),
    'Blood loss': (0.0, 1000.0)
}

# Display settings
MAX_DEMO_SAMPLES = 3
DEFAULT_ANCHOR_THRESHOLD = 0.95
DEFAULT_CF_THRESHOLD = 0.95

# Research information
RESEARCH_PAPER = {
    "title": "From Prediction to Actionable Intervention: An Explainable AI Framework for Preventing Adjacent Segment Disease Using Anchors and Counterfactuals",
    "journal": "Clinical Orthopaedics and Related Research",
    "year": "2025",
    "authors": "Feng Z, Zhao M, Zhang Y, et al."
}

# Contact information
CONTACT_INFO = {
    "institution": "Beijing Chaoyang Hospital, Capital Medical University",
    "department": "Department of Orthopedic Surgery",
    "address": "Gongti South Rd, No. 8, Beijing 100020, China"
}