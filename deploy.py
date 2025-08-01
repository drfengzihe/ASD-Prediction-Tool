#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deployment preparation script
"""

import os
import json
import zipfile
from datetime import datetime


def create_deployment_package():
    """Create deployment package for cloud platforms"""

    deployment_files = [
        'streamlit_cloud_app.py',
        'requirements.txt',
        'config.py',
        'utils/predictor_utils.py',
        '.streamlit/config.toml',
        'README.md'
    ]

    # Create deployment directory
    deploy_dir = "deployment_package"
    os.makedirs(deploy_dir, exist_ok=True)

    # Copy files to deployment directory
    for file_path in deployment_files:
        if os.path.exists(file_path):
            # Create subdirectories if needed
            target_path = os.path.join(deploy_dir, file_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Copy file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)

    # Create deployment info
    deployment_info = {
        "app_name": "ASD Risk Prediction Tool",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "main_file": "streamlit_cloud_app.py",
        "python_version": "3.9+",
        "description": "Explainable AI tool for Adjacent Segment Disease prediction"
    }

    with open(os.path.join(deploy_dir, 'deployment_info.json'), 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print(f"Deployment package created in: {deploy_dir}")
    print("Ready for cloud deployment!")


if __name__ == "__main__":
    create_deployment_package()