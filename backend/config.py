"""
Configuration file for the eXCNN project.
Contains model settings, paths, and hyperparameters.
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
EXAMPLES_DIR = ASSETS_DIR / "examples_imagenet"
DOCS_DIR = ASSETS_DIR / "docs"
PRECOMPUTED_DIR = ASSETS_DIR / "precomputed"

# Model Configuration
MODEL_NAME = "resnet50"  # Options: resnet50, resnet152, convnext_large
DEVICE = "cpu"  # Will be set to cuda if available
NUM_CLASSES = 1000  # ImageNet classes

# Image Processing
IMAGE_SIZE = 224  # Standard ImageNet size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# XAI Method Settings
# Grad-CAM
GRADCAM_TARGET_LAYER = None  # Will be set based on model

# Occlusion Sensitivity
OCCLUSION_WINDOW_SIZE = 15  # Size of the occlusion patch
OCCLUSION_STRIDE = 8  # Stride for sliding window
OCCLUSION_BASELINE = "mean"  # Options: mean, zero, blur

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
UPLOAD_DIR = PROJECT_ROOT / "uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
