"""
Model utilities for validation, file handling, and ML framework detection
Provides comprehensive support for multiple ML frameworks including TensorFlow and PyTorch

This module has been refactored into submodules:
- app.utils.model_utils.validator: Core validation methods
- app.utils.model_utils.frameworks: Framework-specific detection and analysis
- app.utils.model_utils.loaders: Model file management utilities

This file maintains backward compatibility by re-exporting all classes and constants.
"""

# Re-export all classes and utilities for backward compatibility
from app.utils.model_utils.validator import ModelValidator
from app.utils.model_utils.frameworks import (
    FrameworkDetector,
    TENSORFLOW_AVAILABLE,
    PYTORCH_AVAILABLE,
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
)
from app.utils.model_utils.loaders import ModelFileManager

# Re-export for backward compatibility
__all__ = [
    "ModelValidator",
    "FrameworkDetector",
    "ModelFileManager",
    "TENSORFLOW_AVAILABLE",
    "PYTORCH_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
]
