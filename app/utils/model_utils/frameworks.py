"""
Framework-specific detection, analysis, and metadata extraction
Supports TensorFlow, PyTorch, Scikit-learn, XGBoost, LightGBM, and ONNX

This module has been refactored into submodules:
- app.utils.model_utils.frameworks.detector: Core detection logic
- app.utils.model_utils.frameworks.sklearn_detector: Sklearn detection and metadata
- app.utils.model_utils.frameworks.xgboost_detector: XGBoost detection and metadata
- app.utils.model_utils.frameworks.lightgbm_detector: LightGBM detection and metadata
- app.utils.model_utils.frameworks.tensorflow_detector: TensorFlow detection and metadata
- app.utils.model_utils.frameworks.pytorch_detector: PyTorch detection and metadata

This file maintains backward compatibility by re-exporting all classes and constants.
"""

# Re-export for backward compatibility
from app.utils.model_utils.frameworks.detector import FrameworkDetector
from app.utils.model_utils.frameworks import (
    TENSORFLOW_AVAILABLE,
    PYTORCH_AVAILABLE,
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
)

__all__ = [
    "FrameworkDetector",
    "TENSORFLOW_AVAILABLE",
    "PYTORCH_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
]
