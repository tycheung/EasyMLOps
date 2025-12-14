"""
Framework detection package
Exports FrameworkDetector and framework availability flags
"""

from app.utils.model_utils.frameworks.detector import FrameworkDetector

# Import framework availability flags from detector
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
except Exception:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
except Exception:
    LIGHTGBM_AVAILABLE = False

__all__ = [
    "FrameworkDetector",
    "TENSORFLOW_AVAILABLE",
    "PYTORCH_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
]

