"""
Base builder for BentoML services
Common utilities and base classes
"""

import os
from pathlib import Path
from typing import Any, Dict
import logging

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Lazy import for BentoML
try:
    import bentoml
    BENTOML_AVAILABLE = True
except Exception as e:
    bentoml = None
    BENTOML_AVAILABLE = False

# Optional imports for ML frameworks
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    tf = None
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
except Exception as e:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False
except Exception as e:
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Import joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

