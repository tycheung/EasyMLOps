"""
BentoML service management for dynamic model serving
Handles creation, deployment, and lifecycle management of ML model services

This module has been refactored into submodules:
- app.services.bentoml.builders.service_builder: Main ServiceBuilder facade
- app.services.bentoml.builders.sklearn_builder: Sklearn service builder
- app.services.bentoml.builders.tensorflow_builder: TensorFlow service builder
- app.services.bentoml.builders.pytorch_builder: PyTorch service builder
- app.services.bentoml.builders.xgboost_builder: XGBoost service builder
- app.services.bentoml.builders.lightgbm_builder: LightGBM service builder

This file maintains backward compatibility by re-exporting ServiceBuilder.
"""

# Re-export for backward compatibility
from app.services.bentoml.builders.service_builder import ServiceBuilder

__all__ = [
    "ServiceBuilder",
]
