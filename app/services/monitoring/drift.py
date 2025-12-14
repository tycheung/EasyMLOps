"""
Drift detection service
Handles feature drift, prediction drift, and data drift detection

This module has been refactored into submodules:
- app.services.monitoring.drift.drift_service: Main facade
- app.services.monitoring.drift.feature_drift: Feature drift detection
- app.services.monitoring.drift.data_drift: Data drift detection
- app.services.monitoring.drift.prediction_drift: Prediction drift detection
- app.services.monitoring.drift.drift_storage: Drift storage and alerts

This file maintains backward compatibility by re-exporting DriftDetectionService.
"""

# Re-export for backward compatibility
from app.services.monitoring.drift.drift_service import DriftDetectionService

__all__ = [
    "DriftDetectionService",
]
