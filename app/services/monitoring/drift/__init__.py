"""
Drift detection services package
Exports DriftDetectionService and sub-services
"""

from app.services.monitoring.drift.drift_service import DriftDetectionService

__all__ = [
    "DriftDetectionService",
]

