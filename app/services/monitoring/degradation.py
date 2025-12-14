"""
Performance degradation service
Handles performance degradation detection and alerting

This module has been refactored into submodules:
- app.services.monitoring.degradation.degradation_service: Main service facade
- app.services.monitoring.degradation.prediction_logging: Logging predictions with ground truth
- app.services.monitoring.degradation.classification_metrics: Classification metrics calculation
- app.services.monitoring.degradation.regression_metrics: Regression metrics calculation
- app.services.monitoring.degradation.degradation_detection: Degradation detection logic
- app.services.monitoring.degradation.baseline_helpers: Baseline performance helpers
- app.services.monitoring.degradation.history_storage: Performance history storage
- app.services.monitoring.degradation.alert_creation: Alert creation

This file maintains backward compatibility by re-exporting PerformanceDegradationService.
"""

# Re-export for backward compatibility
from app.services.monitoring.degradation.degradation_service import PerformanceDegradationService

__all__ = [
    "PerformanceDegradationService",
]
