"""
Performance monitoring service
Handles model performance metrics, prediction logging, and confidence tracking

This module has been refactored into submodules:
- app.services.monitoring.performance.performance_service: Main facade
- app.services.monitoring.performance.prediction_logging: Prediction logging
- app.services.monitoring.performance.metrics_calculation: Metrics calculation
- app.services.monitoring.performance.metrics_storage: Metrics storage
- app.services.monitoring.performance.aggregation: Metrics aggregation

This file maintains backward compatibility by re-exporting PerformanceMonitoringService.
"""

# Re-export for backward compatibility
from app.services.monitoring.performance.performance_service import PerformanceMonitoringService

__all__ = [
    "PerformanceMonitoringService",
]
