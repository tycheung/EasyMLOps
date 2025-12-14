"""
Performance degradation service module
Exports the main PerformanceDegradationService
"""

from app.services.monitoring.degradation.degradation_service import PerformanceDegradationService

__all__ = [
    "PerformanceDegradationService",
]

