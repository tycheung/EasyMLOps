"""
Performance degradation alert creation
Handles creating alerts for performance degradation
"""

import logging
from typing import Optional

from app.schemas.monitoring import AlertSeverity, ModelPerformanceHistory

logger = logging.getLogger(__name__)


async def create_performance_degradation_alert(perf_history: ModelPerformanceHistory) -> Optional[str]:
    """Create an alert for performance degradation"""
    try:
        severity_map = {
            "critical": AlertSeverity.CRITICAL,
            "high": AlertSeverity.ERROR,
            "medium": AlertSeverity.WARNING,
            "low": AlertSeverity.INFO
        }
        
        alert_severity = severity_map.get(perf_history.degradation_severity or "medium", AlertSeverity.WARNING)
        
        if perf_history.model_type == "classification":
            metric_value = perf_history.accuracy_delta if perf_history.accuracy_delta else perf_history.f1_delta
            metric_name = "accuracy" if perf_history.accuracy_delta else "F1 score"
        else:
            metric_value = perf_history.mae_delta if perf_history.mae_delta else perf_history.r2_delta
            metric_name = "MAE" if perf_history.mae_delta else "R²"
        
        title = f"Performance Degradation Detected for Model {perf_history.model_id}"
        description = (
            f"Model {perf_history.model_type} performance has degraded. "
            f"{metric_name} changed by {metric_value:.3f}. "
            f"Severity: {perf_history.degradation_severity or 'unknown'}"
        )
        
        # Note: create_alert will be accessed via service composition
        logger.warning(f"Would create performance degradation alert: {title} - {description}")
        
        return "placeholder_alert_id"
        
    except Exception as e:
        logger.error(f"Error creating performance degradation alert: {e}")
        return None

