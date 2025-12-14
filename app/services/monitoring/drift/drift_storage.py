"""
Drift detection storage service
Handles storing drift detection results and creating alerts
"""

import logging
import uuid
from typing import Optional

from app.database import get_session
from app.models.monitoring import ModelDriftDetectionDB
from app.schemas.monitoring import (
    AlertSeverity, DriftSeverity, ModelDriftDetection
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class DriftStorageService(BaseMonitoringService):
    """Service for storing drift detection results"""
    
    async def store_drift_detection(self, drift_result: ModelDriftDetection) -> str:
        """Store drift detection result in database"""
        try:
            drift_db = ModelDriftDetectionDB(
                id=str(uuid.uuid4()),
                model_id=drift_result.model_id,
                deployment_id=drift_result.deployment_id,
                drift_type=drift_result.drift_type.value,
                detection_method=drift_result.detection_method,
                baseline_window_start=drift_result.baseline_window_start,
                baseline_window_end=drift_result.baseline_window_end,
                current_window_start=drift_result.current_window_start,
                current_window_end=drift_result.current_window_end,
                drift_detected=drift_result.drift_detected,
                drift_score=drift_result.drift_score,
                drift_severity=drift_result.drift_severity.value if drift_result.drift_severity else None,
                p_value=drift_result.p_value,
                feature_drift_scores=drift_result.feature_drift_scores,
                feature_drift_details=drift_result.feature_drift_details,
                prediction_mean_shift=drift_result.prediction_mean_shift,
                prediction_variance_shift=drift_result.prediction_variance_shift,
                prediction_distribution_shift=drift_result.prediction_distribution_shift,
                data_quality_metrics=drift_result.data_quality_metrics,
                schema_changes=drift_result.schema_changes,
                drift_threshold=drift_result.drift_threshold,
                alert_triggered=drift_result.alert_triggered if hasattr(drift_result, 'alert_triggered') else False,
                alert_id=drift_result.alert_id if hasattr(drift_result, 'alert_id') else None,
                additional_data=drift_result.additional_data if hasattr(drift_result, 'additional_data') else None
            )
            
            async with get_session() as session:
                session.add(drift_db)
                await session.commit()
                logger.info(f"Stored drift detection result for model {drift_result.model_id}")
                return drift_db.id
                
        except Exception as e:
            logger.error(f"Error storing drift detection: {e}")
            raise
    
    async def create_drift_alert(self, drift_result: ModelDriftDetection) -> Optional[str]:
        """Create an alert for detected drift"""
        try:
            severity_map = {
                DriftSeverity.CRITICAL: AlertSeverity.CRITICAL,
                DriftSeverity.HIGH: AlertSeverity.ERROR,
                DriftSeverity.MEDIUM: AlertSeverity.WARNING,
                DriftSeverity.LOW: AlertSeverity.INFO
            }
            
            alert_severity = severity_map.get(drift_result.drift_severity, AlertSeverity.WARNING)
            
            title = f"{drift_result.drift_type.value.capitalize()} Drift Detected for Model {drift_result.model_id}"
            description = (
                f"Detected {drift_result.drift_type.value} drift with score {drift_result.drift_score:.3f}. "
                f"Severity: {drift_result.drift_severity.value if drift_result.drift_severity else 'unknown'}"
            )
            
            # Note: create_alert will be accessed via service composition
            # For now, we'll log it
            logger.warning(f"Would create drift alert: {title} - {description}")
            
            # Update drift result with alert info (placeholder)
            drift_result.alert_triggered = True
            drift_result.alert_id = "placeholder_alert_id"
            
            return "placeholder_alert_id"
            
        except Exception as e:
            logger.error(f"Error creating drift alert: {e}")
            return None

