"""
Prediction drift detection service
Handles detection of prediction distribution drift
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import DriftSeverity, DriftType, ModelDriftDetection
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class PredictionDriftService(BaseMonitoringService):
    """Service for prediction drift detection"""
    
    def __init__(self, drift_storage_service=None):
        """Initialize with optional storage service"""
        self.drift_storage = drift_storage_service
    
    async def detect_prediction_drift(
        self,
        model_id: str,
        baseline_window_start: datetime,
        baseline_window_end: datetime,
        current_window_start: datetime,
        current_window_end: datetime,
        deployment_id: Optional[str] = None,
        drift_threshold: float = 0.2
    ) -> ModelDriftDetection:
        """Detect prediction drift by comparing prediction distributions"""
        try:
            async with get_session() as session:
                # Get baseline predictions
                baseline_stmt = select(PredictionLogDB).where(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= baseline_window_start,
                    PredictionLogDB.timestamp <= baseline_window_end,
                    PredictionLogDB.success == True
                )
                if deployment_id:
                    baseline_stmt = baseline_stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                baseline_result = await session.execute(baseline_stmt)
                baseline_logs = baseline_result.scalars().all()
                
                # Get current predictions
                current_stmt = select(PredictionLogDB).where(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= current_window_start,
                    PredictionLogDB.timestamp <= current_window_end,
                    PredictionLogDB.success == True
                )
                if deployment_id:
                    current_stmt = current_stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                current_result = await session.execute(current_stmt)
                current_logs = current_result.scalars().all()
                
                if not baseline_logs or not current_logs:
                    return ModelDriftDetection(
                        model_id=model_id,
                        deployment_id=deployment_id,
                        drift_type=DriftType.PREDICTION,
                        detection_method="statistical_comparison",
                        baseline_window_start=baseline_window_start,
                        baseline_window_end=baseline_window_end,
                        current_window_start=current_window_start,
                        current_window_end=current_window_end,
                        drift_detected=False,
                        drift_threshold=drift_threshold
                    )
                
                # Extract prediction values
                baseline_predictions = []
                current_predictions = []
                
                for log in baseline_logs:
                    pred_value = self._extract_prediction_value(log.output_data)
                    if pred_value is not None:
                        baseline_predictions.append(pred_value)
                
                for log in current_logs:
                    pred_value = self._extract_prediction_value(log.output_data)
                    if pred_value is not None:
                        current_predictions.append(pred_value)
                
                if len(baseline_predictions) < 10 or len(current_predictions) < 10:
                    return ModelDriftDetection(
                        model_id=model_id,
                        deployment_id=deployment_id,
                        drift_type=DriftType.PREDICTION,
                        detection_method="statistical_comparison",
                        baseline_window_start=baseline_window_start,
                        baseline_window_end=baseline_window_end,
                        current_window_start=current_window_start,
                        current_window_end=current_window_end,
                        drift_detected=False,
                        drift_threshold=drift_threshold
                    )
                
                baseline_arr = np.array(baseline_predictions)
                current_arr = np.array(current_predictions)
                
                # Calculate statistics
                baseline_mean = float(np.mean(baseline_arr))
                current_mean = float(np.mean(current_arr))
                baseline_std = float(np.std(baseline_arr))
                current_std = float(np.std(current_arr))
                
                # Calculate shifts
                mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6) if baseline_std > 0 else 0
                variance_shift = abs(current_std - baseline_std) / (baseline_std + 1e-6) if baseline_std > 0 else 0
                
                # Calculate distribution shift using PSI
                distribution_shift = self._calculate_psi(baseline_arr, current_arr)
                
                # Perform KS test
                ks_statistic, p_value = self._ks_test(baseline_arr, current_arr)
                
                # Determine if drift detected
                drift_detected = (
                    distribution_shift > drift_threshold or
                    p_value < 0.05 or
                    mean_shift > 2.0  # More than 2 standard deviations
                )
                
                # Determine severity
                drift_severity = None
                drift_score = distribution_shift
                if drift_detected:
                    if distribution_shift >= 0.5 or mean_shift >= 3.0:
                        drift_severity = DriftSeverity.CRITICAL
                    elif distribution_shift >= 0.3 or mean_shift >= 2.0:
                        drift_severity = DriftSeverity.HIGH
                    elif distribution_shift >= 0.2:
                        drift_severity = DriftSeverity.MEDIUM
                    else:
                        drift_severity = DriftSeverity.LOW
                
                drift_result = ModelDriftDetection(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    drift_type=DriftType.PREDICTION,
                    detection_method="statistical_comparison",
                    baseline_window_start=baseline_window_start,
                    baseline_window_end=baseline_window_end,
                    current_window_start=current_window_start,
                    current_window_end=current_window_end,
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                    drift_severity=drift_severity,
                    p_value=p_value,
                    prediction_mean_shift=mean_shift,
                    prediction_variance_shift=variance_shift,
                    prediction_distribution_shift=distribution_shift,
                    drift_threshold=drift_threshold
                )
                
                # Store drift detection result if storage service available
                if self.drift_storage:
                    await self.drift_storage.store_drift_detection(drift_result)
                    if drift_detected:
                        await self.drift_storage.create_drift_alert(drift_result)
                
                return drift_result
                
        except Exception as e:
            logger.error(f"Error detecting prediction drift: {e}", exc_info=True)
            raise

