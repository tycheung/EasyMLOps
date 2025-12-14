"""
Feature drift detection service
Handles detection of feature distribution drift using PSI and KS test
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


class FeatureDriftService(BaseMonitoringService):
    """Service for feature drift detection"""
    
    def __init__(self, drift_storage_service=None):
        """Initialize with optional storage service"""
        self.drift_storage = drift_storage_service
    
    async def detect_feature_drift(
        self,
        model_id: str,
        baseline_window_start: datetime,
        baseline_window_end: datetime,
        current_window_start: datetime,
        current_window_end: datetime,
        deployment_id: Optional[str] = None,
        drift_threshold: float = 0.2,
        ks_p_value_threshold: float = 0.05
    ) -> ModelDriftDetection:
        """Detect feature drift using KS test and PSI"""
        try:
            async with get_session() as session:
                # Get baseline prediction logs
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
                
                # Get current prediction logs
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
                        drift_type=DriftType.FEATURE,
                        detection_method="ks_test_psi",
                        baseline_window_start=baseline_window_start,
                        baseline_window_end=baseline_window_end,
                        current_window_start=current_window_start,
                        current_window_end=current_window_end,
                        drift_detected=False,
                        drift_threshold=drift_threshold
                    )
                
                # Extract features from input_data
                feature_drift_scores = {}
                feature_drift_details = {}
                max_drift_score = 0.0
                drift_detected = False
                
                # Get all unique feature names from both datasets
                all_features = set()
                for log in baseline_logs + current_logs:
                    if isinstance(log.input_data, dict):
                        all_features.update(log.input_data.keys())
                
                # Analyze each feature
                for feature_name in all_features:
                    baseline_values = []
                    current_values = []
                    
                    # Extract feature values from baseline
                    for log in baseline_logs:
                        if isinstance(log.input_data, dict) and feature_name in log.input_data:
                            value = log.input_data[feature_name]
                            if isinstance(value, (int, float)):
                                baseline_values.append(float(value))
                    
                    # Extract feature values from current
                    for log in current_logs:
                        if isinstance(log.input_data, dict) and feature_name in log.input_data:
                            value = log.input_data[feature_name]
                            if isinstance(value, (int, float)):
                                current_values.append(float(value))
                    
                    if len(baseline_values) < 10 or len(current_values) < 10:
                        continue
                    
                    baseline_arr = np.array(baseline_values)
                    current_arr = np.array(current_values)
                    
                    # Calculate PSI
                    psi_score = self._calculate_psi(baseline_arr, current_arr)
                    
                    # Perform KS test
                    ks_statistic, p_value = self._ks_test(baseline_arr, current_arr)
                    
                    # Store results
                    feature_drift_scores[feature_name] = psi_score
                    feature_drift_details[feature_name] = {
                        "psi_score": psi_score,
                        "ks_statistic": ks_statistic,
                        "p_value": p_value,
                        "baseline_mean": float(np.mean(baseline_arr)),
                        "current_mean": float(np.mean(current_arr)),
                        "baseline_std": float(np.std(baseline_arr)),
                        "current_std": float(np.std(current_arr)),
                        "baseline_count": len(baseline_values),
                        "current_count": len(current_values)
                    }
                    
                    # Check if drift detected for this feature
                    if psi_score > drift_threshold or p_value < ks_p_value_threshold:
                        drift_detected = True
                        max_drift_score = max(max_drift_score, psi_score)
                
                # Determine overall drift severity
                drift_severity = None
                if drift_detected:
                    if max_drift_score >= 0.5:
                        drift_severity = DriftSeverity.CRITICAL
                    elif max_drift_score >= 0.3:
                        drift_severity = DriftSeverity.HIGH
                    elif max_drift_score >= 0.2:
                        drift_severity = DriftSeverity.MEDIUM
                    else:
                        drift_severity = DriftSeverity.LOW
                
                # Get overall p-value
                overall_p_value = min(
                    [details.get("p_value", 1.0) for details in feature_drift_details.values()],
                    default=1.0
                )
                
                drift_result = ModelDriftDetection(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    drift_type=DriftType.FEATURE,
                    detection_method="ks_test_psi",
                    baseline_window_start=baseline_window_start,
                    baseline_window_end=baseline_window_end,
                    current_window_start=current_window_start,
                    current_window_end=current_window_end,
                    drift_detected=drift_detected,
                    drift_score=max_drift_score,
                    drift_severity=drift_severity,
                    p_value=overall_p_value,
                    feature_drift_scores=feature_drift_scores,
                    feature_drift_details=feature_drift_details,
                    drift_threshold=drift_threshold
                )
                
                # Store drift detection result if storage service available
                if self.drift_storage:
                    await self.drift_storage.store_drift_detection(drift_result)
                    if drift_detected:
                        await self.drift_storage.create_drift_alert(drift_result)
                
                return drift_result
                
        except Exception as e:
            logger.error(f"Error detecting feature drift: {e}", exc_info=True)
            raise

