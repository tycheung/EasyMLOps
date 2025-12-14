"""
Data drift detection service
Handles detection of data drift including schema changes and data quality issues
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import DriftSeverity, DriftType, ModelDriftDetection
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class DataDriftService(BaseMonitoringService):
    """Service for data drift detection"""
    
    def __init__(self, drift_storage_service=None):
        """Initialize with optional storage service"""
        self.drift_storage = drift_storage_service
    
    async def detect_data_drift(
        self,
        model_id: str,
        baseline_window_start: datetime,
        baseline_window_end: datetime,
        current_window_start: datetime,
        current_window_end: datetime,
        deployment_id: Optional[str] = None
    ) -> ModelDriftDetection:
        """Detect data drift including schema changes and data quality issues"""
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
                        drift_type=DriftType.DATA,
                        detection_method="schema_quality_analysis",
                        baseline_window_start=baseline_window_start,
                        baseline_window_end=baseline_window_end,
                        current_window_start=current_window_start,
                        current_window_end=current_window_end,
                        drift_detected=False
                    )
                
                # Analyze schema changes
                baseline_schemas = {}
                current_schemas = {}
                
                for log in baseline_logs:
                    if isinstance(log.input_data, dict):
                        for key, value in log.input_data.items():
                            if key not in baseline_schemas:
                                baseline_schemas[key] = type(value).__name__
                
                for log in current_logs:
                    if isinstance(log.input_data, dict):
                        for key, value in log.input_data.items():
                            if key not in current_schemas:
                                current_schemas[key] = type(value).__name__
                
                schema_changes = {}
                drift_detected = False
                
                # Check for new fields
                new_fields = set(current_schemas.keys()) - set(baseline_schemas.keys())
                if new_fields:
                    schema_changes["new_fields"] = list(new_fields)
                    drift_detected = True
                
                # Check for removed fields
                removed_fields = set(baseline_schemas.keys()) - set(current_schemas.keys())
                if removed_fields:
                    schema_changes["removed_fields"] = list(removed_fields)
                    drift_detected = True
                
                # Check for type changes
                type_changes = {}
                for field in set(baseline_schemas.keys()) & set(current_schemas.keys()):
                    if baseline_schemas[field] != current_schemas[field]:
                        type_changes[field] = {
                            "baseline_type": baseline_schemas[field],
                            "current_type": current_schemas[field]
                        }
                        drift_detected = True
                if type_changes:
                    schema_changes["type_changes"] = type_changes
                
                # Analyze data quality metrics
                data_quality_metrics = {}
                
                # Missing/null values analysis
                baseline_missing = {}
                current_missing = {}
                
                for log in baseline_logs:
                    if isinstance(log.input_data, dict):
                        for key in baseline_schemas.keys():
                            if key not in log.input_data or log.input_data[key] is None:
                                baseline_missing[key] = baseline_missing.get(key, 0) + 1
                
                for log in current_logs:
                    if isinstance(log.input_data, dict):
                        for key in current_schemas.keys():
                            if key not in log.input_data or log.input_data[key] is None:
                                current_missing[key] = current_missing.get(key, 0) + 1
                
                missing_comparison = {}
                for key in set(baseline_missing.keys()) | set(current_missing.keys()):
                    baseline_pct = (baseline_missing.get(key, 0) / len(baseline_logs)) * 100 if baseline_logs else 0
                    current_pct = (current_missing.get(key, 0) / len(current_logs)) * 100 if current_logs else 0
                    missing_comparison[key] = {
                        "baseline_missing_pct": baseline_pct,
                        "current_missing_pct": current_pct,
                        "change": current_pct - baseline_pct
                    }
                    if abs(current_pct - baseline_pct) > 10:  # More than 10% change
                        drift_detected = True
                
                data_quality_metrics["missing_values"] = missing_comparison
                
                # Categorical value distribution changes
                categorical_changes = {}
                for field in set(baseline_schemas.keys()) & set(current_schemas.keys()):
                    baseline_values = []
                    current_values = []
                    
                    for log in baseline_logs:
                        if isinstance(log.input_data, dict) and field in log.input_data:
                            val = log.input_data[field]
                            if isinstance(val, str):  # Categorical
                                baseline_values.append(val)
                    
                    for log in current_logs:
                        if isinstance(log.input_data, dict) and field in log.input_data:
                            val = log.input_data[field]
                            if isinstance(val, str):  # Categorical
                                current_values.append(val)
                    
                    if len(baseline_values) > 10 and len(current_values) > 10:
                        baseline_unique = set(baseline_values)
                        current_unique = set(current_values)
                        
                        new_values = current_unique - baseline_unique
                        removed_values = baseline_unique - current_unique
                        
                        if new_values or removed_values:
                            categorical_changes[field] = {
                                "new_values": list(new_values),
                                "removed_values": list(removed_values),
                                "baseline_unique_count": len(baseline_unique),
                                "current_unique_count": len(current_unique)
                            }
                            drift_detected = True
                
                if categorical_changes:
                    data_quality_metrics["categorical_changes"] = categorical_changes
                
                # Numerical range drift
                numerical_range_changes = {}
                for field in set(baseline_schemas.keys()) & set(current_schemas.keys()):
                    baseline_nums = []
                    current_nums = []
                    
                    for log in baseline_logs:
                        if isinstance(log.input_data, dict) and field in log.input_data:
                            val = log.input_data[field]
                            if isinstance(val, (int, float)):
                                baseline_nums.append(float(val))
                    
                    for log in current_logs:
                        if isinstance(log.input_data, dict) and field in log.input_data:
                            val = log.input_data[field]
                            if isinstance(val, (int, float)):
                                current_nums.append(float(val))
                    
                    if len(baseline_nums) > 10 and len(current_nums) > 10:
                        baseline_min, baseline_max = min(baseline_nums), max(baseline_nums)
                        current_min, current_max = min(current_nums), max(current_nums)
                        
                        range_shift = abs(current_min - baseline_min) + abs(current_max - baseline_max)
                        range_shift_pct = (range_shift / (baseline_max - baseline_min + 1e-6)) * 100 if baseline_max > baseline_min else 0
                        
                        if range_shift_pct > 20:  # More than 20% range shift
                            numerical_range_changes[field] = {
                                "baseline_range": [baseline_min, baseline_max],
                                "current_range": [current_min, current_max],
                                "range_shift_pct": range_shift_pct
                            }
                            drift_detected = True
                
                if numerical_range_changes:
                    data_quality_metrics["numerical_range_changes"] = numerical_range_changes
                
                # Determine severity
                drift_severity = None
                drift_score = 0.0
                if drift_detected:
                    # Calculate overall drift score
                    score_components = []
                    if schema_changes:
                        score_components.append(0.5)  # Schema changes are significant
                    if data_quality_metrics.get("missing_values"):
                        max_missing_change = max(
                            [abs(m["change"]) for m in data_quality_metrics["missing_values"].values()],
                            default=0
                        )
                        score_components.append(min(max_missing_change / 50.0, 0.5))  # Normalize to 0-0.5
                    if data_quality_metrics.get("categorical_changes"):
                        score_components.append(0.3)
                    if data_quality_metrics.get("numerical_range_changes"):
                        score_components.append(0.2)
                    
                    drift_score = min(sum(score_components), 1.0)
                    
                    if drift_score >= 0.7:
                        drift_severity = DriftSeverity.CRITICAL
                    elif drift_score >= 0.5:
                        drift_severity = DriftSeverity.HIGH
                    elif drift_score >= 0.3:
                        drift_severity = DriftSeverity.MEDIUM
                    else:
                        drift_severity = DriftSeverity.LOW
                
                drift_result = ModelDriftDetection(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    drift_type=DriftType.DATA,
                    detection_method="schema_quality_analysis",
                    baseline_window_start=baseline_window_start,
                    baseline_window_end=baseline_window_end,
                    current_window_start=current_window_start,
                    current_window_end=current_window_end,
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                    drift_severity=drift_severity,
                    data_quality_metrics=data_quality_metrics if data_quality_metrics else None,
                    schema_changes=schema_changes if schema_changes else None
                )
                
                # Store drift detection result if storage service available
                if self.drift_storage:
                    await self.drift_storage.store_drift_detection(drift_result)
                    if drift_detected:
                        await self.drift_storage.create_drift_alert(drift_result)
                
                return drift_result
                
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}", exc_info=True)
            raise

