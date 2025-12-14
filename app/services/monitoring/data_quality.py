"""
Data quality service
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from sqlalchemy import select, and_

from app.database import get_session
from app.models.monitoring import (
    OutlierDetectionDB, AnomalyDetectionDB, DataQualityMetricsDB
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class DataQualityService(BaseMonitoringService):
    """Service for data quality"""
    
    async def detect_outliers(
        self,
        model_id: str,
        data: List[Dict[str, Any]],
        method: str = "isolation_forest",
        deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect outliers in input data"""
        try:
            if not data:
                return {
                    "model_id": model_id,
                    "deployment_id": deployment_id,
                    "outlier_count": 0,
                    "outlier_indices": [],
                    "outlier_scores": {},
                    "detection_method": method
                }
            
            outlier_indices = []
            outlier_scores = {}
            
            # Extract numerical features
            if isinstance(data[0], dict):
                features = list(data[0].keys())
                feature_values = {feat: [d.get(feat) for d in data if isinstance(d.get(feat), (int, float))] for feat in features}
            else:
                # If data is a list of lists/arrays
                feature_values = {"feature_0": [float(d) if isinstance(d, (int, float)) else 0.0 for d in data]}
            
            # Detect outliers using z-score method (simplified)
            for feat, values in feature_values.items():
                if len(values) < 3:
                    continue
                
                values_array = np.array(values)
                z_scores = np.abs(stats.zscore(values_array))
                outliers = np.where(z_scores > 3)[0]
                
                for idx in outliers:
                    if idx not in outlier_indices:
                        outlier_indices.append(int(idx))
                    outlier_scores[f"{feat}_{idx}"] = float(z_scores[idx])
            
            result = {
                "model_id": model_id,
                "deployment_id": deployment_id,
                "outlier_count": len(outlier_indices),
                "outlier_indices": outlier_indices,
                "outlier_scores": outlier_scores,
                "detection_method": method,
                "input_data": data[:5] if len(data) > 5 else data,  # Sample
                "is_outlier": len(outlier_indices) > 0
            }
            
            # Store detection if outliers found
            if outlier_indices:
                detection_id = str(uuid.uuid4())
                detection_db = OutlierDetectionDB(
                    id=detection_id,
                    model_id=model_id,
                    deployment_id=deployment_id,
                    detection_method=method,
                    outlier_type="input",
                    input_data=data[:1] if data else None,
                    outlier_score=float(np.mean(list(outlier_scores.values()))) if outlier_scores else 0.0,
                    z_score=float(np.max([v for v in outlier_scores.values()])) if outlier_scores else None,
                    feature_outlier_scores=outlier_scores,
                    outlier_features=list(set([k.split("_")[0] for k in outlier_scores.keys()])),
                    is_outlier=True,
                    severity="medium" if len(outlier_indices) < len(data) * 0.1 else "high"
                )
                
                async with get_session() as session:
                    session.add(detection_db)
                    await session.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}", exc_info=True)
            raise
    
    async def calculate_data_quality_metrics(
        self,
        model_id: str,
        data: List[Dict[str, Any]],
        deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        try:
            if not data:
                return {
                    "model_id": model_id,
                    "deployment_id": deployment_id,
                    "completeness": 0.0,
                    "validity": 0.0,
                    "consistency": 0.0,
                    "timeliness": 1.0,
                    "total_samples": 0
                }
            
            total_samples = len(data)
            total_fields = 0
            missing_fields = 0
            valid_samples = 0
            
            # Calculate completeness and validity
            for sample in data:
                if isinstance(sample, dict):
                    fields = list(sample.keys())
                    total_fields += len(fields)
                    for field, value in sample.items():
                        if value is None:
                            missing_fields += 1
                    if all(v is not None for v in sample.values()):
                        valid_samples += 1
                else:
                    total_fields += 1
                    if sample is not None:
                        valid_samples += 1
            
            completeness = 1.0 - (missing_fields / total_fields) if total_fields > 0 else 1.0
            validity = valid_samples / total_samples if total_samples > 0 else 0.0
            
            # Consistency is simplified - would check for consistent types/patterns
            consistency = 0.95  # Placeholder
            
            overall_quality = (completeness + validity + consistency) / 3.0
            
            metrics = {
                "model_id": model_id,
                "deployment_id": deployment_id,
                "time_window_start": datetime.utcnow().isoformat(),
                "time_window_end": datetime.utcnow().isoformat(),
                "completeness_score": completeness,
                "validity_score": validity,
                "consistency_score": consistency,
                "overall_quality_score": overall_quality,
                "total_samples": total_samples,
                "valid_samples": valid_samples,
                "invalid_samples": total_samples - valid_samples,
                "missing_value_count": missing_fields,
                "schema_violations": 0,
                "range_violations": 0,
                "type_violations": 0,
                "feature_quality_metrics": {},
                "quality_trend": "stable"
            }
            
            # Store metrics
            metrics_id = str(uuid.uuid4())
            metrics_db = DataQualityMetricsDB(
                id=metrics_id,
                model_id=model_id,
                deployment_id=deployment_id,
                time_window_start=datetime.utcnow(),
                time_window_end=datetime.utcnow(),
                completeness_score=completeness,
                validity_score=validity,
                consistency_score=consistency,
                overall_quality_score=overall_quality,
                total_samples=total_samples,
                valid_samples=valid_samples,
                invalid_samples=total_samples - valid_samples,
                missing_value_count=missing_fields
            )
            
            async with get_session() as session:
                session.add(metrics_db)
                await session.commit()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {e}", exc_info=True)
            raise
    
    async def detect_anomaly(
        self,
        model_id: str,
        data: Dict[str, Any],
        deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect anomalies in input data"""
        try:
            # Simplified anomaly detection - would use more sophisticated methods
            anomaly_score = 0.0
            is_anomaly = False
            
            # Check for null values
            if any(v is None for v in data.values()):
                anomaly_score += 0.3
            
            # Check for extreme values (simplified)
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if abs(value) > 1000000:  # Arbitrary threshold
                        anomaly_score += 0.2
            
            is_anomaly = anomaly_score > 0.5
            
            result = {
                "model_id": model_id,
                "deployment_id": deployment_id,
                "anomaly_type": "input",
                "detection_method": "statistical",
                "input_data": data,
                "anomaly_score": anomaly_score,
                "confidence": 0.8 if is_anomaly else 0.2,
                "anomaly_features": [k for k, v in data.items() if v is None or (isinstance(v, (int, float)) and abs(v) > 1000000)],
                "anomaly_reason": "Null values or extreme values detected" if is_anomaly else None,
                "is_anomaly": is_anomaly,
                "severity": "high" if anomaly_score > 0.7 else "medium" if anomaly_score > 0.5 else "low",
                "alert_triggered": is_anomaly and anomaly_score > 0.7
            }
            
            # Store detection if anomaly found
            if is_anomaly:
                detection_id = str(uuid.uuid4())
                detection_db = AnomalyDetectionDB(
                    id=detection_id,
                    model_id=model_id,
                    deployment_id=deployment_id,
                    anomaly_type="input",
                    detection_method="statistical",
                    input_data=data,
                    anomaly_score=anomaly_score,
                    confidence=result["confidence"],
                    anomaly_features=result["anomaly_features"],
                    anomaly_reason=result["anomaly_reason"],
                    is_anomaly=True,
                    severity=result["severity"],
                    alert_triggered=result["alert_triggered"]
                )
                
                async with get_session() as session:
                    session.add(detection_db)
                    await session.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}", exc_info=True)
            raise
