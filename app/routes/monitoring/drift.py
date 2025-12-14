"""
Drift detection routes
Provides endpoints for feature, data, and prediction drift detection
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.monitoring import ModelDriftDetection
from app.services.monitoring_service import monitoring_service
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/models/{model_id}/drift/feature", response_model=ModelDriftDetection)
async def detect_feature_drift(
    model_id: str,
    baseline_window_start: datetime = Query(..., description="Baseline period start time"),
    baseline_window_end: datetime = Query(..., description="Baseline period end time"),
    current_window_start: datetime = Query(..., description="Current period start time"),
    current_window_end: datetime = Query(..., description="Current period end time"),
    deployment_id: Optional[str] = Query(None, description="Optional deployment ID"),
    drift_threshold: float = Query(0.2, ge=0.0, le=1.0, description="PSI drift threshold"),
    ks_p_value_threshold: float = Query(0.05, ge=0.0, le=1.0, description="KS test p-value threshold")
):
    """
    Detect feature drift for a model
    
    Compares feature distributions between baseline and current time windows using:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov (KS) test
    
    Returns drift detection results including per-feature scores and severity.
    """
    try:
        drift_result = await monitoring_service.detect_feature_drift(
            model_id=model_id,
            baseline_window_start=baseline_window_start,
            baseline_window_end=baseline_window_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
            deployment_id=deployment_id,
            drift_threshold=drift_threshold,
            ks_p_value_threshold=ks_p_value_threshold
        )
        
        await monitoring_service.store_drift_detection(drift_result)
        return drift_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting feature drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/drift/data", response_model=ModelDriftDetection)
async def detect_data_drift(
    model_id: str,
    baseline_window_start: datetime = Query(..., description="Baseline period start time"),
    baseline_window_end: datetime = Query(..., description="Baseline period end time"),
    current_window_start: datetime = Query(..., description="Current period start time"),
    current_window_end: datetime = Query(..., description="Current period end time"),
    deployment_id: Optional[str] = Query(None, description="Optional deployment ID")
):
    """
    Detect data drift for a model
    
    Detects changes in data quality, schema, and distribution including:
    - Schema changes (missing/extra features)
    - Data quality issues (nulls, outliers)
    - Distribution shifts
    
    Returns comprehensive data drift analysis.
    """
    try:
        drift_result = await monitoring_service.detect_data_drift(
            model_id=model_id,
            baseline_window_start=baseline_window_start,
            baseline_window_end=baseline_window_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
            deployment_id=deployment_id
        )
        
        await monitoring_service.store_drift_detection(drift_result)
        return drift_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting data drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/drift/prediction", response_model=ModelDriftDetection)
async def detect_prediction_drift(
    model_id: str,
    baseline_window_start: datetime = Query(..., description="Baseline period start time"),
    baseline_window_end: datetime = Query(..., description="Baseline period end time"),
    current_window_start: datetime = Query(..., description="Current period start time"),
    current_window_end: datetime = Query(..., description="Current period end time"),
    deployment_id: Optional[str] = Query(None, description="Optional deployment ID"),
    drift_threshold: float = Query(0.2, ge=0.0, le=1.0, description="Drift threshold")
):
    """
    Detect prediction drift for a model
    
    Compares prediction distributions between baseline and current time windows:
    - Mean shift in predictions
    - Variance shift in predictions
    - Overall distribution shift
    
    Returns prediction drift analysis with statistical metrics.
    """
    try:
        drift_result = await monitoring_service.detect_prediction_drift(
            model_id=model_id,
            baseline_window_start=baseline_window_start,
            baseline_window_end=baseline_window_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
            deployment_id=deployment_id,
            drift_threshold=drift_threshold
        )
        
        await monitoring_service.store_drift_detection(drift_result)
        return drift_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting prediction drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/drift", response_model=List[ModelDriftDetection])
async def get_drift_history(
    model_id: str,
    deployment_id: Optional[str] = Query(None, description="Optional deployment ID"),
    drift_type: Optional[str] = Query(None, description="Filter by drift type: feature, data, or prediction"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results")
):
    """
    Get drift detection history for a model
    
    Returns a list of previous drift detection results, optionally filtered by:
    - Deployment ID
    - Drift type (feature, data, prediction)
    """
    try:
        async with get_session() as session:
            from app.models.monitoring import ModelDriftDetectionDB
            from sqlalchemy import desc
            from sqlmodel import select
            
            stmt = select(ModelDriftDetectionDB).where(
                ModelDriftDetectionDB.model_id == model_id
            )
            
            if deployment_id:
                stmt = stmt.where(ModelDriftDetectionDB.deployment_id == deployment_id)
            if drift_type:
                stmt = stmt.where(ModelDriftDetectionDB.drift_type == drift_type)
            
            stmt = stmt.order_by(desc(ModelDriftDetectionDB.timestamp)).limit(limit)
            result = await session.execute(stmt)
            drift_records = result.scalars().all()
            
            return [
                ModelDriftDetection(
                    id=drift.id,
                    model_id=drift.model_id,
                    deployment_id=drift.deployment_id,
                    drift_type=drift.drift_type,
                    detection_method=drift.detection_method,
                    baseline_window_start=drift.baseline_window_start,
                    baseline_window_end=drift.baseline_window_end,
                    current_window_start=drift.current_window_start,
                    current_window_end=drift.current_window_end,
                    drift_detected=drift.drift_detected,
                    drift_score=drift.drift_score,
                    drift_severity=drift.drift_severity,
                    p_value=drift.p_value,
                    feature_drift_scores=drift.feature_drift_scores,
                    feature_drift_details=drift.feature_drift_details,
                    prediction_mean_shift=drift.prediction_mean_shift,
                    prediction_variance_shift=drift.prediction_variance_shift,
                    prediction_distribution_shift=drift.prediction_distribution_shift,
                    data_quality_metrics=drift.data_quality_metrics,
                    schema_changes=drift.schema_changes,
                    drift_threshold=drift.drift_threshold,
                    alert_triggered=drift.alert_triggered,
                    alert_id=drift.alert_id,
                    timestamp=drift.timestamp,
                    additional_data=drift.additional_data or {}
                )
                for drift in drift_records
            ]
    except Exception as e:
        logger.error(f"Error getting drift history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

