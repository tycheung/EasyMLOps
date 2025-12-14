"""
Data quality routes
Provides endpoints for outlier detection, anomaly detection, and data quality metrics
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
import logging

from app.schemas.monitoring import OutlierDetection, AnomalyDetection, DataQualityMetrics
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/models/{model_id}/data-quality/outliers", response_model=OutlierDetection)
async def detect_outliers(
    model_id: str,
    data: List[Dict[str, Any]] = Body(..., description="Input data"),
    method: str = Query("isolation_forest", description="Outlier detection method"),
    deployment_id: Optional[str] = Query(None)
):
    """Detect outliers in input data"""
    try:
        result = await monitoring_service.detect_outliers(
            model_id=model_id,
            data=data,
            method=method,
            deployment_id=deployment_id
        )
        return OutlierDetection(**result)
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/data-quality/metrics", response_model=DataQualityMetrics)
async def calculate_data_quality_metrics(
    model_id: str,
    data: List[Dict[str, Any]] = Body(..., description="Input data"),
    deployment_id: Optional[str] = Query(None)
):
    """Calculate data quality metrics"""
    try:
        metrics = await monitoring_service.calculate_data_quality_metrics(
            model_id=model_id,
            data=data,
            deployment_id=deployment_id
        )
        return DataQualityMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error calculating data quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/data-quality/anomaly", response_model=AnomalyDetection)
async def detect_anomaly(
    model_id: str,
    data: Dict[str, Any] = Body(..., description="Input data"),
    deployment_id: Optional[str] = Query(None)
):
    """Detect anomalies in input data"""
    try:
        result = await monitoring_service.detect_anomaly(
            model_id=model_id,
            data=data,
            deployment_id=deployment_id
        )
        return AnomalyDetection(**result)
    except Exception as e:
        logger.error(f"Error detecting anomaly: {e}")
        raise HTTPException(status_code=500, detail=str(e))

