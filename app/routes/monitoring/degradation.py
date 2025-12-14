"""
Performance degradation routes
Provides endpoints for detecting and logging performance degradation
"""

from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
import logging

from app.schemas.monitoring import ModelPerformanceHistory
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/models/{model_id}/degradation/log", response_model=Dict[str, str], status_code=201)
async def log_prediction_with_ground_truth(
    model_id: str,
    prediction: float,
    ground_truth: float,
    deployment_id: Optional[str] = Query(None),
    input_data: Optional[Dict[str, Any]] = Body(None)
):
    """Log prediction with ground truth for degradation detection"""
    try:
        log_id = await monitoring_service.log_prediction_with_ground_truth(
            model_id=model_id,
            deployment_id=deployment_id,
            prediction=prediction,
            ground_truth=ground_truth,
            input_data=input_data or {}
        )
        return {"id": log_id, "message": "Prediction logged with ground truth"}
    except Exception as e:
        logger.error(f"Error logging prediction with ground truth: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/degradation/detect", response_model=ModelPerformanceHistory)
async def detect_performance_degradation(
    model_id: str,
    start_time: datetime = Query(..., description="Start time for analysis"),
    end_time: datetime = Query(..., description="End time for analysis"),
    deployment_id: Optional[str] = Query(None),
    degradation_threshold: float = Query(0.1, description="Degradation threshold")
):
    """Detect performance degradation for a model"""
    try:
        result = await monitoring_service.detect_performance_degradation(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id,
            degradation_threshold=degradation_threshold
        )
        return result
    except Exception as e:
        logger.error(f"Error detecting performance degradation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

