"""
Dashboard and health monitoring routes
Provides endpoints for dashboard metrics and system health
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.monitoring import DashboardMetrics, SystemHealthStatus, ModelPerformanceMetrics
from typing import List, Dict, Any
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.get("/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get comprehensive dashboard metrics"""
    try:
        dashboard_metrics = await monitoring_service.get_dashboard_metrics()
        return dashboard_metrics
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=SystemHealthStatus)
async def get_system_health():
    """Get overall system health status"""
    try:
        health_status = await monitoring_service.get_system_health_status()
        return health_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/resources")
async def get_model_resource_usage(
    model_id: str,
    deployment_id: Optional[str] = Query(None)
):
    """Collect model resource usage metrics"""
    try:
        usage = await monitoring_service.collect_model_resource_usage(
            model_id=model_id,
            deployment_id=deployment_id
        )
        return usage
    except Exception as e:
        logger.error(f"Error collecting model resource usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/performance", response_model=ModelPerformanceMetrics)
async def get_model_performance(
    model_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    deployment_id: Optional[str] = Query(None)
):
    """Get performance metrics for a specific model"""
    try:
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        metrics = await monitoring_service.get_model_performance_metrics(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id
        )
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/predictions/logs")
async def get_prediction_logs(
    model_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return")
):
    """Get prediction logs for a model"""
    try:
        logs = await monitoring_service.get_prediction_logs(model_id=model_id, limit=limit)
        return logs
    except Exception as e:
        logger.error(f"Error getting prediction logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/metrics/aggregated")
async def get_aggregated_metrics(
    model_id: str,
    time_range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d")
):
    """Get aggregated metrics"""
    try:
        metrics = await monitoring_service.get_aggregated_metrics(model_id=model_id, time_range=time_range)
        return metrics
    except Exception as e:
        logger.error(f"Error getting aggregated metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}/summary")
async def get_deployment_summary(deployment_id: str):
    """Get deployment summary"""
    try:
        summary = await monitoring_service.get_deployment_summary(deployment_id=deployment_id)
        return summary
    except Exception as e:
        logger.error(f"Error getting deployment summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/confidence")
async def get_confidence_metrics(
    model_id: str,
    start_time: datetime = Query(..., description="Start time for analysis"),
    end_time: datetime = Query(..., description="End time for analysis"),
    deployment_id: Optional[str] = Query(None),
    low_confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
    high_uncertainty_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """Calculate confidence and uncertainty metrics"""
    try:
        from app.schemas.monitoring import ModelConfidenceMetrics
        metrics = await monitoring_service.calculate_confidence_metrics(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id,
            low_confidence_threshold=low_confidence_threshold,
            high_uncertainty_threshold=high_uncertainty_threshold
        )
        return metrics
    except Exception as e:
        logger.error(f"Error calculating confidence metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

