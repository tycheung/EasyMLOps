"""
FastAPI routes for monitoring and operations
Provides endpoints for accessing performance metrics, system health, alerts, and audit logs
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import logging

from app.schemas.monitoring import (
    PredictionLog, ModelPerformanceMetrics, SystemHealthStatus, Alert, AuditLog,
    ModelUsageAnalytics, DashboardMetrics, SystemHealthMetric
)
from app.services.monitoring_service import monitoring_service
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


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
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
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
    except Exception as e:
        logger.error(f"Error getting model performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    active_only: bool = Query(True),
    severity: Optional[str] = Query(None),
    component: Optional[str] = Query(None),
    limit: int = Query(50)
):
    """Get system alerts"""
    try:
        async with get_session() as session:
            from app.models.monitoring import AlertDB
            from sqlalchemy import and_, desc
            
            query = session.query(AlertDB)
            
            if active_only:
                query = query.filter(AlertDB.is_active == True)
            if severity:
                query = query.filter(AlertDB.severity == severity)
            if component:
                query = query.filter(AlertDB.component == component)
            
            query = query.order_by(desc(AlertDB.triggered_at)).limit(limit)
            result = await session.execute(query)
            alerts = result.scalars().all()
            
            return [
                Alert(
                    id=alert.id,
                    severity=alert.severity,
                    component=alert.component,
                    title=alert.title,
                    description=alert.description,
                    triggered_at=alert.triggered_at,
                    resolved_at=alert.resolved_at,
                    acknowledged_at=alert.acknowledged_at,
                    acknowledged_by=alert.acknowledged_by,
                    metric_value=alert.metric_value,
                    threshold_value=alert.threshold_value,
                    affected_models=alert.affected_models or [],
                    is_active=alert.is_active,
                    is_acknowledged=alert.is_acknowledged
                )
                for alert in alerts
            ]
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def log_prediction_middleware(
    request: Request,
    model_id: str,
    input_data: Dict[str, Any],
    output_data: Any,
    latency_ms: float,
    success: bool = True,
    error_message: Optional[str] = None
):
    """Middleware function to log predictions from other routes"""
    try:
        await monitoring_service.log_prediction(
            model_id=model_id,
            deployment_id=None,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            api_endpoint=str(request.url),
            success=success,
            error_message=error_message,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None
        )
    except Exception as e:
        logger.error(f"Error logging prediction: {e}") 