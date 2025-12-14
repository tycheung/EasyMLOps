"""
Dashboard service
Handles dashboard metrics aggregation
"""

import logging
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import desc, func, select

from app.database import get_session
from app.models.monitoring import AlertDB, PredictionLogDB
from app.models.model import Model, ModelDeployment
from app.schemas.monitoring import DashboardMetrics
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class DashboardService(BaseMonitoringService):
    """Service for dashboard metrics"""
    
    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get comprehensive dashboard metrics"""
        try:
            async with get_session() as session:
                # Get basic counts
                total_models_stmt = select(func.count(Model.id))
                total_models = await session.execute(total_models_stmt)
                
                active_deployments_stmt = select(func.count(ModelDeployment.id)).where(
                    ModelDeployment.status == "active"
                )
                active_deployments = await session.execute(active_deployments_stmt)
                
                # Get today's predictions
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_predictions_stmt = select(func.count(PredictionLogDB.id)).where(
                    PredictionLogDB.timestamp >= today_start
                )
                today_predictions = await session.execute(today_predictions_stmt)
                
                # Get today's average response time
                today_avg_latency_stmt = select(func.avg(PredictionLogDB.latency_ms)).where(
                    PredictionLogDB.timestamp >= today_start
                )
                today_avg_latency = await session.execute(today_avg_latency_stmt)
                
                # Get active alerts
                active_alerts_stmt = select(func.count(AlertDB.id)).where(
                    AlertDB.is_active == True
                )
                active_alerts = await session.execute(active_alerts_stmt)
                
                # Get system health metrics - will need to delegate to health service
                # For now, use a placeholder
                from app.schemas.monitoring import SystemHealthStatus, SystemStatus
                health_status = SystemHealthStatus(
                    overall_status=SystemStatus.OPERATIONAL,
                    components=[]  # components is a list, not a dict
                )
                
                # Get recent deployments
                recent_deployments_stmt = select(ModelDeployment).order_by(
                    desc(ModelDeployment.created_at)
                ).limit(5)
                recent_deployments_query = await session.execute(recent_deployments_stmt)
                recent_deployments = [
                    {
                        "id": dep.id,
                        "name": dep.name,
                        "model_id": dep.model_id,
                        "status": dep.status,
                        "created_at": dep.created_at.isoformat()
                    }
                    for dep in recent_deployments_query.scalars().all()
                ]
                
                # Generate trend data (simplified - in production would be more sophisticated)
                request_trend_24h = [100, 120, 95, 130, 145, 160, 180, 165, 150, 140, 135, 125,
                                   110, 115, 105, 130, 140, 155, 170, 165, 160, 145, 130, 120]
                error_trend_24h = [0.5, 0.8, 0.3, 1.2, 0.9, 0.6, 0.4, 0.7, 0.5, 0.8, 0.6, 0.4,
                                  0.3, 0.5, 0.7, 0.9, 0.6, 0.4, 0.5, 0.6, 0.8, 0.7, 0.5, 0.4]
                
                return DashboardMetrics(
                    total_models=total_models.scalar() or 0,
                    active_deployments=active_deployments.scalar() or 0,
                    total_predictions_today=today_predictions.scalar() or 0,
                    avg_response_time_today=float(today_avg_latency.scalar() or 0),
                    system_status=health_status.overall_status,
                    active_alerts=active_alerts.scalar() or 0,
                    cpu_usage=0,  # Would get from health_status.components in production
                    memory_usage=0,  # Would get from health_status.components in production
                    most_used_model=None,  # Would implement most used model logic
                    fastest_model=None,  # Would implement fastest model logic
                    recent_deployments=recent_deployments,
                    request_trend_24h=request_trend_24h,
                    error_trend_24h=error_trend_24h
                )
                
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            raise

