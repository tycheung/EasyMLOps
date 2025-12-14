"""
Metrics aggregation service
Handles aggregated metrics and deployment summaries
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import select

from app.database import get_session
from app.models.model import ModelDeployment
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class MetricsAggregationService(BaseMonitoringService):
    """Service for metrics aggregation"""
    
    def __init__(self, metrics_calculation_service):
        """Initialize with metrics calculation service"""
        self.metrics_calculation = metrics_calculation_service
    
    async def get_aggregated_metrics(self, model_id: Optional[str] = None, time_range: str = "24h") -> Dict[str, Any]:
        """Get aggregated metrics"""
        try:
            # Parse time range
            if time_range == "1h":
                start_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                start_time = datetime.utcnow() - timedelta(hours=24)
            elif time_range == "7d":
                start_time = datetime.utcnow() - timedelta(days=7)
            else:
                start_time = datetime.utcnow() - timedelta(hours=24)
            
            end_time = datetime.utcnow()
            
            if model_id:
                metrics = await self.metrics_calculation.get_model_performance_metrics(model_id, start_time, end_time)
                return {
                    "model_id": model_id,
                    "total_requests": metrics.total_requests,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate
                }
            else:
                # System-wide metrics - this will need to be implemented in health service
                return {
                    "time_range": time_range
                }
        except Exception as e:
            logger.error(f"Error getting aggregated metrics: {e}")
            return {}
    
    async def get_deployment_summary(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment summary"""
        try:
            async with get_session() as session:
                # Get deployment info
                result = await session.execute(
                    select(ModelDeployment).where(ModelDeployment.id == deployment_id)
                )
                deployment = result.scalar_one_or_none()
                
                if not deployment:
                    return {"error": "Deployment not found"}
                
                # Get recent metrics
                start_time = datetime.utcnow() - timedelta(hours=24)
                end_time = datetime.utcnow()
                
                metrics = await self.metrics_calculation.get_model_performance_metrics(
                    deployment.model_id, start_time, end_time, deployment_id
                )
                
                return {
                    "deployment_id": deployment_id,
                    "model_id": deployment.model_id,
                    "status": deployment.status,
                    "total_requests": metrics.total_requests,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "success_rate": metrics.success_rate,
                    "last_24h_metrics": {
                        "requests": metrics.total_requests,
                        "errors": metrics.failed_requests
                    }
                }
        except Exception as e:
            logger.error(f"Error getting deployment summary: {e}")
            return {"error": str(e)}

