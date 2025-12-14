"""
Alert service
Handles alert creation, management, and resolution
"""

import asyncio
import logging
import statistics
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, select

from app.database import get_session
from app.models.monitoring import AlertDB, SystemHealthMetricDB
from app.schemas.monitoring import (
    Alert, AlertSeverity, MetricType, SystemComponent
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class AlertService(BaseMonitoringService):
    """Service for alert management"""
    
    async def check_and_create_alerts(self) -> List[Alert]:
        """Check metrics against thresholds and create alerts if needed"""
        try:
            alerts = []
            
            # Get recent system health metrics
            current_time = datetime.utcnow()
            check_window = current_time - timedelta(minutes=5)
            
            async with get_session() as session:
                # Check CPU usage
                cpu_stmt = select(SystemHealthMetricDB).where(
                    and_(
                        SystemHealthMetricDB.metric_type == MetricType.CPU_USAGE.value,
                        SystemHealthMetricDB.timestamp >= check_window
                    )
                )
                cpu_metrics = await session.execute(cpu_stmt)
                
                recent_cpu = cpu_metrics.scalars().all()
                if recent_cpu:
                    avg_cpu = statistics.mean([m.value for m in recent_cpu])
                    if avg_cpu > self.alert_thresholds["cpu_usage"]:
                        alert = await self.create_alert(
                            severity=AlertSeverity.WARNING,
                            component=SystemComponent.API_SERVER,
                            title="High CPU Usage",
                            description=f"CPU usage averaged {avg_cpu:.1f}% over the last 5 minutes",
                            metric_value=avg_cpu,
                            threshold_value=self.alert_thresholds["cpu_usage"]
                        )
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
    
    async def create_alert(
        self,
        severity: AlertSeverity,
        component: SystemComponent,
        title: str,
        description: str,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        affected_models: Optional[List[str]] = None
    ) -> Alert:
        """Create a new system alert"""
        try:
            alert = AlertDB(
                id=str(uuid.uuid4()),
                severity=severity.value,
                component=component.value,
                title=title,
                description=description,
                metric_value=metric_value,
                threshold_value=threshold_value,
                affected_models=affected_models or []
            )
            
            async with get_session() as session:
                session.add(alert)
                await session.commit()
                
                logger.warning(f"Created {severity.value} alert: {title}")
                
                return Alert(
                    id=alert.id,
                    severity=severity,
                    component=component,
                    title=title,
                    description=description,
                    triggered_at=alert.triggered_at,
                    metric_value=metric_value,
                    threshold_value=threshold_value,
                    affected_models=affected_models or [],
                    is_active=True,
                    is_acknowledged=False
                )
                
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise
    
    async def get_alerts(self, severity: Optional[str] = None, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts"""
        try:
            async with get_session() as session:
                query = select(AlertDB)
                if severity:
                    query = query.where(AlertDB.severity == severity)
                if resolved is not None:
                    query = query.where(AlertDB.resolved == resolved)
                # Use triggered_at instead of created_at (AlertDB doesn't have created_at)
                query = query.order_by(desc(AlertDB.triggered_at))
                
                result = await session.execute(query)
                alerts = result.scalars().all()
                
                return [
                    {
                        "id": alert.id,
                        "severity": alert.severity,
                        "component": alert.component,
                        "title": alert.title,
                        "description": alert.description,
                        "resolved": alert.resolved,
                        "triggered_at": alert.triggered_at.isoformat()
                    }
                    for alert in alerts
                ]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(AlertDB).where(AlertDB.id == alert_id)
                )
                alert = result.scalar_one_or_none()
                
                if alert:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    await session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> None:
        """Acknowledge an alert"""
        try:
            async with get_session() as session:
                alert_db = await session.get(AlertDB, alert_id)
                if not alert_db:
                    raise ValueError(f"Alert {alert_id} not found")
                
                alert_db.is_acknowledged = True
                alert_db.acknowledged_at = datetime.utcnow()
                alert_db.acknowledged_by = acknowledged_by
                
                await session.commit()
            
            # Note: Audit logging will be handled via service composition
            # await self.log_audit_event(...)
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}", exc_info=True)
            raise
    
    async def _alert_check_loop(self):
        """Background task for checking alerts"""
        while True:
            try:
                await self.check_and_create_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")
                await asyncio.sleep(60)

