"""
Monitoring and operations service for EasyMLOps platform
Handles model performance monitoring, system health monitoring, alerts, and audit logging
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics
import uuid

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import sessionmaker

from app.database import get_session
from app.models.monitoring import (
    PredictionLogDB, ModelPerformanceMetricsDB, SystemHealthMetricDB,
    AlertDB, AuditLogDB, ModelUsageAnalyticsDB, SystemStatusDB
)
from app.models.model import Model, ModelDeployment
from app.schemas.monitoring import (
    PredictionLog, ModelPerformanceMetrics, SystemHealthMetric, SystemHealthStatus,
    Alert, AuditLog, ModelUsageAnalytics, DashboardMetrics, MetricType, AlertSeverity,
    SystemComponent
)
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class MonitoringService:
    """Service for handling all monitoring and operations functionality"""
    
    def __init__(self):
        self.start_time = time.time()
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 5.0,
            "avg_latency_ms": 1000.0
        }
    
    # 5.1 Model Performance Monitoring
    
    async def log_prediction(
        self,
        model_id: str,
        deployment_id: Optional[str],
        input_data: Dict[str, Any],
        output_data: Any,
        latency_ms: float,
        api_endpoint: str,
        success: bool = True,
        error_message: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Log individual prediction for performance monitoring"""
        try:
            log_entry = PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=model_id,
                deployment_id=deployment_id,
                request_id=str(uuid.uuid4()),
                input_data=input_data,
                output_data=output_data,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                user_agent=user_agent,
                ip_address=ip_address,
                api_endpoint=api_endpoint,
                success=success,
                error_message=error_message
            )
            
            async with get_session() as session:
                session.add(log_entry)
                await session.commit()
                logger.info(f"Logged prediction for model {model_id}")
                return log_entry.id
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            raise
    
    async def get_model_performance_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None
    ) -> ModelPerformanceMetrics:
        """Calculate aggregated performance metrics for a model"""
        try:
            async with get_session() as session:
                # Build query filter
                query_filter = and_(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time
                )
                
                if deployment_id:
                    query_filter = and_(query_filter, PredictionLogDB.deployment_id == deployment_id)
                
                # Get all prediction logs for the time window
                result = await session.execute(
                    session.query(PredictionLogDB).filter(query_filter).all()
                )
                logs = result.scalars().all()
                
                if not logs:
                    return ModelPerformanceMetrics(
                        model_id=model_id,
                        deployment_id=deployment_id,
                        time_window_start=start_time,
                        time_window_end=end_time,
                        total_requests=0,
                        successful_requests=0,
                        failed_requests=0,
                        requests_per_minute=0.0,
                        avg_latency_ms=0.0,
                        p50_latency_ms=0.0,
                        p95_latency_ms=0.0,
                        p99_latency_ms=0.0,
                        max_latency_ms=0.0,
                        success_rate=0.0,
                        error_rate=0.0
                    )
                
                # Calculate metrics
                total_requests = len(logs)
                successful_requests = sum(1 for log in logs if log.success)
                failed_requests = total_requests - successful_requests
                
                # Calculate time-based metrics
                time_diff_minutes = (end_time - start_time).total_seconds() / 60
                requests_per_minute = total_requests / time_diff_minutes if time_diff_minutes > 0 else 0
                
                # Calculate latency metrics
                latencies = [log.latency_ms for log in logs]
                latencies.sort()
                
                avg_latency_ms = statistics.mean(latencies)
                p50_latency_ms = statistics.median(latencies)
                p95_latency_ms = latencies[int(0.95 * len(latencies))] if latencies else 0
                p99_latency_ms = latencies[int(0.99 * len(latencies))] if latencies else 0
                max_latency_ms = max(latencies) if latencies else 0
                
                # Calculate success/error rates
                success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
                error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
                
                return ModelPerformanceMetrics(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    total_requests=total_requests,
                    successful_requests=successful_requests,
                    failed_requests=failed_requests,
                    requests_per_minute=requests_per_minute,
                    avg_latency_ms=avg_latency_ms,
                    p50_latency_ms=p50_latency_ms,
                    p95_latency_ms=p95_latency_ms,
                    p99_latency_ms=p99_latency_ms,
                    max_latency_ms=max_latency_ms,
                    success_rate=success_rate,
                    error_rate=error_rate
                )
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    async def store_performance_metrics(self, metrics: ModelPerformanceMetrics) -> str:
        """Store aggregated performance metrics in database"""
        try:
            metrics_db = ModelPerformanceMetricsDB(
                id=str(uuid.uuid4()),
                model_id=metrics.model_id,
                deployment_id=metrics.deployment_id,
                time_window_start=metrics.time_window_start,
                time_window_end=metrics.time_window_end,
                total_requests=metrics.total_requests,
                successful_requests=metrics.successful_requests,
                failed_requests=metrics.failed_requests,
                requests_per_minute=metrics.requests_per_minute,
                avg_latency_ms=metrics.avg_latency_ms,
                p50_latency_ms=metrics.p50_latency_ms,
                p95_latency_ms=metrics.p95_latency_ms,
                p99_latency_ms=metrics.p99_latency_ms,
                max_latency_ms=metrics.max_latency_ms,
                success_rate=metrics.success_rate,
                error_rate=metrics.error_rate
            )
            
            async with get_session() as session:
                session.add(metrics_db)
                await session.commit()
                logger.info(f"Stored performance metrics for model {metrics.model_id}")
                return metrics_db.id
                
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
            raise
    
    # 5.2 System Health Monitoring
    
    async def collect_system_health_metrics(self) -> List[SystemHealthMetric]:
        """Collect current system health metrics"""
        try:
            metrics = []
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(SystemHealthMetric(
                component=SystemComponent.API_SERVER,
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                unit="percent",
                host=psutil.uname().node if hasattr(psutil, 'uname') else "unknown"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(SystemHealthMetric(
                component=SystemComponent.API_SERVER,
                metric_type=MetricType.MEMORY_USAGE,
                value=memory.percent,
                unit="percent",
                host=psutil.uname().node if hasattr(psutil, 'uname') else "unknown"
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.append(SystemHealthMetric(
                component=SystemComponent.STORAGE,
                metric_type=MetricType.DISK_USAGE,
                value=disk.percent,
                unit="percent",
                host=psutil.uname().node if hasattr(psutil, 'uname') else "unknown"
            ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system health metrics: {e}")
            return []
    
    async def store_health_metric(self, metric: SystemHealthMetric) -> str:
        """Store system health metric in database"""
        try:
            metric_db = SystemHealthMetricDB(
                id=str(uuid.uuid4()),
                component=metric.component.value,
                metric_type=metric.metric_type.value,
                value=metric.value,
                unit=metric.unit,
                timestamp=metric.timestamp,
                host=metric.host,
                tags=metric.tags
            )
            
            async with get_session() as session:
                session.add(metric_db)
                await session.commit()
                return metric_db.id
                
        except Exception as e:
            logger.error(f"Error storing health metric: {e}")
            raise
    
    async def get_system_health_status(self) -> SystemHealthStatus:
        """Get overall system health status"""
        try:
            # Collect current metrics
            current_metrics = await self.collect_system_health_metrics()
            
            # Store metrics
            for metric in current_metrics:
                await self.store_health_metric(metric)
            
            # Determine overall status
            cpu_usage = next((m.value for m in current_metrics if m.metric_type == MetricType.CPU_USAGE), 0)
            memory_usage = next((m.value for m in current_metrics if m.metric_type == MetricType.MEMORY_USAGE), 0)
            
            overall_status = "healthy"
            if cpu_usage > 90 or memory_usage > 90:
                overall_status = "critical"
            elif cpu_usage > 80 or memory_usage > 80:
                overall_status = "warning"
            
            # Build component status
            components = {
                "api_server": {
                    "status": "healthy" if cpu_usage < 80 else "warning",
                    "cpu": cpu_usage,
                    "memory": memory_usage
                },
                "database": {
                    "status": "healthy",  # Would check database connection
                    "connections": 0,  # Would query actual connection count
                    "response_time": 0.0
                },
                "model_services": {
                    "status": "healthy",
                    "active_models": 0,  # Would query actual count
                    "avg_latency": 0.0
                }
            }
            
            return SystemHealthStatus(
                overall_status=overall_status,
                components=components,
                uptime_seconds=time.time() - self.start_time
            )
            
        except Exception as e:
            logger.error(f"Error getting system health status: {e}")
            raise
    
    # Alert Management
    
    async def check_and_create_alerts(self) -> List[Alert]:
        """Check metrics against thresholds and create alerts if needed"""
        try:
            alerts = []
            
            # Get recent system health metrics
            current_time = datetime.utcnow()
            check_window = current_time - timedelta(minutes=5)
            
            async with get_session() as session:
                # Check CPU usage
                cpu_metrics = await session.execute(
                    session.query(SystemHealthMetricDB).filter(
                        and_(
                            SystemHealthMetricDB.metric_type == MetricType.CPU_USAGE.value,
                            SystemHealthMetricDB.timestamp >= check_window
                        )
                    ).all()
                )
                
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
    
    # 5.3 Audit & Logging
    
    async def log_audit_event(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log audit event for compliance and tracking"""
        try:
            audit_log = AuditLogDB(
                id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                old_values=old_values,
                new_values=new_values,
                success=success,
                error_message=error_message,
                additional_data=additional_data or {}
            )
            
            async with get_session() as session:
                session.add(audit_log)
                await session.commit()
                
                logger.info(f"Logged audit event: {action} on {resource_type} {resource_id}")
                return audit_log.id
                
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            raise
    
    # Dashboard and Analytics
    
    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get comprehensive dashboard metrics"""
        try:
            async with get_session() as session:
                # Get basic counts
                total_models = await session.execute(
                    session.query(func.count(Model.id)).scalar()
                )
                
                active_deployments = await session.execute(
                    session.query(func.count(ModelDeployment.id)).filter(
                        ModelDeployment.status == "active"
                    ).scalar()
                )
                
                # Get today's predictions
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_predictions = await session.execute(
                    session.query(func.count(PredictionLogDB.id)).filter(
                        PredictionLogDB.timestamp >= today_start
                    ).scalar()
                )
                
                # Get today's average response time
                today_avg_latency = await session.execute(
                    session.query(func.avg(PredictionLogDB.latency_ms)).filter(
                        PredictionLogDB.timestamp >= today_start
                    ).scalar()
                )
                
                # Get active alerts
                active_alerts = await session.execute(
                    session.query(func.count(AlertDB.id)).filter(
                        AlertDB.is_active == True
                    ).scalar()
                )
                
                # Get system health metrics
                health_status = await self.get_system_health_status()
                
                # Get recent deployments
                recent_deployments_query = await session.execute(
                    session.query(ModelDeployment).order_by(
                        desc(ModelDeployment.created_at)
                    ).limit(5).all()
                )
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
                    cpu_usage=health_status.components.get("api_server", {}).get("cpu", 0),
                    memory_usage=health_status.components.get("api_server", {}).get("memory", 0),
                    most_used_model=None,  # Would implement most used model logic
                    fastest_model=None,  # Would implement fastest model logic
                    recent_deployments=recent_deployments,
                    request_trend_24h=request_trend_24h,
                    error_trend_24h=error_trend_24h
                )
                
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            raise
    
    # Background monitoring tasks
    
    async def start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        logger.info("Starting monitoring background tasks")
        
        # Schedule regular health checks
        asyncio.create_task(self._health_check_loop())
        
        # Schedule metrics aggregation
        asyncio.create_task(self._metrics_aggregation_loop())
        
        # Schedule alert checking
        asyncio.create_task(self._alert_check_loop())
    
    async def _health_check_loop(self):
        """Background task for regular health checks"""
        while True:
            try:
                await self.get_system_health_status()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_aggregation_loop(self):
        """Background task for aggregating performance metrics"""
        while True:
            try:
                # Aggregate metrics for all models every 5 minutes
                current_time = datetime.utcnow()
                start_time = current_time - timedelta(minutes=5)
                
                async with get_session() as session:
                    models = await session.execute(session.query(Model).all())
                    
                    for model in models.scalars():
                        metrics = await self.get_model_performance_metrics(
                            model.id, start_time, current_time
                        )
                        if metrics.total_requests > 0:
                            await self.store_performance_metrics(metrics)
                
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
                await asyncio.sleep(300)
    
    async def _alert_check_loop(self):
        """Background task for checking alerts"""
        while True:
            try:
                await self.check_and_create_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")
                await asyncio.sleep(60)


# Global monitoring service instance
monitoring_service = MonitoringService() 