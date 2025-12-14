"""
System health monitoring service
Handles system health metrics, resource usage, and health checks
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Optional import for psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system health monitoring will be limited")

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import ModelResourceUsageDB, SystemHealthMetricDB
from app.models.model import Model
from app.schemas.monitoring import (
    ModelResourceUsage, SystemHealthMetric, SystemHealthStatus,
    MetricType, SystemComponent, SystemStatus
)
from app.config import get_settings
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)
settings = get_settings()


class SystemHealthService(BaseMonitoringService):
    """Service for system health monitoring"""
    
    async def collect_system_health_metrics(self) -> List[SystemHealthMetric]:
        """Collect current system health metrics"""
        try:
            metrics = []
            node_name = "unknown"
            if PSUTIL_AVAILABLE and hasattr(psutil, 'uname'):
                node_name = await asyncio.to_thread(getattr, psutil.uname(), 'node')

            # CPU usage
            if not PSUTIL_AVAILABLE:
                logger.warning("psutil not available - skipping CPU metrics")
                cpu_percent = 0.0
            else:
                cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=1)
            cpu_status = SystemStatus.OPERATIONAL
            if cpu_percent > self.alert_thresholds.get("cpu_usage_critical", 90.0):
                cpu_status = SystemStatus.UNHEALTHY
            elif cpu_percent > self.alert_thresholds.get("cpu_usage_warning", 75.0):
                cpu_status = SystemStatus.DEGRADED
            metrics.append(SystemHealthMetric(
                component=SystemComponent.API_SERVER,
                status=cpu_status,
                message=f"CPU at {cpu_percent}%",
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                unit="percent",
                host=node_name
            ))
            
            # Memory usage
            if not PSUTIL_AVAILABLE:
                logger.warning("psutil not available - skipping memory metrics")
                memory_percent = 0.0
            else:
                memory = await asyncio.to_thread(psutil.virtual_memory)
                memory_percent = memory.percent
            memory_status = SystemStatus.OPERATIONAL
            if memory_percent > self.alert_thresholds.get("memory_usage_critical", 90.0):
                memory_status = SystemStatus.UNHEALTHY
            elif memory_percent > self.alert_thresholds.get("memory_usage_warning", 80.0):
                memory_status = SystemStatus.DEGRADED
            metrics.append(SystemHealthMetric(
                component=SystemComponent.API_SERVER,
                status=memory_status,
                message=f"Memory at {memory_percent}%",
                metric_type=MetricType.MEMORY_USAGE,
                value=memory_percent,
                unit="percent",
                host=node_name
            ))
            
            # Disk usage for root directory
            if not PSUTIL_AVAILABLE:
                logger.warning("psutil not available - skipping disk metrics")
                disk_root_percent = 0.0
            else:
                disk_root = await asyncio.to_thread(psutil.disk_usage, '/')
                disk_root_percent = disk_root.percent
            disk_root_status = SystemStatus.OPERATIONAL
            if disk_root_percent > self.alert_thresholds.get("disk_usage_critical", 90.0):
                disk_root_status = SystemStatus.UNHEALTHY
            elif disk_root_percent > self.alert_thresholds.get("disk_usage_warning", 80.0):
                disk_root_status = SystemStatus.DEGRADED
            metrics.append(SystemHealthMetric(
                component=SystemComponent.STORAGE,
                status=disk_root_status,
                message=f"Root disk usage at {disk_root_percent}% on {node_name}",
                metric_type=MetricType.DISK_USAGE,
                mount_point='/',
                value=disk_root_percent,
                unit="percent",
                host=node_name
            ))

            # Disk usage for models directory
            try:
                models_dir_path = settings.MODELS_DIR
                abs_models_dir_path = os.path.abspath(models_dir_path)
                if os.path.exists(abs_models_dir_path) and PSUTIL_AVAILABLE:
                    disk_models = await asyncio.to_thread(psutil.disk_usage, abs_models_dir_path)
                    disk_models_percent = disk_models.percent
                    disk_models_status = SystemStatus.OPERATIONAL
                    if disk_models_percent > self.alert_thresholds.get("disk_usage_models_critical", 90.0):
                        disk_models_status = SystemStatus.UNHEALTHY
                    elif disk_models_percent > self.alert_thresholds.get("disk_usage_models_warning", 80.0):
                        disk_models_status = SystemStatus.DEGRADED
                    metrics.append(SystemHealthMetric(
                        component=SystemComponent.STORAGE,
                        status=disk_models_status,
                        message=f"Models disk usage at {disk_models_percent}% on {node_name}",
                        metric_type=MetricType.DISK_USAGE_MODELS,
                        mount_point=abs_models_dir_path,
                        value=disk_models_percent,
                        unit="percent",
                        host=node_name
                    ))
            except Exception as e:
                logger.warning(f"Could not get disk usage for models directory {settings.MODELS_DIR}: {e}")
                metrics.append(SystemHealthMetric(
                    component=SystemComponent.STORAGE,
                    status=SystemStatus.DEGRADED,
                    message=f"Could not get disk usage for models directory: {e}",
                    metric_type=MetricType.DISK_USAGE_MODELS,
                    host=node_name
                ))
            
            # Database health
            db_healthy, db_message = await self._check_database_health()
            db_status = SystemStatus.OPERATIONAL if db_healthy else SystemStatus.UNHEALTHY
            metrics.append(SystemHealthMetric(
                component=SystemComponent.DATABASE,
                status=db_status,
                message=db_message,
                metric_type=MetricType.DB_CONNECTION_STATUS,
                value=1.0 if db_healthy else 0.0,
                unit="status",
                host=settings.POSTGRES_SERVER if not settings.is_sqlite() else "sqlite_db"
            ))

            # BentoML system health
            bento_healthy, bento_message = await self.check_bentoml_system_health()
            bento_status = SystemStatus.OPERATIONAL if bento_healthy else SystemStatus.UNHEALTHY
            metrics.append(SystemHealthMetric(
                component=SystemComponent.BENTOML,
                status=bento_status,
                message=bento_message,
                metric_type=MetricType.BENTOML_SERVICE_STATUS,
                value=1.0 if bento_healthy else 0.0,
                unit="status"
            ))
            
            # Network I/O metrics
            try:
                net_io = await asyncio.to_thread(psutil.net_io_counters)
                if net_io:
                    metrics.append(SystemHealthMetric(
                        component=SystemComponent.API_SERVER,
                        status=SystemStatus.OPERATIONAL,
                        message=f"Network bytes sent: {net_io.bytes_sent}",
                        metric_type=MetricType.NETWORK_BYTES_SENT,
                        value=float(net_io.bytes_sent),
                        unit="bytes",
                        host=node_name
                    ))
                    
                    metrics.append(SystemHealthMetric(
                        component=SystemComponent.API_SERVER,
                        status=SystemStatus.OPERATIONAL,
                        message=f"Network bytes received: {net_io.bytes_recv}",
                        metric_type=MetricType.NETWORK_BYTES_RECV,
                        value=float(net_io.bytes_recv),
                        unit="bytes",
                        host=node_name
                    ))
                    
                    metrics.append(SystemHealthMetric(
                        component=SystemComponent.API_SERVER,
                        status=SystemStatus.OPERATIONAL,
                        message=f"Network packets sent: {net_io.packets_sent}",
                        metric_type=MetricType.NETWORK_PACKETS_SENT,
                        value=float(net_io.packets_sent),
                        unit="packets",
                        host=node_name
                    ))
                    
                    metrics.append(SystemHealthMetric(
                        component=SystemComponent.API_SERVER,
                        status=SystemStatus.OPERATIONAL,
                        message=f"Network packets received: {net_io.packets_recv}",
                        metric_type=MetricType.NETWORK_PACKETS_RECV,
                        value=float(net_io.packets_recv),
                        unit="packets",
                        host=node_name
                    ))
            except Exception as e:
                logger.warning(f"Could not get network I/O metrics: {e}")
            
            # GPU usage tracking
            try:
                gpu_metrics = await self._collect_gpu_metrics(node_name)
                metrics.extend(gpu_metrics)
            except Exception as e:
                logger.debug(f"GPU metrics not available: {e}")
            
            logger.debug(f"Collected system health metrics: {len(metrics)} metrics")
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system health metrics: {e}", exc_info=True)
            return [SystemHealthMetric(
                component=SystemComponent.SYSTEM,
                status=SystemStatus.UNHEALTHY,
                message=f"Error collecting system health metrics: {e}"
            )]
    
    async def store_health_metric(self, metric: SystemHealthMetric) -> str:
        """Store system health metric in database"""
        try:
            import uuid
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
            component_metrics = await self.collect_system_health_metrics()
            
            # Determine overall status based on components
            overall_status = SystemStatus.OPERATIONAL
            if any(m.status == SystemStatus.UNHEALTHY for m in component_metrics):
                overall_status = SystemStatus.UNHEALTHY
            elif any(m.status == SystemStatus.DEGRADED for m in component_metrics):
                overall_status = SystemStatus.DEGRADED
            
            return SystemHealthStatus(
                overall_status=overall_status,
                components=component_metrics
            )
            
        except Exception as e:
            logger.error(f"Error getting system health status: {e}", exc_info=True)
            return SystemHealthStatus(
                overall_status=SystemStatus.UNHEALTHY,
                components=[SystemHealthMetric(
                    component=SystemComponent.SYSTEM,
                    status=SystemStatus.UNHEALTHY,
                    message=f"Failed to determine system health: {e}"
                )]
            )
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check basic system health"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            return {
                "status": "healthy" if cpu_percent < 80 and memory_info.percent < 85 else "unhealthy",
                "cpu_usage": cpu_percent,
                "memory_usage": memory_info.percent,
                "disk_usage": disk_info.percent,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status (alias for check_system_health)"""
        return await self.check_system_health()
    
    async def check_bentoml_system_health(self) -> Tuple[bool, str]:
        """Placeholder for checking BentoML system health"""
        logger.info("Simulating BentoML system health check (placeholder).")
        return True, "BentoML services appear to be operational (simulated)."
    
    async def collect_model_resource_usage(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        time_window_start: Optional[datetime] = None,
        time_window_end: Optional[datetime] = None
    ) -> ModelResourceUsage:
        """Collect resource usage metrics for a specific model"""
        try:
            now = datetime.utcnow()
            if not time_window_start:
                time_window_start = now - timedelta(minutes=5)
            if not time_window_end:
                time_window_end = now
            
            # Note: get_model_performance_metrics will be accessed via service composition
            # For now, we'll need to handle this dependency
            
            # Get system resource metrics
            system_metrics = await self.collect_system_health_metrics()
            
            # Extract relevant metrics
            cpu_usage = None
            memory_usage_mb = None
            gpu_usage = None
            gpu_memory_mb = None
            network_bytes_sent = None
            network_bytes_recv = None
            
            for metric in system_metrics:
                if metric.metric_type == MetricType.CPU_USAGE:
                    cpu_usage = metric.value
                elif metric.metric_type == MetricType.MEMORY_USAGE:
                    try:
                        memory = await asyncio.to_thread(psutil.virtual_memory)
                        memory_usage_mb = (memory.total * metric.value / 100) / (1024 * 1024)
                    except:
                        pass
                elif metric.metric_type == MetricType.GPU_USAGE:
                    gpu_usage = metric.value
                elif metric.metric_type == MetricType.GPU_MEMORY_USAGE:
                    if metric.tags and "gpu_memory_total_mb" in metric.tags:
                        total_gpu_mem = float(metric.tags["gpu_memory_total_mb"])
                        gpu_memory_mb = (total_gpu_mem * metric.value / 100)
                elif metric.metric_type == MetricType.NETWORK_BYTES_SENT:
                    network_bytes_sent = metric.value
                elif metric.metric_type == MetricType.NETWORK_BYTES_RECV:
                    network_bytes_recv = metric.value
            
            # Get host name
            host = None
            if system_metrics:
                host = system_metrics[0].host
            
            return ModelResourceUsage(
                model_id=model_id,
                deployment_id=deployment_id,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                gpu_usage_percent=gpu_usage,
                gpu_memory_usage_mb=gpu_memory_mb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                avg_latency_ms=None,  # Will be set via service composition
                requests_per_second=None,  # Will be set via service composition
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                host=host,
                tags={"model_id": model_id, "deployment_id": deployment_id or "none"}
            )
            
        except Exception as e:
            logger.error(f"Error collecting model resource usage: {e}")
            raise
    
    async def store_model_resource_usage(self, resource_usage: ModelResourceUsage) -> str:
        """Store model resource usage metrics in database"""
        try:
            import uuid
            usage_db = ModelResourceUsageDB(
                id=str(uuid.uuid4()),
                model_id=resource_usage.model_id,
                deployment_id=resource_usage.deployment_id,
                cpu_usage_percent=resource_usage.cpu_usage_percent,
                memory_usage_mb=resource_usage.memory_usage_mb,
                gpu_usage_percent=resource_usage.gpu_usage_percent,
                gpu_memory_usage_mb=resource_usage.gpu_memory_usage_mb,
                network_bytes_sent=resource_usage.network_bytes_sent,
                network_bytes_recv=resource_usage.network_bytes_recv,
                avg_latency_ms=resource_usage.avg_latency_ms,
                requests_per_second=resource_usage.requests_per_second,
                timestamp=resource_usage.timestamp,
                time_window_start=resource_usage.time_window_start,
                time_window_end=resource_usage.time_window_end,
                host=resource_usage.host,
                tags=resource_usage.tags
            )
            
            async with get_session() as session:
                session.add(usage_db)
                await session.commit()
                logger.info(f"Stored resource usage for model {resource_usage.model_id}")
                return usage_db.id
                
        except Exception as e:
            logger.error(f"Error storing model resource usage: {e}")
            raise
    
    async def _collect_gpu_metrics(self, host: str) -> List[SystemHealthMetric]:
        """Collect GPU usage metrics if available"""
        metrics = []
        try:
            # Try to import pynvml for NVIDIA GPU monitoring
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                    
                    metrics.append(SystemHealthMetric(
                        component=SystemComponent.MODEL_SERVICE,
                        status=SystemStatus.OPERATIONAL if gpu_usage < 90 else SystemStatus.DEGRADED,
                        message=f"GPU {i} utilization: {gpu_usage}%",
                        metric_type=MetricType.GPU_USAGE,
                        value=float(gpu_usage),
                        unit="percent",
                        host=host,
                        tags={"gpu_index": str(i), "gpu_name": pynvml.nvmlDeviceGetName(handle).decode('utf-8')}
                    ))
                    
                    # GPU memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_percent = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
                    
                    metrics.append(SystemHealthMetric(
                        component=SystemComponent.MODEL_SERVICE,
                        status=SystemStatus.OPERATIONAL if mem_used_percent < 90 else SystemStatus.DEGRADED,
                        message=f"GPU {i} memory usage: {mem_used_percent:.1f}%",
                        metric_type=MetricType.GPU_MEMORY_USAGE,
                        value=float(mem_used_percent),
                        unit="percent",
                        host=host,
                        tags={"gpu_index": str(i), "gpu_memory_total_mb": str(mem_info.total // (1024 * 1024))}
                    ))
                
                pynvml.nvmlShutdown()
            except ImportError:
                # pynvml not available, try torch for GPU info
                try:
                    import torch
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
                            mem_used_percent = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0
                            
                            metrics.append(SystemHealthMetric(
                                component=SystemComponent.MODEL_SERVICE,
                                status=SystemStatus.OPERATIONAL if mem_used_percent < 90 else SystemStatus.DEGRADED,
                                message=f"GPU {i} memory usage: {mem_used_percent:.1f}%",
                                metric_type=MetricType.GPU_MEMORY_USAGE,
                                value=float(mem_used_percent),
                                unit="percent",
                                host=host,
                                tags={"gpu_index": str(i), "gpu_name": torch.cuda.get_device_name(i)}
                            ))
                except ImportError:
                    pass
        except Exception as e:
            logger.debug(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    async def start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        logger.info("Starting monitoring background tasks")
        
        # Schedule regular health checks
        asyncio.create_task(self._health_check_loop())
        
        # Schedule metrics aggregation
        asyncio.create_task(self._metrics_aggregation_loop())
        
        # Note: Alert checking will be started via service composition
    
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
                # Note: This will need access to PerformanceMonitoringService
                # For now, we'll skip the aggregation or handle via service composition
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
                await asyncio.sleep(300)

