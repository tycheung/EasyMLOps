"""
Canary deployment service
Handles canary deployment creation, rollout management, health checks, and metrics
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import CanaryDeploymentDB, CanaryMetricsDB, PredictionLogDB
from app.schemas.monitoring import (
    AlertSeverity, CanaryDeployment, CanaryDeploymentStatus, CanaryMetrics,
    SystemComponent
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class CanaryDeploymentService(BaseMonitoringService):
    """Service for canary deployments"""
    
    async def create_canary_deployment(
        self,
        deployment_name: str,
        model_id: str,
        production_deployment_id: str,
        canary_deployment_id: str,
        target_traffic_percentage: float = 100.0,
        rollout_step_size: float = 10.0,
        rollout_step_duration_minutes: int = 60,
        max_error_rate_threshold: float = 5.0,
        max_latency_increase_pct: float = 50.0,
        min_health_check_duration_minutes: int = 5,
        health_check_window_minutes: int = 15,
        created_by: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> CanaryDeployment:
        """Create a canary deployment configuration"""
        try:
            # Calculate total steps
            total_steps = int((target_traffic_percentage / rollout_step_size) + 0.5) if rollout_step_size > 0 else 1
            
            canary = CanaryDeployment(
                deployment_name=deployment_name,
                model_id=model_id,
                production_deployment_id=production_deployment_id,
                canary_deployment_id=canary_deployment_id,
                current_traffic_percentage=0.0,
                target_traffic_percentage=target_traffic_percentage,
                rollout_step_size=rollout_step_size,
                rollout_step_duration_minutes=rollout_step_duration_minutes,
                max_error_rate_threshold=max_error_rate_threshold,
                max_latency_increase_pct=max_latency_increase_pct,
                min_health_check_duration_minutes=min_health_check_duration_minutes,
                health_check_window_minutes=health_check_window_minutes,
                status=CanaryDeploymentStatus.PENDING,
                current_step=0,
                total_steps=total_steps,
                created_by=created_by,
                config=config or {}
            )
            
            # Store canary deployment
            stored_id = await self.store_canary_deployment(canary)
            canary.id = stored_id
            
            logger.info(f"Created canary deployment: {deployment_name}")
            return canary
            
        except Exception as e:
            logger.error(f"Error creating canary deployment: {e}", exc_info=True)
            raise
    
    async def store_canary_deployment(self, canary: CanaryDeployment) -> str:
        """Store canary deployment in database"""
        try:
            canary_db = CanaryDeploymentDB(
                id=str(uuid.uuid4()),
                deployment_name=canary.deployment_name,
                model_id=canary.model_id,
                production_deployment_id=canary.production_deployment_id,
                canary_deployment_id=canary.canary_deployment_id,
                current_traffic_percentage=canary.current_traffic_percentage,
                target_traffic_percentage=canary.target_traffic_percentage,
                rollout_schedule=canary.rollout_schedule if hasattr(canary, 'rollout_schedule') else None,
                rollout_step_size=canary.rollout_step_size,
                rollout_step_duration_minutes=canary.rollout_step_duration_minutes,
                max_error_rate_threshold=canary.max_error_rate_threshold,
                max_latency_increase_pct=canary.max_latency_increase_pct,
                min_health_check_duration_minutes=canary.min_health_check_duration_minutes,
                health_check_window_minutes=canary.health_check_window_minutes,
                status=canary.status.value if hasattr(canary.status, 'value') else canary.status,
                started_at=canary.started_at if hasattr(canary, 'started_at') else None,
                completed_at=canary.completed_at if hasattr(canary, 'completed_at') else None,
                rolled_back_at=canary.rolled_back_at if hasattr(canary, 'rolled_back_at') else None,
                current_step=canary.current_step,
                total_steps=canary.total_steps,
                next_step_time=canary.next_step_time if hasattr(canary, 'next_step_time') else None,
                canary_error_rate=canary.canary_error_rate if hasattr(canary, 'canary_error_rate') else None,
                production_error_rate=canary.production_error_rate if hasattr(canary, 'production_error_rate') else None,
                canary_avg_latency_ms=canary.canary_avg_latency_ms if hasattr(canary, 'canary_avg_latency_ms') else None,
                production_avg_latency_ms=canary.production_avg_latency_ms if hasattr(canary, 'production_avg_latency_ms') else None,
                health_status=canary.health_status if hasattr(canary, 'health_status') else None,
                last_health_check=canary.last_health_check if hasattr(canary, 'last_health_check') else None,
                rollback_reason=canary.rollback_reason if hasattr(canary, 'rollback_reason') else None,
                rollback_triggered_by=canary.rollback_triggered_by if hasattr(canary, 'rollback_triggered_by') else None,
                config=canary.config,
                created_by=canary.created_by
            )
            
            async with get_session() as session:
                session.add(canary_db)
                await session.commit()
                logger.info(f"Stored canary deployment {canary_db.id}")
                return canary_db.id
                
        except Exception as e:
            logger.error(f"Error storing canary deployment: {e}")
            raise
    
    async def start_canary_rollout(self, canary_id: str) -> bool:
        """Start a canary deployment rollout"""
        try:
            async with get_session() as session:
                canary = await session.get(CanaryDeploymentDB, canary_id)
                if not canary:
                    raise ValueError(f"Canary deployment {canary_id} not found")
                
                if canary.status != "pending":
                    raise ValueError(f"Cannot start canary in status: {canary.status}")
                
                canary.status = "rolling_out"
                canary.started_at = datetime.utcnow()
                canary.current_traffic_percentage = canary.rollout_step_size or 10.0
                canary.current_step = 1
                canary.next_step_time = datetime.utcnow() + timedelta(minutes=canary.rollout_step_duration_minutes or 60)
                canary.updated_at = datetime.utcnow()
                
                await session.commit()
                logger.info(f"Started canary rollout {canary_id} at {canary.current_traffic_percentage}%")
                return True
                
        except Exception as e:
            logger.error(f"Error starting canary rollout: {e}")
            raise
    
    async def check_canary_health(self, canary_id: str) -> Tuple[bool, str, Optional[str]]:
        """Check canary deployment health and determine if rollout should continue or rollback
        
        Returns:
            (is_healthy, health_status, rollback_reason)
        """
        try:
            async with get_session() as session:
                canary = await session.get(CanaryDeploymentDB, canary_id)
                if not canary:
                    raise ValueError(f"Canary deployment {canary_id} not found")
                
                if canary.status != "rolling_out":
                    return True, "healthy", None
                
                # Check if minimum health check duration has passed
                if canary.started_at:
                    elapsed_minutes = (datetime.utcnow() - canary.started_at).total_seconds() / 60
                    if elapsed_minutes < (canary.min_health_check_duration_minutes or 5):
                        return True, "healthy", None  # Too early to check
                
                # Calculate metrics for health check window
                window_end = datetime.utcnow()
                window_start = window_end - timedelta(minutes=canary.health_check_window_minutes or 15)
                
                metrics = await self.calculate_canary_metrics(
                    canary_id=canary_id,
                    start_time=window_start,
                    end_time=window_end
                )
                
                # Update canary with latest metrics
                canary.canary_error_rate = metrics.canary_error_rate
                canary.production_error_rate = metrics.production_error_rate
                canary.canary_avg_latency_ms = metrics.canary_avg_latency_ms
                canary.production_avg_latency_ms = metrics.production_avg_latency_ms
                canary.last_health_check = datetime.utcnow()
                
                # Check error rate threshold
                if metrics.canary_error_rate is not None and canary.max_error_rate_threshold is not None:
                    if metrics.canary_error_rate > canary.max_error_rate_threshold:
                        canary.health_status = "unhealthy"
                        canary.updated_at = datetime.utcnow()
                        await session.commit()
                        return False, "unhealthy", f"Error rate {metrics.canary_error_rate:.2f}% exceeds threshold {canary.max_error_rate_threshold}%"
                
                # Check latency increase threshold
                if metrics.latency_increase_pct is not None and canary.max_latency_increase_pct is not None:
                    if metrics.latency_increase_pct > canary.max_latency_increase_pct:
                        canary.health_status = "degraded"
                        canary.updated_at = datetime.utcnow()
                        await session.commit()
                        return False, "degraded", f"Latency increase {metrics.latency_increase_pct:.2f}% exceeds threshold {canary.max_latency_increase_pct}%"
                
                # Determine health status from metrics
                is_healthy_value = getattr(metrics, 'is_healthy', metrics.health_check_passed)
                health_score_value = getattr(metrics, 'health_score', None)
                
                if is_healthy_value:
                    canary.health_status = "healthy"
                elif health_score_value is not None and health_score_value < 70:
                    canary.health_status = "unhealthy"
                else:
                    canary.health_status = "degraded"
                
                canary.updated_at = datetime.utcnow()
                await session.commit()
                
                return is_healthy_value, canary.health_status or "healthy", None
                
        except Exception as e:
            logger.error(f"Error checking canary health: {e}", exc_info=True)
            return False, "unhealthy", f"Health check error: {str(e)}"
    
    async def advance_canary_rollout(self, canary_id: str) -> bool:
        """Advance canary rollout to next step if health checks pass"""
        try:
            async with get_session() as session:
                canary = await session.get(CanaryDeploymentDB, canary_id)
                if not canary:
                    raise ValueError(f"Canary deployment {canary_id} not found")
                
                if canary.status != "rolling_out":
                    return False
                
                # Check if it's time for next step
                if canary.next_step_time and datetime.utcnow() < canary.next_step_time:
                    return False  # Not time yet
                
                # Perform health check
                is_healthy, health_status, rollback_reason = await self.check_canary_health(canary_id)
                
                if not is_healthy:
                    # Rollback due to health issues
                    await self.rollback_canary(canary_id, rollback_reason or "Health check failed")
                    return False
                
                # Advance to next step
                if canary.rollout_step_size:
                    new_percentage = min(
                        canary.current_traffic_percentage + canary.rollout_step_size,
                        canary.target_traffic_percentage
                    )
                else:
                    new_percentage = canary.target_traffic_percentage
                
                canary.current_traffic_percentage = new_percentage
                canary.current_step += 1
                
                # Check if rollout is complete
                if new_percentage >= canary.target_traffic_percentage:
                    canary.status = "completed"
                    canary.completed_at = datetime.utcnow()
                    canary.next_step_time = None
                else:
                    # Schedule next step
                    canary.next_step_time = datetime.utcnow() + timedelta(minutes=canary.rollout_step_duration_minutes or 60)
                
                canary.updated_at = datetime.utcnow()
                await session.commit()
                
                logger.info(f"Advanced canary {canary_id} to {new_percentage}% traffic (step {canary.current_step})")
                return True
                
        except Exception as e:
            logger.error(f"Error advancing canary rollout: {e}", exc_info=True)
            raise
    
    async def rollback_canary(self, canary_id: str, reason: str, triggered_by: str = "health_check") -> bool:
        """Rollback a canary deployment"""
        try:
            async with get_session() as session:
                canary = await session.get(CanaryDeploymentDB, canary_id)
                if not canary:
                    raise ValueError(f"Canary deployment {canary_id} not found")
                
                canary.status = "rolled_back"
                canary.rolled_back_at = datetime.utcnow()
                canary.rollback_reason = reason
                canary.rollback_triggered_by = triggered_by
                canary.current_traffic_percentage = 0.0
                canary.updated_at = datetime.utcnow()
                
                await session.commit()
                
                # Note: create_alert will be accessed via service composition
                logger.warning(f"Would create rollback alert for canary {canary_id}: {reason}")
                
                logger.warning(f"Rolled back canary deployment {canary_id}: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Error rolling back canary: {e}")
            raise
    
    async def calculate_canary_metrics(
        self,
        canary_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> CanaryMetrics:
        """Calculate and compare metrics for canary vs production"""
        try:
            async with get_session() as session:
                canary = await session.get(CanaryDeploymentDB, canary_id)
                if not canary:
                    raise ValueError(f"Canary deployment {canary_id} not found")
                
                # Note: get_model_performance_metrics and calculate_classification_metrics
                # will be accessed via service composition
                # For now, we'll create metrics with placeholder values
                
                # Get prediction logs for detailed metrics
                # Canary logs
                canary_stmt = select(PredictionLogDB).where(
                    PredictionLogDB.deployment_id == canary.canary_deployment_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time
                )
                canary_result = await session.execute(canary_stmt)
                canary_logs = canary_result.scalars().all()
                
                # Production logs
                prod_stmt = select(PredictionLogDB).where(
                    PredictionLogDB.deployment_id == canary.production_deployment_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time
                )
                prod_result = await session.execute(prod_stmt)
                prod_logs = prod_result.scalars().all()
                
                # Calculate metrics
                canary_total = len(canary_logs)
                canary_successful = sum(1 for log in canary_logs if log.success)
                canary_failed = canary_total - canary_successful
                canary_error_rate = (canary_failed / canary_total * 100.0) if canary_total > 0 else 0.0
                
                prod_total = len(prod_logs)
                prod_successful = sum(1 for log in prod_logs if log.success)
                prod_failed = prod_total - prod_successful
                prod_error_rate = (prod_failed / prod_total * 100.0) if prod_total > 0 else 0.0
                
                # Calculate deltas
                error_rate_delta = canary_error_rate - prod_error_rate
                latency_delta_ms = None
                latency_increase_pct = None
                
                # Calculate health score (0-100)
                health_score = 100.0
                if error_rate_delta > 0:
                    health_score -= min(error_rate_delta * 10, 50)  # Penalize error rate
                if latency_increase_pct and latency_increase_pct > 0:
                    health_score -= min(latency_increase_pct / 2, 30)  # Penalize latency
                
                health_score = max(0.0, min(100.0, health_score))
                is_healthy = health_score >= 70.0 and error_rate_delta <= 2.0
                
                # Determine health status string
                if is_healthy:
                    health_status_str = "healthy"
                elif health_score >= 50:
                    health_status_str = "degraded"
                else:
                    health_status_str = "unhealthy"
                
                metrics = CanaryMetrics(
                    canary_deployment_id=canary_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    canary_total_requests=canary_total,
                    canary_successful_requests=canary_successful,
                    canary_failed_requests=canary_failed,
                    canary_error_rate=canary_error_rate,
                    canary_avg_latency_ms=None,  # Will be set via service composition
                    production_total_requests=prod_total,
                    production_successful_requests=prod_successful,
                    production_failed_requests=prod_failed,
                    production_error_rate=prod_error_rate,
                    production_avg_latency_ms=None,  # Will be set via service composition
                    error_rate_delta=error_rate_delta,
                    latency_delta_ms=latency_delta_ms,
                    latency_increase_pct=latency_increase_pct,
                    health_status=health_status_str,  # Required field
                    health_check_passed=is_healthy  # Required field
                )
                
                # Store additional fields using object.__setattr__ to bypass Pydantic validation
                # These fields are needed for DB storage but not in the schema
                object.__setattr__(metrics, 'is_healthy', is_healthy)
                object.__setattr__(metrics, 'health_score', health_score)
                
                # Store metrics
                await self.store_canary_metrics(metrics)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating canary metrics: {e}", exc_info=True)
            raise
    
    async def store_canary_metrics(self, metrics: CanaryMetrics) -> str:
        """Store canary metrics in database"""
        try:
            metrics_db = CanaryMetricsDB(
                id=str(uuid.uuid4()),
                canary_deployment_id=metrics.canary_deployment_id,
                time_window_start=metrics.time_window_start,
                time_window_end=metrics.time_window_end,
                canary_total_requests=metrics.canary_total_requests,
                canary_successful_requests=metrics.canary_successful_requests,
                canary_failed_requests=metrics.canary_failed_requests,
                canary_error_rate=metrics.canary_error_rate,
                canary_avg_latency_ms=metrics.canary_avg_latency_ms,
                canary_p50_latency_ms=getattr(metrics, 'canary_p50_latency_ms', None),
                canary_p95_latency_ms=getattr(metrics, 'canary_p95_latency_ms', None),
                canary_p99_latency_ms=getattr(metrics, 'canary_p99_latency_ms', None),
                production_total_requests=metrics.production_total_requests,
                production_successful_requests=metrics.production_successful_requests,
                production_failed_requests=metrics.production_failed_requests,
                production_error_rate=metrics.production_error_rate,
                production_avg_latency_ms=metrics.production_avg_latency_ms,
                production_p50_latency_ms=getattr(metrics, 'production_p50_latency_ms', None),
                production_p95_latency_ms=getattr(metrics, 'production_p95_latency_ms', None),
                production_p99_latency_ms=getattr(metrics, 'production_p99_latency_ms', None),
                error_rate_delta=metrics.error_rate_delta,
                latency_delta_ms=metrics.latency_delta_ms,
                latency_increase_pct=metrics.latency_increase_pct,
                canary_accuracy=getattr(metrics, 'canary_accuracy', None),
                production_accuracy=getattr(metrics, 'production_accuracy', None),
                accuracy_delta=getattr(metrics, 'accuracy_delta', None),
                is_healthy=getattr(metrics, 'is_healthy', getattr(metrics, 'health_check_passed', False)),
                health_score=getattr(metrics, 'health_score', None)
                # Note: health_status and health_check_passed are schema-only fields, not in DB model
            )
            
            async with get_session() as session:
                session.add(metrics_db)
                await session.commit()
                return metrics_db.id
                
        except Exception as e:
            logger.error(f"Error storing canary metrics: {e}")
            raise

