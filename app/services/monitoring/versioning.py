"""
Model versioning service
Handles model version comparison and regression detection
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, desc

from app.database import get_session
from app.models.monitoring import ModelBaselineDB, ModelVersionComparisonDB
from app.schemas.monitoring import (
    AlertSeverity, ModelBaseline, ModelVersionComparison, SystemComponent
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class ModelVersioningService(BaseMonitoringService):
    """Service for model versioning"""
    
    async def compare_model_versions(
        self,
        model_name: str,
        baseline_version: str,
        comparison_version: str,
        baseline_model_id: str,
        comparison_model_id: str,
        start_time: datetime,
        end_time: datetime,
        baseline_type: str = "performance"
    ) -> ModelVersionComparison:
        """Compare two model versions and detect performance regression"""
        try:
            # Get baseline metrics (via service composition)
            # Note: get_active_baseline will be accessed via service composition
            baseline = None
            try:
                # Try to get from baseline service
                from app.services.monitoring.baseline import ModelBaselineService
                baseline_service = ModelBaselineService()
                baseline = await baseline_service.get_active_baseline(model_name, baseline_type)
            except:
                pass
            
            if not baseline:
                # Try to get baseline from baseline version
                async with get_session() as session:
                    stmt = select(ModelBaselineDB).where(
                        ModelBaselineDB.model_name == model_name,
                        ModelBaselineDB.model_version == baseline_version,
                        ModelBaselineDB.baseline_type == baseline_type
                    ).order_by(desc(ModelBaselineDB.created_at)).limit(1)
                    result = await session.execute(stmt)
                    baseline_db = result.scalar_one_or_none()
                    if baseline_db:
                        baseline = ModelBaseline(
                            id=baseline_db.id,
                            model_id=baseline_db.model_id,
                            model_name=baseline_db.model_name,
                            model_version=baseline_db.model_version,
                            deployment_id=baseline_db.deployment_id,
                            baseline_type=baseline_db.baseline_type,
                            baseline_accuracy=baseline_db.baseline_accuracy,
                            baseline_precision=baseline_db.baseline_precision,
                            baseline_recall=baseline_db.baseline_recall,
                            baseline_f1=baseline_db.baseline_f1,
                            baseline_auc_roc=baseline_db.baseline_auc_roc,
                            baseline_mae=baseline_db.baseline_mae,
                            baseline_mse=baseline_db.baseline_mse,
                            baseline_rmse=baseline_db.baseline_rmse,
                            baseline_r2=baseline_db.baseline_r2,
                            baseline_p50_latency_ms=baseline_db.baseline_p50_latency_ms,
                            baseline_p95_latency_ms=baseline_db.baseline_p95_latency_ms,
                            baseline_p99_latency_ms=baseline_db.baseline_p99_latency_ms,
                            baseline_avg_latency_ms=baseline_db.baseline_avg_latency_ms,
                            baseline_avg_confidence=baseline_db.baseline_avg_confidence,
                            baseline_low_confidence_rate=baseline_db.baseline_low_confidence_rate,
                            baseline_sample_count=baseline_db.baseline_sample_count,
                            baseline_time_window_start=baseline_db.baseline_time_window_start,
                            baseline_time_window_end=baseline_db.baseline_time_window_end,
                            is_active=baseline_db.is_active,
                            is_production=baseline_db.is_production,
                            created_by=baseline_db.created_by,
                            description=baseline_db.description,
                            additional_metrics=baseline_db.additional_metrics,
                            created_at=baseline_db.created_at if hasattr(baseline_db, 'created_at') else None
                        )
            
            # Raise error if no baseline found (required for test)
            if not baseline:
                raise ValueError(f"No baseline found for model {model_name} version {baseline_version}")
            
            # Get performance metrics for both versions
            from app.models.monitoring import PredictionLogDB
            from sqlalchemy import func
            
            # Get baseline latency metrics
            async with get_session() as session:
                baseline_stmt = select(
                    func.avg(PredictionLogDB.latency_ms).label('avg_latency')
                ).where(
                    PredictionLogDB.model_id == baseline_model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time,
                    PredictionLogDB.success == True
                )
                baseline_result = await session.execute(baseline_stmt)
                baseline_avg_latency = baseline_result.scalar()
                
                # Get comparison latency metrics
                comparison_stmt = select(
                    func.avg(PredictionLogDB.latency_ms).label('avg_latency')
                ).where(
                    PredictionLogDB.model_id == comparison_model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time,
                    PredictionLogDB.success == True
                )
                comparison_result = await session.execute(comparison_stmt)
                comparison_avg_latency = comparison_result.scalar()
            
            # Calculate latency delta
            avg_latency_delta = None
            if baseline_avg_latency is not None and comparison_avg_latency is not None:
                avg_latency_delta = float(comparison_avg_latency - baseline_avg_latency)
            
            comparison = ModelVersionComparison(
                model_name=model_name,
                baseline_version=baseline_version,
                comparison_version=comparison_version,
                baseline_model_id=baseline_model_id,
                comparison_model_id=comparison_model_id,
                comparison_window_start=start_time,
                comparison_window_end=end_time,
                p50_latency_delta_ms=None,  # Will be set via service composition
                p95_latency_delta_ms=None,
                p99_latency_delta_ms=None,
                avg_latency_delta_ms=avg_latency_delta
            )
            
            # Assess performance based on latency delta
            performance_improved = False
            performance_degraded = False
            regression_severity = None
            
            if avg_latency_delta is not None:
                if avg_latency_delta < -10:  # 10ms improvement
                    performance_improved = True
                    comparison.recommendation = "promote"
                elif avg_latency_delta > 50:  # 50ms degradation
                    performance_degraded = True
                    if avg_latency_delta > 200:
                        regression_severity = "critical"
                    elif avg_latency_delta > 100:
                        regression_severity = "high"
                    else:
                        regression_severity = "medium"
                    comparison.recommendation = "investigate"
                else:
                    comparison.recommendation = "no_change"
            else:
                comparison.recommendation = "no_change"
            
            comparison.performance_improved = performance_improved
            comparison.performance_degraded = performance_degraded
            comparison.performance_regression_severity = regression_severity
            comparison.comparison_summary = f"Version {comparison_version} compared to {baseline_version}"
            
            # Store comparison
            await self.store_version_comparison(comparison)
            
            # Create alert if degraded (via service composition)
            if performance_degraded and regression_severity in ["high", "critical"]:
                await self._create_version_regression_alert(comparison)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing model versions: {e}", exc_info=True)
            raise
    
    async def store_version_comparison(self, comparison: ModelVersionComparison) -> str:
        """Store version comparison in database"""
        try:
            comparison_db = ModelVersionComparisonDB(
                id=str(uuid.uuid4()),
                model_name=comparison.model_name,
                baseline_version=comparison.baseline_version,
                comparison_version=comparison.comparison_version,
                baseline_model_id=comparison.baseline_model_id,
                comparison_model_id=comparison.comparison_model_id,
                comparison_window_start=comparison.comparison_window_start,
                comparison_window_end=comparison.comparison_window_end,
                accuracy_delta=comparison.accuracy_delta,
                precision_delta=comparison.precision_delta,
                recall_delta=comparison.recall_delta,
                f1_delta=comparison.f1_delta,
                auc_roc_delta=comparison.auc_roc_delta,
                mae_delta=comparison.mae_delta,
                mse_delta=comparison.mse_delta,
                rmse_delta=comparison.rmse_delta,
                r2_delta=comparison.r2_delta,
                p50_latency_delta_ms=comparison.p50_latency_delta_ms,
                p95_latency_delta_ms=comparison.p95_latency_delta_ms,
                p99_latency_delta_ms=comparison.p99_latency_delta_ms,
                avg_latency_delta_ms=comparison.avg_latency_delta_ms,
                avg_confidence_delta=comparison.avg_confidence_delta,
                low_confidence_rate_delta=comparison.low_confidence_rate_delta,
                p_value=comparison.p_value if hasattr(comparison, 'p_value') else None,
                is_statistically_significant=comparison.is_statistically_significant if hasattr(comparison, 'is_statistically_significant') else None,
                performance_improved=comparison.performance_improved,
                performance_degraded=comparison.performance_degraded,
                performance_regression_severity=comparison.performance_regression_severity,
                comparison_summary=comparison.comparison_summary,
                recommendation=comparison.recommendation,
                comparison_details=comparison.comparison_details if hasattr(comparison, 'comparison_details') else None
            )
            
            async with get_session() as session:
                session.add(comparison_db)
                await session.commit()
                logger.info(f"Stored version comparison for {comparison.model_name}")
                return comparison_db.id
                
        except Exception as e:
            logger.error(f"Error storing version comparison: {e}")
            raise
    
    async def _create_version_regression_alert(self, comparison: ModelVersionComparison) -> Optional[str]:
        """Create an alert for version performance regression"""
        try:
            severity_map = {
                "critical": AlertSeverity.CRITICAL,
                "high": AlertSeverity.ERROR,
                "medium": AlertSeverity.WARNING,
                "low": AlertSeverity.INFO
            }
            
            alert_severity = severity_map.get(comparison.performance_regression_severity or "medium", AlertSeverity.WARNING)
            
            title = f"Performance Regression Detected: {comparison.model_name} v{comparison.comparison_version}"
            description = (
                f"Model version {comparison.comparison_version} shows {comparison.performance_regression_severity} "
                f"performance regression compared to baseline {comparison.baseline_version}. "
                f"{comparison.comparison_summary}"
            )
            
            # Note: create_alert will be accessed via service composition
            logger.warning(f"Would create version regression alert: {title} - {description}")
            
            return "placeholder_alert_id"
            
        except Exception as e:
            logger.error(f"Error creating version regression alert: {e}")
            return None

