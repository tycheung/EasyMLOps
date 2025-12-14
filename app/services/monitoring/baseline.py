"""
Model baseline service
Handles model baseline creation, storage, and retrieval
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, desc

from app.database import get_session
from app.models.monitoring import ModelBaselineDB
from app.schemas.monitoring import ModelBaseline
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class ModelBaselineService(BaseMonitoringService):
    """Service for model baselines"""
    
    async def create_model_baseline(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        baseline_type: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None,
        is_production: bool = False,
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> ModelBaseline:
        """Create a performance baseline for a model version"""
        try:
            # Count prediction logs in the time window to get sample count
            from app.models.monitoring import PredictionLogDB
            from sqlalchemy import func
            
            async with get_session() as session:
                stmt = select(func.count(PredictionLogDB.id)).where(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time,
                    PredictionLogDB.success == True
                )
                if deployment_id:
                    stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                result = await session.execute(stmt)
                sample_count = result.scalar() or 0
                
                # Calculate average latency from logs
                latency_stmt = select(func.avg(PredictionLogDB.latency_ms)).where(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time,
                    PredictionLogDB.success == True
                )
                if deployment_id:
                    latency_stmt = latency_stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                latency_result = await session.execute(latency_stmt)
                avg_latency = latency_result.scalar()
            
            # Note: get_model_performance_metrics, calculate_classification_metrics,
            # calculate_regression_metrics, and calculate_confidence_metrics will be
            # accessed via service composition
            # For now, we'll create a baseline with calculated values
            
            # Build baseline
            baseline = ModelBaseline(
                model_id=model_id,
                model_name=model_name,
                model_version=model_version,
                deployment_id=deployment_id,
                baseline_type=baseline_type,
                baseline_p50_latency_ms=None,  # Will be set via service composition
                baseline_p95_latency_ms=None,
                baseline_p99_latency_ms=None,
                baseline_avg_latency_ms=float(avg_latency) if avg_latency else None,
                baseline_sample_count=sample_count,
                baseline_time_window_start=start_time,
                baseline_time_window_end=end_time,
                is_production=is_production,
                description=description,
                created_by=created_by
            )
            
            # Deactivate other baselines for the same model name
            await self._deactivate_other_baselines(model_name, baseline_type)
            
            # Store baseline
            stored_id = await self.store_model_baseline(baseline)
            baseline.id = stored_id
            
            logger.info(f"Created baseline for model {model_name} version {model_version}")
            return baseline
            
        except Exception as e:
            logger.error(f"Error creating model baseline: {e}", exc_info=True)
            raise
    
    async def _deactivate_other_baselines(self, model_name: str, baseline_type: str):
        """Deactivate other baselines of the same type for a model"""
        try:
            async with get_session() as session:
                stmt = select(ModelBaselineDB).where(
                    ModelBaselineDB.model_name == model_name,
                    ModelBaselineDB.baseline_type == baseline_type,
                    ModelBaselineDB.is_active == True
                )
                result = await session.execute(stmt)
                baselines = result.scalars().all()
                
                for baseline in baselines:
                    baseline.is_active = False
                    baseline.updated_at = datetime.utcnow()
                
                await session.commit()
        except Exception as e:
            logger.error(f"Error deactivating baselines: {e}")
    
    async def store_model_baseline(self, baseline: ModelBaseline) -> str:
        """Store model baseline in database"""
        try:
            baseline_db = ModelBaselineDB(
                id=str(uuid.uuid4()),
                model_id=baseline.model_id,
                model_name=baseline.model_name,
                model_version=baseline.model_version,
                deployment_id=baseline.deployment_id,
                baseline_type=baseline.baseline_type,
                baseline_accuracy=baseline.baseline_accuracy,
                baseline_precision=baseline.baseline_precision,
                baseline_recall=baseline.baseline_recall,
                baseline_f1=baseline.baseline_f1,
                baseline_auc_roc=baseline.baseline_auc_roc,
                baseline_mae=baseline.baseline_mae,
                baseline_mse=baseline.baseline_mse,
                baseline_rmse=baseline.baseline_rmse,
                baseline_r2=baseline.baseline_r2,
                baseline_p50_latency_ms=baseline.baseline_p50_latency_ms,
                baseline_p95_latency_ms=baseline.baseline_p95_latency_ms,
                baseline_p99_latency_ms=baseline.baseline_p99_latency_ms,
                baseline_avg_latency_ms=baseline.baseline_avg_latency_ms,
                baseline_avg_confidence=baseline.baseline_avg_confidence,
                baseline_low_confidence_rate=baseline.baseline_low_confidence_rate,
                baseline_sample_count=baseline.baseline_sample_count,
                baseline_time_window_start=baseline.baseline_time_window_start,
                baseline_time_window_end=baseline.baseline_time_window_end,
                is_active=baseline.is_active if hasattr(baseline, 'is_active') else True,
                is_production=baseline.is_production,
                created_by=baseline.created_by,
                description=baseline.description,
                additional_metrics=baseline.additional_metrics if hasattr(baseline, 'additional_metrics') else None
            )
            
            async with get_session() as session:
                session.add(baseline_db)
                await session.commit()
                logger.info(f"Stored baseline {baseline_db.id} for model {baseline.model_name}")
                return baseline_db.id
                
        except Exception as e:
            logger.error(f"Error storing model baseline: {e}")
            raise
    
    async def get_active_baseline(
        self,
        model_name: str,
        baseline_type: str
    ) -> Optional[ModelBaseline]:
        """Get the active baseline for a model"""
        try:
            async with get_session() as session:
                stmt = select(ModelBaselineDB).where(
                    ModelBaselineDB.model_name == model_name,
                    ModelBaselineDB.baseline_type == baseline_type,
                    ModelBaselineDB.is_active == True
                ).order_by(desc(ModelBaselineDB.created_at)).limit(1)
                
                result = await session.execute(stmt)
                baseline_db = result.scalar_one_or_none()
                
                if baseline_db:
                    return ModelBaseline(
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
                return None
                
        except Exception as e:
            logger.error(f"Error getting active baseline: {e}")
            return None

