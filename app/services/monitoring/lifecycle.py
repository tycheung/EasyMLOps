"""
Model lifecycle service
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import select, and_, desc

from app.database import get_session
from app.models.monitoring import (
    RetrainingJobDB, RetrainingTriggerConfigDB, ModelCardDB
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class ModelLifecycleService(BaseMonitoringService):
    """Service for model lifecycle"""
    
    async def configure_retraining_trigger(
        self,
        model_id: str,
        trigger_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Configure retraining trigger for a model"""
        try:
            trigger_id = str(uuid.uuid4())
            trigger_db = RetrainingTriggerConfigDB(
                id=trigger_id,
                model_id=model_id,
                trigger_type=trigger_type,
                is_enabled=config.get("is_enabled", True),
                performance_threshold=config.get("performance_threshold"),
                performance_metric=config.get("performance_metric"),
                degradation_window_hours=config.get("degradation_window_hours"),
                drift_threshold=config.get("drift_threshold"),
                drift_type=config.get("drift_type"),
                retraining_interval_days=config.get("retraining_interval_days"),
                last_retraining_date=config.get("last_retraining_date"),
                min_data_samples=config.get("min_data_samples"),
                data_window_days=config.get("data_window_days"),
                auto_replace_on_improvement=config.get("auto_replace_on_improvement", False),
                min_improvement_threshold=config.get("min_improvement_threshold"),
                created_by=config.get("created_by")
            )
            
            async with get_session() as session:
                session.add(trigger_db)
                await session.commit()
                logger.info(f"Configured retraining trigger {trigger_type} for model {model_id}")
                return trigger_id
        except Exception as e:
            logger.error(f"Error configuring retraining trigger: {e}", exc_info=True)
            raise
    
    async def create_retraining_job(
        self,
        model_id: str,
        trigger_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Create a retraining job"""
        try:
            job_id = str(uuid.uuid4())
            job_db = RetrainingJobDB(
                id=job_id,
                model_id=model_id,
                base_model_id=config.get("base_model_id"),
                job_name=config.get("job_name", f"Retraining job for {model_id}"),
                description=config.get("description"),
                trigger_type=trigger_type,
                trigger_reason=config.get("trigger_reason"),
                triggered_by=config.get("triggered_by"),
                training_config=config.get("training_config", {}),
                data_source=config.get("data_source"),
                data_window_start=config.get("data_window_start"),
                data_window_end=config.get("data_window_end"),
                status="pending",
                scheduled_at=config.get("scheduled_at"),
                auto_replace_enabled=config.get("auto_replace_enabled", False)
            )
            
            async with get_session() as session:
                session.add(job_db)
                await session.commit()
                logger.info(f"Created retraining job {job_id} for model {model_id}")
                return job_id
        except Exception as e:
            logger.error(f"Error creating retraining job: {e}", exc_info=True)
            raise
    
    async def generate_model_card(self, model_id: str) -> Dict[str, Any]:
        """Generate model card for a model"""
        try:
            # Get model information
            from app.models.model import Model
            from app.models.monitoring import ModelPerformanceMetricsDB
            
            async with get_session() as session:
                # Get model
                model = await session.get(Model, model_id)
                if not model:
                    raise ValueError(f"Model {model_id} not found")
                
                # Get recent performance metrics
                stmt = select(ModelPerformanceMetricsDB).where(
                    ModelPerformanceMetricsDB.model_id == model_id
                ).order_by(desc(ModelPerformanceMetricsDB.timestamp)).limit(1)
                result = await session.execute(stmt)
                latest_metrics = result.scalar_one_or_none()
                
                # Build model card content
                card_content = {
                    "model_details": {
                        "model_id": model_id,
                        "model_name": getattr(model, 'name', f"Model {model_id}"),
                        "framework": getattr(model, 'framework', 'unknown'),
                        "model_type": getattr(model, 'model_type', 'unknown'),
                        "version": getattr(model, 'version', '1.0.0')
                    },
                    "performance": {
                        "accuracy": latest_metrics.accuracy if latest_metrics else None,
                        "precision": latest_metrics.precision if latest_metrics else None,
                        "recall": latest_metrics.recall if latest_metrics else None,
                        "f1_score": latest_metrics.f1_score if latest_metrics else None
                    } if latest_metrics else {},
                    "training_info": {
                        "description": getattr(model, 'description', '')
                    },
                    "usage_guidelines": {
                        "recommended_use": "Production deployment",
                        "limitations": "Model performance may vary with input data distribution"
                    }
                }
                
                card = {
                    "model_id": model_id,
                    "model_version": getattr(model, 'version', '1.0.0'),
                    "card_content": card_content,
                    "card_version": "1.0",
                    "performance_metrics": {
                        "accuracy": latest_metrics.accuracy if latest_metrics else None,
                        "precision": latest_metrics.precision if latest_metrics else None,
                        "recall": latest_metrics.recall if latest_metrics else None,
                        "f1_score": latest_metrics.f1_score if latest_metrics else None
                    } if latest_metrics else {},
                    "training_data_info": {},
                    "training_parameters": {},
                    "evaluation_metrics": {}
                }
                
                return card
                
        except Exception as e:
            logger.error(f"Error generating model card: {e}", exc_info=True)
            raise
    
    async def get_model_card(self, model_id: str) -> Dict[str, Any]:
        """Get model card for a model"""
        try:
            async with get_session() as session:
                stmt = select(ModelCardDB).where(
                    ModelCardDB.model_id == model_id
                ).order_by(desc(ModelCardDB.created_at)).limit(1)
                
                result = await session.execute(stmt)
                card_db = result.scalar_one_or_none()
                
                if card_db:
                    return {
                        "id": card_db.id,
                        "model_id": card_db.model_id,
                        "model_version": card_db.model_version,
                        "card_content": card_db.card_content,
                        "card_version": card_db.card_version,
                        "performance_metrics": card_db.performance_metrics or {},
                        "training_data_info": card_db.training_data_info or {},
                        "training_date": card_db.training_date.isoformat() if card_db.training_date else None,
                        "training_duration_hours": card_db.training_duration_hours,
                        "training_parameters": card_db.training_parameters or {},
                        "model_architecture": card_db.model_architecture,
                        "model_limitations": card_db.model_limitations,
                        "usage_guidelines": card_db.usage_guidelines,
                        "ethical_considerations": card_db.ethical_considerations,
                        "evaluation_metrics": card_db.evaluation_metrics or {},
                        "evaluation_dataset_info": card_db.evaluation_dataset_info or {},
                        "evaluation_date": card_db.evaluation_date.isoformat() if card_db.evaluation_date else None,
                        "tags": card_db.tags or [],
                        "custom_fields": card_db.custom_fields or {},
                        "created_by": card_db.created_by,
                        "created_at": card_db.created_at.isoformat() if card_db.created_at else None,
                        "updated_at": card_db.updated_at.isoformat() if card_db.updated_at else None
                    }
                else:
                    # Generate a new card if none exists
                    return await self.generate_model_card(model_id)
                    
        except Exception as e:
            logger.error(f"Error getting model card: {e}", exc_info=True)
            raise
