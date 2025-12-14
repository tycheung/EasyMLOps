"""
Prediction logging service
Handles logging and retrieving prediction logs
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select

from app.database import get_session
from app.models.monitoring import PredictionLogDB
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class PredictionLoggingService(BaseMonitoringService):
    """Service for prediction logging"""
    
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
        ip_address: Optional[str] = None,
        inference_time_ms: Optional[float] = None,
        ttfb_ms: Optional[float] = None,
        batch_size: Optional[int] = None,
        is_batch: bool = False
    ) -> str:
        """Log individual prediction for performance monitoring with automatic confidence extraction"""
        try:
            # Extract confidence and uncertainty
            confidence_score = self._extract_confidence_score(output_data)
            confidence_scores = self._extract_confidence_scores(output_data)
            uncertainty_score = self._extract_uncertainty(output_data)
            interval_lower, interval_upper = self._extract_prediction_interval(output_data)
            
            log_entry = PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=model_id,
                deployment_id=deployment_id,
                request_id=str(uuid.uuid4()),
                input_data=input_data,
                output_data=output_data,
                latency_ms=latency_ms,
                inference_time_ms=inference_time_ms,
                ttfb_ms=ttfb_ms,
                batch_size=batch_size,
                is_batch=is_batch,
                timestamp=datetime.utcnow(),
                user_agent=user_agent,
                ip_address=ip_address,
                api_endpoint=api_endpoint,
                success=success,
                error_message=error_message,
                confidence_score=confidence_score,
                confidence_scores=confidence_scores,
                uncertainty_score=uncertainty_score,
                prediction_interval_lower=interval_lower,
                prediction_interval_upper=interval_upper
            )
            
            async with get_session() as session:
                session.add(log_entry)
                await session.commit()
                logger.info(f"Logged prediction for model {model_id} with confidence={confidence_score}")
                return log_entry.id
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            raise
    
    async def get_prediction_logs(self, model_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction logs"""
        try:
            async with get_session() as session:
                query = select(PredictionLogDB)
                if model_id:
                    query = query.where(PredictionLogDB.model_id == model_id)
                query = query.order_by(desc(PredictionLogDB.timestamp)).limit(limit)
                
                result = await session.execute(query)
                logs = result.scalars().all()
                
                return [
                    {
                        "id": log.id,
                        "model_id": log.model_id,
                        "deployment_id": log.deployment_id,
                        "timestamp": log.timestamp.isoformat(),
                        "latency_ms": log.latency_ms,
                        "success": log.success,
                        "error_message": log.error_message
                    }
                    for log in logs
                ]
        except Exception as e:
            logger.error(f"Error getting prediction logs: {e}")
            return []

