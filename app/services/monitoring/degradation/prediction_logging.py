"""
Prediction logging with ground truth
Handles logging predictions with ground truth for performance evaluation
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.database import get_session
from app.models.monitoring import PredictionLogDB

logger = logging.getLogger(__name__)


async def log_prediction_with_ground_truth(
    model_id: str,
    deployment_id: Optional[str],
    input_data: Dict[str, Any],
    output_data: Any,
    ground_truth: Any,
    latency_ms: float,
    api_endpoint: str,
    success: bool = True,
    error_message: Optional[str] = None,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None
) -> str:
    """Log prediction with ground truth for performance evaluation"""
    try:
        log_entry = PredictionLogDB(
            id=str(uuid.uuid4()),
            model_id=model_id,
            deployment_id=deployment_id,
            request_id=str(uuid.uuid4()),
            input_data=input_data,
            output_data=output_data,
            ground_truth=ground_truth,
            ground_truth_timestamp=datetime.utcnow(),
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
            logger.info(f"Logged prediction with ground truth for model {model_id}")
            return log_entry.id
            
    except Exception as e:
        logger.error(f"Error logging prediction with ground truth: {e}")
        raise

