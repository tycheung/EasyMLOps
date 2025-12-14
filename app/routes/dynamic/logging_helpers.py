"""
Prediction logging helpers
Handles logging of prediction requests and responses
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any

from app.models.monitoring import PredictionLogDB
from app.models.model import ModelDeployment

logger = logging.getLogger(__name__)


async def log_prediction(session, deployment_id: str, request_data: Dict[str, Any], 
                         response_data: Dict[str, Any], is_batch: bool = False, 
                         endpoint: str = "predict"):
    """Log prediction request and response"""
    try:
        # Get deployment to get model_id
        deployment = await session.get(ModelDeployment, deployment_id)
        if not deployment:
            logger.warning(f"Deployment {deployment_id} not found for logging")
            return
        
        prediction_log = PredictionLogDB(
            id=str(uuid.uuid4()),
            model_id=deployment.model_id,
            deployment_id=deployment_id,
            input_data=request_data,
            output_data=response_data,
            latency_ms=45.0,  # Mock latency - should be calculated in real implementation
            request_id=str(uuid.uuid4()),
            api_endpoint=endpoint,
            success=True,
            timestamp=datetime.utcnow(),
            is_batch=is_batch
        )
        
        session.add(prediction_log)
        await session.commit()
        
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        # Don't fail the prediction if logging fails

