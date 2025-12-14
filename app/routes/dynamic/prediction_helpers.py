"""
Prediction helper functions
Handles making prediction calls to deployed services
"""

import logging
from datetime import datetime
from typing import Dict, Any

from app.models.model import ModelDeployment
from app.routes.dynamic.simulation_helpers import (
    simulate_sklearn_prediction,
    simulate_tensorflow_prediction,
    simulate_pytorch_prediction,
    simulate_boosting_prediction,
    simulate_generic_prediction
)

logger = logging.getLogger(__name__)


async def make_prediction_call(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction call to the deployed service"""
    try:
        # In a real implementation, this would make an HTTP call to the BentoML service
        # For now, we'll simulate based on the framework
        
        if deployment.framework == "sklearn":
            return simulate_sklearn_prediction(deployment, request_data)
        elif deployment.framework == "tensorflow":
            return simulate_tensorflow_prediction(deployment, request_data)
        elif deployment.framework == "pytorch":
            return simulate_pytorch_prediction(deployment, request_data)
        elif deployment.framework in ["xgboost", "lightgbm"]:
            return simulate_boosting_prediction(deployment, request_data)
        else:
            return simulate_generic_prediction(deployment, request_data)
            
    except Exception as e:
        logger.error(f"Error making prediction call: {e}")
        raise


async def make_batch_prediction_call(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make batch prediction call to the deployed service"""
    try:
        batch_data = request_data["data"]
        batch_size = len(batch_data)
        
        # Simulate batch predictions
        predictions = []
        for i in range(batch_size):
            # Simulate individual prediction
            pred = 0.5 + (i * 0.1) % 0.5  # Mock prediction
            predictions.append(pred)
        
        return {
            "predictions": predictions,
            "batch_size": batch_size,
            "model_id": deployment.model_id,
            "deployment_id": deployment.id,
            "framework": deployment.framework,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making batch prediction call: {e}")
        raise


async def make_proba_prediction_call(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make probability prediction call to the deployed service"""
    try:
        # Simulate probability predictions
        probabilities = [[0.3, 0.7], [0.8, 0.2]]  # Mock probabilities for 2 classes
        
        return {
            "probabilities": probabilities,
            "model_id": deployment.model_id,
            "deployment_id": deployment.id,
            "framework": deployment.framework,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making probability prediction call: {e}")
        raise

