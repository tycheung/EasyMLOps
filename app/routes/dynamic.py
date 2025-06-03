"""
Dynamic routes for deployed model prediction endpoints
Automatically generates and manages prediction routes for deployed models
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse
import httpx
import logging

from app.database import get_session
from app.models.model import ModelDeployment, ModelPrediction
from app.schemas.model import DeploymentStatus
from app.services.schema_service import schema_service

logger = logging.getLogger(__name__)

router = APIRouter()


class DynamicRouteManager:
    """Manages dynamic routes for deployed models"""
    
    def __init__(self):
        self.active_routes: Dict[str, Dict[str, Any]] = {}
    
    async def register_deployment_route(self, deployment: ModelDeployment):
        """Register a new route for a deployed model"""
        try:
            route_info = {
                'deployment_id': deployment.id,
                'model_id': deployment.model_id,
                'service_name': deployment.service_name,
                'endpoint_url': deployment.endpoint_url,
                'framework': deployment.framework,
                'endpoints': deployment.endpoints,
                'created_at': datetime.utcnow()
            }
            
            self.active_routes[deployment.id] = route_info
            logger.info(f"Registered dynamic route for deployment {deployment.id}")
            
        except Exception as e:
            logger.error(f"Error registering route for deployment {deployment.id}: {e}")
    
    async def unregister_deployment_route(self, deployment_id: str):
        """Unregister a route for a deployment"""
        try:
            if deployment_id in self.active_routes:
                del self.active_routes[deployment_id]
                logger.info(f"Unregistered dynamic route for deployment {deployment_id}")
                
        except Exception as e:
            logger.error(f"Error unregistering route for deployment {deployment_id}: {e}")
    
    async def get_route_info(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get route information for a deployment"""
        return self.active_routes.get(deployment_id)


# Global route manager instance
route_manager = DynamicRouteManager()


@router.post("/predict/{deployment_id}")
async def predict(deployment_id: str, request: Request):
    """
    Universal prediction endpoint for deployed models with schema validation
    
    This endpoint dynamically handles predictions for any deployed model.
    If the model has a defined input schema, the request will be validated
    against that schema before making the prediction.
    
    Example requests:
    
    For sklearn/xgboost/lightgbm models:
    ```
    {
        "data": [1.0, 2.0, 3.0, 4.0]  // Single sample
    }
    ```
    or
    ```
    {
        "data": [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0}
        ]
    }
    ```
    
    For tensorflow/pytorch models:
    ```
    {
        "data": [[1.0, 2.0, 3.0, 4.0]]  // Batch format
    }
    ```
    
    With defined schema (house price example):
    ```
    {
        "square_feet": 2000.5,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "age": 10,
        "location": "downtown"
    }
    ```
    """
    try:
        async with get_session() as session:
            # Get deployment info
            deployment = await session.get(ModelDeployment, deployment_id)
            if not deployment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Deployment {deployment_id} not found"
                )
            
            if deployment.status != DeploymentStatus.ACTIVE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Deployment {deployment_id} is not active"
                )
            
            # Get request data
            try:
                request_data = await request.json()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON data: {str(e)}"
                )
            
            # Validate data against schema if available
            validation_result = None
            validated_data = request_data
            
            # Check if request data has schema-formatted input (direct field values)
            # vs traditional format with "data" wrapper
            input_schema, _ = await schema_service.get_model_schemas(deployment.model_id)
            
            if input_schema and "data" not in request_data:
                # Direct schema-based input - validate it
                is_valid, validation_message, validated_fields = await schema_service.validate_prediction_data(
                    deployment.model_id, request_data
                )
                
                if not is_valid:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Input validation failed: {validation_message}"
                    )
                
                # Convert schema-validated data to model-expected format
                validated_data = {"data": validated_fields}
                validation_result = {
                    "validation_performed": True,
                    "validation_message": validation_message,
                    "schema_applied": True
                }
            elif input_schema and "data" in request_data:
                # Traditional format but schema exists - validate the inner data
                if isinstance(request_data["data"], dict):
                    is_valid, validation_message, validated_fields = await schema_service.validate_prediction_data(
                        deployment.model_id, request_data["data"]
                    )
                    
                    if not is_valid:
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Input validation failed: {validation_message}"
                        )
                    
                    validated_data = {"data": validated_fields}
                    validation_result = {
                        "validation_performed": True,
                        "validation_message": validation_message,
                        "schema_applied": True
                    }
                else:
                    validation_result = {
                        "validation_performed": False,
                        "validation_message": "Array input cannot be validated against field schema",
                        "schema_applied": False
                    }
            else:
                validation_result = {
                    "validation_performed": False,
                    "validation_message": "No schema defined for this model",
                    "schema_applied": False
                }
            
            # Make prediction call to the deployed service
            prediction_result = await _make_prediction_call(
                deployment, validated_data
            )
            
            # Add validation info to response
            prediction_result["validation"] = validation_result
            
            # Log the prediction
            await _log_prediction(session, deployment_id, request_data, prediction_result)
            
            return prediction_result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction endpoint for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/{deployment_id}/batch")
async def predict_batch(deployment_id: str, request: Request):
    """
    Batch prediction endpoint for deployed models with schema validation
    
    Handles multiple predictions in a single request for better efficiency.
    Validates each item in the batch against the model's schema if defined.
    
    Example request:
    ```
    {
        "data": [
            {"square_feet": 2000, "bedrooms": 3, "bathrooms": 2, "age": 10},
            {"square_feet": 1500, "bedrooms": 2, "bathrooms": 1, "age": 15}
        ]
    }
    ```
    """
    try:
        async with get_session() as session:
            # Get deployment info
            deployment = await session.get(ModelDeployment, deployment_id)
            if not deployment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Deployment {deployment_id} not found"
                )
            
            if deployment.status != DeploymentStatus.ACTIVE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Deployment {deployment_id} is not active"
                )
            
            # Get request data
            try:
                request_data = await request.json()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON data: {str(e)}"
                )
            
            # Validate batch data format
            if "data" not in request_data or not isinstance(request_data["data"], list):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Batch data must be provided as a list under 'data' key"
                )
            
            # Validate each item in batch against schema if available
            validation_results = []
            validated_batch_data = []
            
            input_schema, _ = await schema_service.get_model_schemas(deployment.model_id)
            
            if input_schema:
                for i, item in enumerate(request_data["data"]):
                    if isinstance(item, dict):
                        is_valid, validation_message, validated_fields = await schema_service.validate_prediction_data(
                            deployment.model_id, item
                        )
                        
                        if not is_valid:
                            raise HTTPException(
                                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"Validation failed for batch item {i}: {validation_message}"
                            )
                        
                        validated_batch_data.append(validated_fields)
                        validation_results.append({
                            "item_index": i,
                            "validation_performed": True,
                            "validation_message": validation_message
                        })
                    else:
                        validated_batch_data.append(item)
                        validation_results.append({
                            "item_index": i,
                            "validation_performed": False,
                            "validation_message": "Non-object items cannot be validated"
                        })
            else:
                validated_batch_data = request_data["data"]
                validation_results = [{
                    "validation_performed": False,
                    "validation_message": "No schema defined for this model"
                }]
            
            # Make batch prediction call
            batch_request = {"data": validated_batch_data}
            prediction_result = await _make_batch_prediction_call(
                deployment, batch_request
            )
            
            # Add validation info to response
            prediction_result["validation"] = {
                "batch_validation_performed": bool(input_schema),
                "item_validations": validation_results,
                "total_items": len(validated_batch_data)
            }
            
            # Log the batch prediction
            await _log_prediction(session, deployment_id, request_data, prediction_result, is_batch=True)
            
            return prediction_result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/predict/{deployment_id}/proba")
async def predict_proba(deployment_id: str, request: Request):
    """
    Prediction probability endpoint for classification models with schema validation
    
    Returns class probabilities instead of just predictions.
    Only available for classification models that support probability output.
    Input data is validated against the model's schema if defined.
    """
    try:
        async with get_session() as session:
            # Get deployment info
            deployment = await session.get(ModelDeployment, deployment_id)
            if not deployment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Deployment {deployment_id} not found"
                )
            
            if deployment.status != DeploymentStatus.ACTIVE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Deployment {deployment_id} is not active"
                )
            
            # Check if predict_proba is supported
            if 'predict_proba' not in deployment.endpoints:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Probability prediction not supported for this model"
                )
            
            # Get request data
            try:
                request_data = await request.json()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON data: {str(e)}"
                )
            
            # Validate data against schema (same logic as predict endpoint)
            validation_result = None
            validated_data = request_data
            
            input_schema, _ = await schema_service.get_model_schemas(deployment.model_id)
            
            if input_schema and "data" not in request_data:
                # Direct schema-based input
                is_valid, validation_message, validated_fields = await schema_service.validate_prediction_data(
                    deployment.model_id, request_data
                )
                
                if not is_valid:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Input validation failed: {validation_message}"
                    )
                
                validated_data = {"data": validated_fields}
                validation_result = {
                    "validation_performed": True,
                    "validation_message": validation_message,
                    "schema_applied": True
                }
            elif input_schema and "data" in request_data and isinstance(request_data["data"], dict):
                # Traditional format with inner data validation
                is_valid, validation_message, validated_fields = await schema_service.validate_prediction_data(
                    deployment.model_id, request_data["data"]
                )
                
                if not is_valid:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Input validation failed: {validation_message}"
                    )
                
                validated_data = {"data": validated_fields}
                validation_result = {
                    "validation_performed": True,
                    "validation_message": validation_message,
                    "schema_applied": True
                }
            else:
                validation_result = {
                    "validation_performed": False,
                    "validation_message": "No schema defined or array input",
                    "schema_applied": False
                }
            
            # Make probability prediction call
            prediction_result = await _make_proba_prediction_call(
                deployment, validated_data
            )
            
            # Add validation info to response
            prediction_result["validation"] = validation_result
            
            # Log the prediction
            await _log_prediction(session, deployment_id, request_data, prediction_result, endpoint="predict_proba")
            
            return prediction_result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in probability prediction endpoint for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Probability prediction failed: {str(e)}"
        )


@router.get("/predict/{deployment_id}/schema")
async def get_prediction_schema(deployment_id: str):
    """
    Get the input/output schema for a deployed model's prediction endpoint
    
    Returns the expected input format and output format for the model,
    including field types, validation rules, and example data.
    """
    try:
        async with get_session() as session:
            # Get deployment info
            deployment = await session.get(ModelDeployment, deployment_id)
            if not deployment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Deployment {deployment_id} not found"
                )
            
            # Get model schemas
            input_schema, output_schema = await schema_service.get_model_schemas(deployment.model_id)
            
            # Generate example data
            example_data = await schema_service.get_model_example_data(deployment.model_id)
            
            return {
                "deployment_id": deployment_id,
                "model_id": deployment.model_id,
                "framework": deployment.framework,
                "endpoints": deployment.endpoints,
                "input_schema": input_schema.dict() if input_schema else None,
                "output_schema": output_schema.dict() if output_schema else None,
                "example_input": example_data,
                "validation_enabled": bool(input_schema),
                "description": "Schema information for the deployed model prediction endpoint"
            }
            
    except Exception as e:
        logger.error(f"Error getting prediction schema for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction schema: {str(e)}"
        )


async def _make_prediction_call(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction call to the deployed service"""
    try:
        # In a real implementation, this would make an HTTP call to the BentoML service
        # For now, we'll simulate based on the framework
        
        if deployment.framework == "sklearn":
            return _simulate_sklearn_prediction(deployment, request_data)
        elif deployment.framework == "tensorflow":
            return _simulate_tensorflow_prediction(deployment, request_data)
        elif deployment.framework == "pytorch":
            return _simulate_pytorch_prediction(deployment, request_data)
        elif deployment.framework in ["xgboost", "lightgbm"]:
            return _simulate_boosting_prediction(deployment, request_data)
        else:
            return _simulate_generic_prediction(deployment, request_data)
            
    except Exception as e:
        logger.error(f"Error making prediction call: {e}")
        raise


async def _make_batch_prediction_call(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
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


async def _make_proba_prediction_call(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
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


def _simulate_sklearn_prediction(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate sklearn prediction"""
    data = request_data.get("data", [])
    
    # Mock prediction based on input
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            predictions = [0.75] * len(data)  # Mock predictions for dict input
        else:
            predictions = [0.65]  # Mock prediction for single sample
    elif isinstance(data, dict):
        predictions = [0.85]  # Mock prediction for single dict sample
    else:
        predictions = [0.5]  # Default mock prediction
    
    return {
        "predictions": predictions,
        "model_id": deployment.model_id,
        "deployment_id": deployment.id,
        "framework": deployment.framework,
        "model_type": "classification",
        "timestamp": datetime.utcnow().isoformat()
    }


def _simulate_tensorflow_prediction(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate TensorFlow prediction"""
    data = request_data.get("data", [])
    
    # Mock TensorFlow-style prediction
    predictions = [[0.2, 0.3, 0.5]]  # Mock softmax output
    
    return {
        "predictions": predictions,
        "model_id": deployment.model_id,
        "deployment_id": deployment.id,
        "framework": deployment.framework,
        "timestamp": datetime.utcnow().isoformat()
    }


def _simulate_pytorch_prediction(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate PyTorch prediction"""
    data = request_data.get("data", [])
    
    # Mock PyTorch-style prediction
    predictions = [0.85]  # Mock regression output
    
    return {
        "predictions": predictions,
        "model_id": deployment.model_id,
        "deployment_id": deployment.id,
        "framework": deployment.framework,
        "timestamp": datetime.utcnow().isoformat()
    }


def _simulate_boosting_prediction(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate XGBoost/LightGBM prediction"""
    data = request_data.get("data", [])
    
    # Mock boosting prediction
    predictions = [0.92]  # Mock high-confidence prediction
    
    return {
        "predictions": predictions,
        "model_id": deployment.model_id,
        "deployment_id": deployment.id,
        "framework": deployment.framework,
        "timestamp": datetime.utcnow().isoformat()
    }


def _simulate_generic_prediction(deployment: ModelDeployment, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate generic prediction"""
    return {
        "predictions": [0.5],
        "model_id": deployment.model_id,
        "deployment_id": deployment.id,
        "framework": deployment.framework,
        "timestamp": datetime.utcnow().isoformat()
    }


async def _log_prediction(session, deployment_id: str, request_data: Dict[str, Any], 
                         response_data: Dict[str, Any], is_batch: bool = False, 
                         endpoint: str = "predict"):
    """Log prediction request and response"""
    try:
        prediction_log = ModelPrediction(
            id=str(datetime.utcnow().timestamp()),
            deployment_id=deployment_id,
            request_data=request_data,
            response_data=response_data,
            endpoint=endpoint,
            is_batch=is_batch,
            request_size=len(str(request_data)),
            response_size=len(str(response_data)),
            latency_ms=45.0,  # Mock latency
            status="success",
            created_at=datetime.utcnow()
        )
        
        session.add(prediction_log)
        await session.commit()
        
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        # Don't fail the prediction if logging fails 