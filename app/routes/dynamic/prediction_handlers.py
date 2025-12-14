"""
Prediction route handlers
Handles predict, predict_batch, and predict_proba endpoints
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.models.model import ModelDeployment
from app.schemas.model import DeploymentStatus
from app.services.schema_service import schema_service
from app.routes.dynamic.prediction_helpers import (
    make_prediction_call,
    make_batch_prediction_call,
    make_proba_prediction_call
)
from app.routes.dynamic.logging_helpers import log_prediction

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict/{deployment_id}")
async def predict(deployment_id: str, request: Request, session: AsyncSession = Depends(get_async_session)):
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
        prediction_response = await make_prediction_call(deployment, validated_data)
        
        # Log the prediction
        await log_prediction(session, deployment_id, request_data, prediction_response, False, "predict")
        
        # Return combined response
        return {
            **prediction_response,
            "validation": validation_result
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction endpoint for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/{deployment_id}/batch")
async def predict_batch(deployment_id: str, request: Request, session: AsyncSession = Depends(get_async_session)):
    """
    Batch prediction endpoint for deployed models
    
    Accepts multiple samples for prediction in a single request.
    
    Example request:
    ```
    {
        "data": [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0},
            {"feature1": 5.0, "feature2": 6.0}
        ]
    }
    ```
    
    Returns predictions for all samples along with validation results.
    """
    try:
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
        
        # Validate batch format
        if "data" not in request_data or not isinstance(request_data["data"], list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch data must be provided as a list under 'data' key"
            )
        
        batch_data = request_data["data"]
        if len(batch_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch data cannot be empty"
            )
        
        # Validate each item against schema if available
        validation_results = []
        validated_batch_data = []
        
        input_schema, _ = await schema_service.get_model_schemas(deployment.model_id)
        
        for i, item in enumerate(batch_data):
            if input_schema and isinstance(item, dict):
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
                    "validation_message": validation_message,
                    "schema_applied": True
                })
            else:
                validated_batch_data.append(item)
                validation_results.append({
                    "item_index": i,
                    "validation_performed": False,
                    "validation_message": "No schema defined or item not a dict",
                    "schema_applied": False
                })
        
        # Prepare validated request
        validated_request = {"data": validated_batch_data}
        
        # Make batch prediction call
        prediction_response = await make_batch_prediction_call(deployment, validated_request)
        
        # Log the prediction
        await log_prediction(session, deployment_id, request_data, prediction_response, True, "predict_batch")
        
        # Return combined response
        return {
            **prediction_response,
            "validation": {
                "validation_performed": any(v["validation_performed"] for v in validation_results),
                "schema_applied": any(v["schema_applied"] for v in validation_results),
                "item_validations": validation_results
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/predict/{deployment_id}/proba")
async def predict_proba(deployment_id: str, request: Request, session: AsyncSession = Depends(get_async_session)):
    """
    Probability prediction endpoint for deployed models
    
    Returns class probabilities for classification models.
    
    Example request:
    ```
    {
        "data": {"feature1": 1.0, "feature2": 2.0}
    }
    ```
    
    Returns probability distributions for each class.
    """
    try:
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
        
        # Check if model supports probability prediction
        if "predict_proba" not in deployment.endpoints:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Probability prediction is not supported for this model"
            )
        
        # Get request data
        try:
            request_data = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON data: {str(e)}"
            )
        
        # Validate data against schema if available (similar to predict endpoint)
        validation_result = None
        validated_data = request_data
        
        input_schema, _ = await schema_service.get_model_schemas(deployment.model_id)
        
        if input_schema and "data" not in request_data:
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
                "validation_message": "No schema defined or invalid format",
                "schema_applied": False
            }
        
        # Make probability prediction call
        prediction_response = await make_proba_prediction_call(deployment, validated_data)
        
        # Log the prediction
        await log_prediction(session, deployment_id, request_data, prediction_response, False, "predict_proba")
        
        # Return combined response
        return {
            **prediction_response,
            "validation": validation_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in probability prediction endpoint for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Probability prediction failed: {str(e)}"
        )

