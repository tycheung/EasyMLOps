"""
Schema route handler
Handles schema information endpoint
"""

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.models.model import ModelDeployment
from app.services.schema_service import schema_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/predict/{deployment_id}/schema")
async def get_prediction_schema(deployment_id: str, session: AsyncSession = Depends(get_async_session)):
    """
    Get prediction schema information for a deployed model
    
    Returns the input and output schemas, example data, and validation information.
    This helps users understand the expected format for prediction requests.
    """
    try:
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction schema for deployment {deployment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction schema: {str(e)}"
        )

