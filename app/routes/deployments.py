"""
Deployment routes for model deployment management
Provides REST API endpoints for deploying, managing, and monitoring model deployments
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse

from app.schemas.model import (
    ModelDeploymentCreate,
    ModelDeploymentResponse,
    ModelDeploymentUpdate,
    DeploymentStatus
)
from app.services.deployment_service import deployment_service

router = APIRouter()


@router.post("/", response_model=ModelDeploymentResponse, status_code=status.HTTP_201_CREATED)
async def create_deployment(deployment_data: ModelDeploymentCreate):
    """
    Create a new model deployment
    
    Creates a BentoML service for the model and deploys it as a REST API endpoint.
    The model must be validated before it can be deployed.
    """
    success, message, deployment = await deployment_service.create_deployment(deployment_data)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return deployment


@router.get("/", response_model=List[ModelDeploymentResponse])
async def list_deployments(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    status: Optional[DeploymentStatus] = Query(None, description="Filter by deployment status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of deployments to return"),
    offset: int = Query(0, ge=0, description="Number of deployments to skip")
):
    """
    List deployments with optional filtering
    
    Returns a list of deployments, optionally filtered by model ID and/or status.
    """
    deployments = await deployment_service.list_deployments(
        model_id=model_id,
        status=status,
        limit=limit,
        offset=offset
    )
    
    return deployments


@router.get("/{deployment_id}", response_model=ModelDeploymentResponse)
async def get_deployment(deployment_id: str):
    """
    Get a specific deployment by ID
    
    Returns detailed information about a deployment including its configuration,
    status, and endpoint information.
    """
    deployment = await deployment_service.get_deployment(deployment_id)
    
    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )
    
    return deployment


@router.patch("/{deployment_id}", response_model=ModelDeploymentResponse)
async def update_deployment(deployment_id: str, update_data: ModelDeploymentUpdate):
    """
    Update a deployment
    
    Allows updating deployment name, description, configuration, and status.
    Changing status to STOPPED will undeploy the service, and changing back to ACTIVE will redeploy it.
    """
    success, message, deployment = await deployment_service.update_deployment(deployment_id, update_data)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return deployment


@router.put("/{deployment_id}", response_model=ModelDeploymentResponse)
async def update_deployment_put(deployment_id: str, update_data: ModelDeploymentUpdate):
    """
    Update a deployment (PUT method)
    
    Allows updating deployment name, description, configuration, and status.
    Changing status to STOPPED will undeploy the service, and changing back to ACTIVE will redeploy it.
    """
    success, message, deployment = await deployment_service.update_deployment(deployment_id, update_data)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return deployment


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deployment(deployment_id: str):
    """
    Delete a deployment
    
    Undeploys the service and removes the deployment record.
    If this is the only deployment for a model, the model status will be updated back to VALIDATED.
    """
    success, message = await deployment_service.delete_deployment(deployment_id)
    
    if not success:
        if "not found" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
    
    return None  # 204 No Content


@router.get("/{deployment_id}/status")
async def get_deployment_status(deployment_id: str):
    """
    Get detailed status of a deployment
    
    Returns comprehensive status information including deployment status,
    service health, endpoint availability, and framework information.
    """
    status_info = await deployment_service.get_deployment_status(deployment_id)
    
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )
    
    return status_info


@router.post("/{deployment_id}/test")
async def test_deployment(deployment_id: str, test_data: Dict[str, Any]):
    """
    Test a deployment with sample data
    
    Sends test data to the deployed model and returns the prediction results.
    The deployment must be in ACTIVE status to accept test requests.
    
    Expected input format:
    ```
    {
        "data": [...] or {...}
    }
    ```
    """
    success, message, result = await deployment_service.test_deployment(deployment_id, test_data)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return result


@router.get("/{deployment_id}/metrics")
async def get_deployment_metrics(deployment_id: str):
    """
    Get metrics for a deployment
    
    Returns performance metrics including request counts, latency,
    error rates, and uptime information.
    """
    metrics = await deployment_service.get_deployment_metrics(deployment_id)
    
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )
    
    return metrics


@router.post("/{deployment_id}/start")
async def start_deployment(deployment_id: str):
    """
    Start a stopped deployment
    
    Redeploys the service and changes status to ACTIVE.
    """
    success, message = await deployment_service.start_deployment(deployment_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Deployment started successfully"}
    )


@router.post("/{deployment_id}/stop")
async def stop_deployment(deployment_id: str):
    """
    Stop an active deployment
    
    Undeploys the service and changes status to STOPPED.
    The deployment record is preserved and can be restarted later.
    """
    success, message = await deployment_service.stop_deployment(deployment_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Deployment stopped successfully"}
    )


@router.post("/{deployment_id}/scale")
async def scale_deployment(deployment_id: str, scale_data: Dict[str, Any]):
    """
    Scale a deployment
    
    Updates the scaling configuration for the deployment.
    """
    success, message = await deployment_service.scale_deployment(deployment_id, scale_data)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Deployment scaled successfully"}
    ) 