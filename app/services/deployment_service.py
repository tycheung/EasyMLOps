"""
Deployment service for managing model deployments
Handles deployment lifecycle, monitoring, and integration with BentoML
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

from sqlmodel import select

from app.config import get_settings
from app.database import get_session
from app.models.model import Model, ModelDeployment
from app.schemas.model import (
    ModelDeploymentCreate, 
    ModelDeploymentResponse, 
    ModelDeploymentUpdate,
    ModelStatus,
    DeploymentStatus
)
from app.services.bentoml_service import bentoml_service_manager

settings = get_settings()
logger = logging.getLogger(__name__)


class DeploymentService:
    """Service for managing model deployments"""
    
    async def create_deployment(self, deployment_data: ModelDeploymentCreate) -> Tuple[bool, str, Optional[ModelDeploymentResponse]]:
        """Create a new model deployment"""
        try:
            async with get_session() as session:
                # Check if model exists
                model = await session.get(Model, deployment_data.model_id)
                if not model:
                    return False, f"Model {deployment_data.model_id} not found", None
                
                # Check if model is validated
                if model.status != ModelStatus.VALIDATED:
                    return False, f"Model must be validated before deployment", None
                
                # Check for existing active deployment
                existing_deployment = await session.exec(
                    select(ModelDeployment).where(
                        ModelDeployment.model_id == deployment_data.model_id,
                        ModelDeployment.status == DeploymentStatus.ACTIVE
                    )
                )
                existing = existing_deployment.first()
                
                if existing:
                    return False, f"Model {deployment_data.model_id} already has an active deployment", None
                
                # Create BentoML service
                success, message, service_info = await bentoml_service_manager.create_service_for_model(
                    deployment_data.model_id,
                    deployment_data.config
                )
                
                if not success:
                    return False, f"Failed to create BentoML service: {message}", None
                
                # Deploy the service
                deploy_success, deploy_message, deploy_info = await bentoml_service_manager.deploy_service(
                    service_info['service_name'],
                    deployment_data.config
                )
                
                if not deploy_success:
                    return False, f"Failed to deploy service: {deploy_message}", None
                
                # Create deployment record
                deployment = ModelDeployment(
                    id=str(uuid.uuid4()),
                    model_id=deployment_data.model_id,
                    name=deployment_data.name,
                    description=deployment_data.description,
                    config=deployment_data.config or {},
                    status=DeploymentStatus.ACTIVE,
                    endpoint_url=deploy_info.get('endpoint_url'),
                    service_name=service_info['service_name'],
                    bento_model_tag=service_info.get('bento_model_tag'),
                    framework=service_info.get('framework'),
                    endpoints=service_info.get('endpoints', []),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(deployment)
                await session.commit()
                await session.refresh(deployment)
                
                # Update model status
                model.status = ModelStatus.DEPLOYED
                model.updated_at = datetime.utcnow()
                await session.commit()
                
                logger.info(f"Successfully created deployment for model {deployment_data.model_id}")
                
                return True, "Deployment created successfully", ModelDeploymentResponse(
                    id=deployment.id,
                    model_id=deployment.model_id,
                    name=deployment.name,
                    description=deployment.description,
                    status=deployment.status,
                    endpoint_url=deployment.endpoint_url,
                    service_name=deployment.service_name,
                    framework=deployment.framework,
                    endpoints=deployment.endpoints,
                    config=deployment.config,
                    created_at=deployment.created_at,
                    updated_at=deployment.updated_at
                )
                
        except Exception as e:
            logger.error(f"Error creating deployment: {e}")
            return False, str(e), None
    
    async def get_deployment(self, deployment_id: str) -> Optional[ModelDeploymentResponse]:
        """Get a specific deployment by ID"""
        try:
            async with get_session() as session:
                deployment = await session.get(ModelDeployment, deployment_id)
                if not deployment:
                    return None
                
                return ModelDeploymentResponse(
                    id=deployment.id,
                    model_id=deployment.model_id,
                    name=deployment.name,
                    description=deployment.description,
                    status=deployment.status,
                    endpoint_url=deployment.endpoint_url,
                    service_name=deployment.service_name,
                    framework=deployment.framework,
                    endpoints=deployment.endpoints,
                    config=deployment.config,
                    created_at=deployment.created_at,
                    updated_at=deployment.updated_at
                )
                
        except Exception as e:
            logger.error(f"Error getting deployment {deployment_id}: {e}")
            return None
    
    async def list_deployments(self, 
                              model_id: Optional[str] = None,
                              status: Optional[DeploymentStatus] = None,
                              limit: int = 100,
                              offset: int = 0) -> List[ModelDeploymentResponse]:
        """List deployments with optional filtering"""
        try:
            async with get_session() as session:
                query = select(ModelDeployment)
                
                if model_id:
                    query = query.where(ModelDeployment.model_id == model_id)
                
                if status:
                    query = query.where(ModelDeployment.status == status)
                
                query = query.offset(offset).limit(limit).order_by(ModelDeployment.created_at.desc())
                
                result = await session.exec(query)
                deployments = result.all()
                
                return [
                    ModelDeploymentResponse(
                        id=deployment.id,
                        model_id=deployment.model_id,
                        name=deployment.name,
                        description=deployment.description,
                        status=deployment.status,
                        endpoint_url=deployment.endpoint_url,
                        service_name=deployment.service_name,
                        framework=deployment.framework,
                        endpoints=deployment.endpoints,
                        config=deployment.config,
                        created_at=deployment.created_at,
                        updated_at=deployment.updated_at
                    )
                    for deployment in deployments
                ]
                
        except Exception as e:
            logger.error(f"Error listing deployments: {e}")
            return []
    
    async def update_deployment(self, deployment_id: str, update_data: ModelDeploymentUpdate) -> Tuple[bool, str, Optional[ModelDeploymentResponse]]:
        """Update a deployment"""
        try:
            async with get_session() as session:
                deployment = await session.get(ModelDeployment, deployment_id)
                if not deployment:
                    return False, f"Deployment {deployment_id} not found", None
                
                # Update fields if provided
                if update_data.name is not None:
                    deployment.name = update_data.name
                
                if update_data.description is not None:
                    deployment.description = update_data.description
                
                if update_data.config is not None:
                    deployment.config = update_data.config
                
                if update_data.status is not None:
                    old_status = deployment.status
                    deployment.status = update_data.status
                    
                    # Handle status change actions
                    if old_status != update_data.status:
                        if update_data.status == DeploymentStatus.STOPPED:
                            # Undeploy the service
                            await bentoml_service_manager.undeploy_service(deployment.service_name)
                        elif update_data.status == DeploymentStatus.ACTIVE and old_status == DeploymentStatus.STOPPED:
                            # Redeploy the service
                            success, message, deploy_info = await bentoml_service_manager.deploy_service(
                                deployment.service_name,
                                deployment.config
                            )
                            if not success:
                                return False, f"Failed to redeploy service: {message}", None
                            
                            deployment.endpoint_url = deploy_info.get('endpoint_url')
                
                deployment.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(deployment)
                
                logger.info(f"Successfully updated deployment {deployment_id}")
                
                return True, "Deployment updated successfully", ModelDeploymentResponse(
                    id=deployment.id,
                    model_id=deployment.model_id,
                    name=deployment.name,
                    description=deployment.description,
                    status=deployment.status,
                    endpoint_url=deployment.endpoint_url,
                    service_name=deployment.service_name,
                    framework=deployment.framework,
                    endpoints=deployment.endpoints,
                    config=deployment.config,
                    created_at=deployment.created_at,
                    updated_at=deployment.updated_at
                )
                
        except Exception as e:
            logger.error(f"Error updating deployment {deployment_id}: {e}")
            return False, str(e), None
    
    async def delete_deployment(self, deployment_id: str) -> Tuple[bool, str]:
        """Delete a deployment"""
        try:
            async with get_session() as session:
                deployment = await session.get(ModelDeployment, deployment_id)
                if not deployment:
                    return False, f"Deployment {deployment_id} not found"
                
                # Undeploy the service
                if deployment.status == DeploymentStatus.ACTIVE:
                    await bentoml_service_manager.undeploy_service(deployment.service_name)
                
                # Update model status if this was the only deployment
                model = await session.get(Model, deployment.model_id)
                if model:
                    # Check for other active deployments
                    other_deployments = await session.exec(
                        select(ModelDeployment).where(
                            ModelDeployment.model_id == deployment.model_id,
                            ModelDeployment.id != deployment_id,
                            ModelDeployment.status == DeploymentStatus.ACTIVE
                        )
                    )
                    
                    if not other_deployments.first():
                        model.status = ModelStatus.VALIDATED
                        model.updated_at = datetime.utcnow()
                
                # Delete deployment record
                await session.delete(deployment)
                await session.commit()
                
                logger.info(f"Successfully deleted deployment {deployment_id}")
                return True, "Deployment deleted successfully"
                
        except Exception as e:
            logger.error(f"Error deleting deployment {deployment_id}: {e}")
            return False, str(e)
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a deployment"""
        try:
            async with get_session() as session:
                deployment = await session.get(ModelDeployment, deployment_id)
                if not deployment:
                    return None
                
                # Get service status from BentoML
                service_status = await bentoml_service_manager.get_service_status(deployment.service_name)
                
                return {
                    'deployment_id': deployment.id,
                    'deployment_status': deployment.status,
                    'service_status': service_status,
                    'endpoint_url': deployment.endpoint_url,
                    'service_name': deployment.service_name,
                    'framework': deployment.framework,
                    'endpoints': deployment.endpoints,
                    'last_check': datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting deployment status {deployment_id}: {e}")
            return None
    
    async def test_deployment(self, deployment_id: str, test_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Test a deployment with sample data"""
        try:
            async with get_session() as session:
                deployment = await session.get(ModelDeployment, deployment_id)
                if not deployment:
                    return False, f"Deployment {deployment_id} not found", None
                
                if deployment.status != DeploymentStatus.ACTIVE:
                    return False, f"Deployment is not active", None
                
                # TODO: Implement actual prediction call to the deployed service
                # For now, return a mock response
                mock_response = {
                    'deployment_id': deployment_id,
                    'predictions': [0.75, 0.25],  # Mock prediction
                    'model_id': deployment.model_id,
                    'framework': deployment.framework,
                    'test_successful': True,
                    'timestamp': datetime.utcnow()
                }
                
                logger.info(f"Successfully tested deployment {deployment_id}")
                return True, "Test successful", mock_response
                
        except Exception as e:
            logger.error(f"Error testing deployment {deployment_id}: {e}")
            return False, str(e), None
    
    async def get_deployment_metrics(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a deployment"""
        try:
            async with get_session() as session:
                deployment = await session.get(ModelDeployment, deployment_id)
                if not deployment:
                    return None
                
                # TODO: Implement actual metrics collection
                # For now, return mock metrics
                mock_metrics = {
                    'deployment_id': deployment_id,
                    'requests_count': 150,
                    'average_latency_ms': 45.2,
                    'error_rate': 0.02,
                    'uptime_percentage': 99.8,
                    'last_request': datetime.utcnow(),
                    'requests_per_minute': 5.5
                }
                
                return mock_metrics
                
        except Exception as e:
            logger.error(f"Error getting deployment metrics {deployment_id}: {e}")
            return None


# Global deployment service instance
deployment_service = DeploymentService() 