"""
BentoML Service Manager
Core service lifecycle management: creation, deployment, status, and listing
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import aiofiles.os as aios

from app.schemas.model import ModelFramework
from app.models.model import Model
from app.database import get_session
from app.services.bentoml.builders import ServiceBuilder

logger = logging.getLogger(__name__)

# Lazy import for BentoML
try:
    from bentoml import Service
    BENTOML_AVAILABLE = True
except Exception:
    Service = None
    BENTOML_AVAILABLE = False


class BentoMLServiceManager:
    """Manages BentoML services for dynamic model serving"""
    
    def __init__(self):
        self.active_services: Dict[str, Service] = {}
        self.model_cache: Dict[str, Any] = {}
        self.service_builder = ServiceBuilder()
        
    async def create_service_for_model(self, model_id: str, deployment_config: Dict[str, Any] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Create a BentoML service for a specific model"""
        try:
            if not BENTOML_AVAILABLE:
                logger.warning("BentoML not available - service creation disabled")
                return False, "BentoML not available", {}
            
            # Get model from database
            async with get_session() as session:
                model = await session.get(Model, model_id)
                if not model:
                    return False, f"Model {model_id} not found", {}
            
            # Load and validate the model
            model_path = model.file_path
            if not await aios.path.exists(model_path):
                return False, f"Model file not found: {model_path}", {}
            
            # Create the service based on framework
            service_name = f"model_service_{model_id}"
            
            if model.framework == ModelFramework.SKLEARN:
                success, message, service_info = await self.service_builder.create_sklearn_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.TENSORFLOW:
                success, message, service_info = await self.service_builder.create_tensorflow_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.PYTORCH:
                success, message, service_info = await self.service_builder.create_pytorch_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.XGBOOST:
                success, message, service_info = await self.service_builder.create_xgboost_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.LIGHTGBM:
                success, message, service_info = await self.service_builder.create_lightgbm_service(
                    model, service_name, deployment_config
                )
            else:
                return False, f"Framework {model.framework} not supported yet", {}
            
            if success:
                # Cache the service
                service_info['created_at'] = datetime.utcnow()
                service_info['model_id'] = model_id
                
                logger.info(f"Successfully created BentoML service for model {model_id}")
                
            return success, message, service_info
            
        except Exception as e:
            logger.error(f"Error creating BentoML service for model {model_id}: {e}")
            return False, str(e), {}
    
    async def deploy_service(self, service_name: str, config: Dict[str, Any] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Deploy a BentoML service"""
        try:
            # This would use BentoML's deployment capabilities
            # For now, we'll simulate the deployment
            
            deployment_info = {
                'status': 'deployed',
                'endpoint_url': f"http://localhost:3000/{service_name}",
                'deployment_id': str(uuid.uuid4()),
                'deployed_at': datetime.utcnow()
            }
            
            logger.info(f"Service {service_name} deployed successfully")
            return True, "Service deployed successfully", deployment_info
            
        except Exception as e:
            logger.error(f"Error deploying service {service_name}: {e}")
            return False, str(e), {}
    
    async def undeploy_service(self, service_name: str) -> Tuple[bool, str]:
        """Undeploy a BentoML service"""
        try:
            # Remove from active services if present
            if service_name in self.active_services:
                del self.active_services[service_name]
            
            logger.info(f"Service {service_name} undeployed successfully")
            return True, "Service undeployed successfully"
            
        except Exception as e:
            logger.error(f"Error undeploying service {service_name}: {e}")
            return False, str(e)
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get the status of a BentoML service"""
        try:
            # Check if service exists and is active
            if service_name in self.active_services:
                return {
                    'status': 'active',
                    'service_name': service_name,
                    'last_check': datetime.utcnow()
                }
            else:
                return {
                    'status': 'inactive',
                    'service_name': service_name,
                    'last_check': datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting service status for {service_name}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'service_name': service_name
            }
    
    async def list_services(self) -> List[Dict[str, Any]]:
        """List all active BentoML services"""
        try:
            services = []
            for service_name, service_info in self.active_services.items():
                services.append({
                    'service_name': service_name,
                    'model_id': service_info.get('model_id'),
                    'created_at': service_info.get('created_at'),
                    'status': 'active'
                })
            
            return services
            
        except Exception as e:
            logger.error(f"Error listing services: {e}")
            return []
    
    # Delegation methods for backward compatibility with tests
    def _generate_sklearn_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate sklearn service code - delegates to builder"""
        return self.service_builder.generate_sklearn_service_code(model, bento_model_tag, config)
    
    def _generate_tensorflow_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate TensorFlow service code - delegates to builder"""
        return self.service_builder.generate_tensorflow_service_code(model, bento_model_tag, config)
    
    def _generate_pytorch_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate PyTorch service code - delegates to builder"""
        return self.service_builder.generate_pytorch_service_code(model, bento_model_tag, config)
    
    def _generate_xgboost_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate XGBoost service code - delegates to builder"""
        return self.service_builder.generate_xgboost_service_code(model, bento_model_tag, config)
    
    def _generate_lightgbm_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate LightGBM service code - delegates to builder"""
        return self.service_builder.generate_lightgbm_service_code(model, bento_model_tag, config)

