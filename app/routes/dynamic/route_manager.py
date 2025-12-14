"""
Dynamic route manager
Manages registration and tracking of dynamic routes for deployed models
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from app.models.model import ModelDeployment

logger = logging.getLogger(__name__)


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
                'service_name': deployment.deployment_name,
                'endpoint_url': deployment.deployment_url,
                'framework': getattr(deployment, 'framework', 'unknown'),
                'endpoints': ['predict', 'predict_proba'],
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

