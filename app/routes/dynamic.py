"""
Dynamic routes for deployed model prediction endpoints
Automatically generates and manages prediction routes for deployed models

This module has been refactored into submodules:
- app.routes.dynamic.dynamic_routes: Main router combining all handlers
- app.routes.dynamic.route_manager: Route manager class
- app.routes.dynamic.prediction_handlers: Prediction route handlers
- app.routes.dynamic.schema_handler: Schema information handler
- app.routes.dynamic.prediction_helpers: Prediction helper functions
- app.routes.dynamic.simulation_helpers: Framework simulation functions
- app.routes.dynamic.logging_helpers: Prediction logging functions

This file maintains backward compatibility by re-exporting router and DynamicRouteManager.
"""

# Re-export for backward compatibility
from app.routes.dynamic.dynamic_routes import router
from app.routes.dynamic.route_manager import DynamicRouteManager

# Global route manager instance for backward compatibility
route_manager = DynamicRouteManager()

__all__ = [
    "router",
    "DynamicRouteManager",
    "route_manager",
]
