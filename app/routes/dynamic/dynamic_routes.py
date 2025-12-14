"""
Dynamic routes for deployed model prediction endpoints
Automatically generates and manages prediction routes for deployed models

This module has been refactored into submodules:
- app.routes.dynamic.route_manager: Route manager class
- app.routes.dynamic.prediction_handlers: Prediction route handlers
- app.routes.dynamic.schema_handler: Schema information handler
- app.routes.dynamic.prediction_helpers: Prediction helper functions
- app.routes.dynamic.simulation_helpers: Framework simulation functions
- app.routes.dynamic.logging_helpers: Prediction logging functions

This file combines all route handlers into a single router.
"""

from fastapi import APIRouter

from app.routes.dynamic.prediction_handlers import router as prediction_router
from app.routes.dynamic.schema_handler import router as schema_router

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(prediction_router)
router.include_router(schema_router)

