"""
BentoML service management for dynamic model serving
Handles creation, deployment, and lifecycle management of ML model services

This module has been refactored into submodules:
- app.services.bentoml.manager: Core service lifecycle management
- app.services.bentoml.builders: Framework-specific service creation and code generation
- app.services.bentoml.utils: Utility methods for testing and helpers

This file maintains backward compatibility by re-exporting all classes and instances.
"""

# Re-export for backward compatibility
from app.services.bentoml.manager import BentoMLServiceManager
from app.services.bentoml.builders import ServiceBuilder
from app.services.bentoml.utils import BentoMLUtils

# Create global service manager instance for backward compatibility
bentoml_service_manager = BentoMLServiceManager()

__all__ = [
    "BentoMLServiceManager",
    "ServiceBuilder",
    "BentoMLUtils",
    "bentoml_service_manager",
]
