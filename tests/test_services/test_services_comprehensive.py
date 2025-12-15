"""
Comprehensive tests for all service layer components
Tests deployment, monitoring, schema, and BentoML services business logic

This file has been refactored into service-specific test files in tests/test_services/:
- test_deployment_service.py: Deployment service tests
- test_schema_service.py: Schema service tests
- test_bentoml_service.py: BentoML service tests
- test_integration_service.py: Service integration tests

This file maintains backward compatibility by re-exporting all test classes.
"""

# Re-export all test classes for backward compatibility
from tests.test_services.test_deployment_service import TestDeploymentService
from tests.test_services.test_schema_service import TestSchemaService
from tests.test_services.test_bentoml_service import TestBentoMLService
from tests.test_services.test_integration_service import TestServiceIntegration

__all__ = [
    "TestDeploymentService",
    "TestSchemaService",
    "TestBentoMLService",
    "TestServiceIntegration",
]
