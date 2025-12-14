"""
Comprehensive tests for dynamic routes
Tests dynamic prediction endpoints, schema validation, batch processing, and route management

This file has been refactored into:
- tests/test_routes/test_dynamic_creation.py: Route creation, model upload and deployment tests
- tests/test_routes/test_dynamic_execution.py: Route execution, prediction and inference tests

This file maintains backward compatibility by re-exporting all test classes.
"""

# Re-export all test classes for backward compatibility
from tests.test_routes.test_dynamic_creation import (
    TestDynamicRouteManager,
    TestModelUpload,
    TestModelListing,
    TestModelDetails,
    TestModelManagement,
    TestModelValidation,
    TestModelMetrics,
    TestErrorHandling as TestErrorHandlingCreation,
    TestIntegrationScenarios as TestIntegrationScenariosCreation,
)

from tests.test_routes.test_dynamic_execution import (
    TestPredictEndpoint,
    TestBatchPredictEndpoint,
    TestPredictProbaEndpoint,
    TestPredictionSchemaEndpoint,
    TestPredictionHelpers,
    TestPredictionLogging,
    TestDynamicRoutesIntegration,
    TestErrorHandling as TestErrorHandlingExecution,
    TestIntegrationScenarios as TestIntegrationScenariosExecution,
)

__all__ = [
    "TestDynamicRouteManager",
    "TestModelUpload",
    "TestModelListing",
    "TestModelDetails",
    "TestModelManagement",
    "TestModelValidation",
    "TestModelMetrics",
    "TestPredictEndpoint",
    "TestBatchPredictEndpoint",
    "TestPredictProbaEndpoint",
    "TestPredictionSchemaEndpoint",
    "TestPredictionHelpers",
    "TestPredictionLogging",
    "TestDynamicRoutesIntegration",
    "TestErrorHandlingCreation",
    "TestErrorHandlingExecution",
    "TestIntegrationScenariosCreation",
    "TestIntegrationScenariosExecution",
]
