"""
Test routes module
Exports all route test classes for backward compatibility
"""

# Import from new dynamic subdirectory
from .dynamic.test_creation import *
from .dynamic.test_execution import *

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
]

