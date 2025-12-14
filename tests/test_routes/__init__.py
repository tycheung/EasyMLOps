"""
Test routes module
Exports all route test classes for backward compatibility
"""

from .test_dynamic_creation import *
from .test_dynamic_execution import *

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

