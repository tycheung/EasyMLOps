"""
Comprehensive tests for main FastAPI application
Tests application setup, middleware, exception handlers, lifespan events, and core functionality

This file has been refactored into:
- tests/test_main/test_startup.py: Application initialization tests, configuration tests, database setup tests
- tests/test_main/test_runtime.py: Runtime behavior tests, request handling tests, error handling tests

This file maintains backward compatibility by re-exporting all test classes.
"""

# Re-export all test classes for backward compatibility
from tests.test_main.test_startup import (
    TestApplicationSetup,
    TestLifespanEvents,
    TestArgumentParsing,
    TestDatabaseConfiguration,
    TestApplicationConfiguration,
    TestMainFunction,
)

from tests.test_main.test_runtime import (
    TestRequestLoggingMiddleware,
    TestExceptionHandlers,
    TestHealthEndpoints,
    TestStaticFileServing,
    TestApplicationIntegration,
)

__all__ = [
    "TestApplicationSetup",
    "TestLifespanEvents",
    "TestArgumentParsing",
    "TestDatabaseConfiguration",
    "TestApplicationConfiguration",
    "TestMainFunction",
    "TestRequestLoggingMiddleware",
    "TestExceptionHandlers",
    "TestHealthEndpoints",
    "TestStaticFileServing",
    "TestApplicationIntegration",
]
