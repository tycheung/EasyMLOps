"""
Test main application module
Exports all main application test classes for backward compatibility
"""

from .test_startup import (
    TestApplicationSetup,
    TestLifespanEvents,
    TestArgumentParsing,
    TestDatabaseConfiguration,
    TestApplicationConfiguration,
    TestMainFunction,
)

from .test_runtime import (
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

