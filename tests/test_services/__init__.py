"""
Test services module
Exports all service test classes for backward compatibility
"""

from .test_monitoring_basic import (
    TestMonitoringService,
    TestServiceIntegration as TestServiceIntegrationMonitoring,
    TestServiceErrorHandling as TestServiceErrorHandlingMonitoring,
)

from .test_other_services import (
    TestBentoMLService,
    TestSchemaService,
    TestDeploymentService,
    TestServiceIntegration as TestServiceIntegrationOther,
    TestServiceErrorHandling as TestServiceErrorHandlingOther,
)

__all__ = [
    "TestMonitoringService",
    "TestBentoMLService",
    "TestSchemaService",
    "TestDeploymentService",
    "TestServiceIntegrationMonitoring",
    "TestServiceIntegrationOther",
    "TestServiceErrorHandlingMonitoring",
    "TestServiceErrorHandlingOther",
]
