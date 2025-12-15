"""
Comprehensive tests for monitoring routes
Tests all monitoring REST API endpoints including health checks, metrics, logs, and alerts

This file has been refactored into domain-specific test files in tests/test_routes/monitoring/:
- test_health.py: System health tests
- test_performance.py: Performance monitoring tests
- test_alerts.py: Alert management tests
- test_drift.py: Drift detection tests
- test_degradation.py: Performance degradation tests
- test_baseline.py: Model baseline tests
- test_ab_testing.py: A/B testing tests
- test_canary.py: Canary deployment tests
- test_explainability.py: Explainability tests
- test_data_quality.py: Data quality tests
- test_fairness.py: Fairness tests
- test_lifecycle.py: Lifecycle tests
- test_governance.py: Governance tests
- test_analytics.py: Analytics tests
- test_integration.py: Integration tests
- test_versioning.py: Model versioning tests
- test_audit.py: Audit log tests
- test_error_handling.py: Error handling tests

This file maintains backward compatibility by re-exporting all test classes.
"""

# Re-export all test classes for backward compatibility
from tests.test_routes.monitoring.test_health import (
    TestSystemHealth,
    TestSystemHealthAdvanced,
)
from tests.test_routes.monitoring.test_performance import (
    TestModelPerformance,
    TestPerformanceMonitoring,
)
from tests.test_routes.monitoring.test_alerts import (
    TestAlerts,
    TestAlertManagement,
    TestAlertRules,
    TestAlertManagementAdvanced,
    TestNotifications,
)
from tests.test_routes.monitoring.test_drift import TestDriftDetection
from tests.test_routes.monitoring.test_degradation import TestPerformanceDegradation
from tests.test_routes.monitoring.test_baseline import TestModelBaseline
from tests.test_routes.monitoring.test_ab_testing import (
    TestABTesting,
    TestABTestingAdvanced,
)
from tests.test_routes.monitoring.test_canary import (
    TestCanaryDeployment,
    TestCanaryAdvanced,
)
from tests.test_routes.monitoring.test_explainability import TestExplainability
from tests.test_routes.monitoring.test_data_quality import TestDataQuality
from tests.test_routes.monitoring.test_fairness import (
    TestFairness,
    TestFairnessAdvanced,
)
from tests.test_routes.monitoring.test_lifecycle import TestLifecycleAdvanced
from tests.test_routes.monitoring.test_governance import TestGovernanceAdvanced
from tests.test_routes.monitoring.test_analytics import TestAnalyticsAdvanced
from tests.test_routes.monitoring.test_integration import TestIntegrationAdvanced
from tests.test_routes.monitoring.test_versioning import TestModelVersioning
from tests.test_routes.monitoring.test_audit import TestAuditLogs
from tests.test_routes.monitoring.test_error_handling import TestMonitoringErrorHandling

__all__ = [
    "TestSystemHealth",
    "TestSystemHealthAdvanced",
    "TestModelPerformance",
    "TestPerformanceMonitoring",
    "TestAlerts",
    "TestAlertManagement",
    "TestAlertRules",
    "TestAlertManagementAdvanced",
    "TestNotifications",
    "TestDriftDetection",
    "TestPerformanceDegradation",
    "TestModelBaseline",
    "TestABTesting",
    "TestABTestingAdvanced",
    "TestCanaryDeployment",
    "TestCanaryAdvanced",
    "TestExplainability",
    "TestDataQuality",
    "TestFairness",
    "TestFairnessAdvanced",
    "TestLifecycleAdvanced",
    "TestGovernanceAdvanced",
    "TestAnalyticsAdvanced",
    "TestIntegrationAdvanced",
    "TestModelVersioning",
    "TestAuditLogs",
    "TestMonitoringErrorHandling",
]
