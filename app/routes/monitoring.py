"""
Monitoring routes - Facade pattern
Re-exports router from monitoring sub-modules for backward compatibility

This module has been refactored into submodules:
- app.routes.monitoring.dashboard: Dashboard and health endpoints
- app.routes.monitoring.alerts: Alert management, rules, and notifications
- app.routes.monitoring.drift: Drift detection endpoints
- app.routes.monitoring.degradation: Performance degradation endpoints
- app.routes.monitoring.baseline: Model baseline and versioning endpoints
- app.routes.monitoring.testing: A/B testing and canary deployment endpoints
- app.routes.monitoring.fairness: Bias and fairness endpoints
- app.routes.monitoring.explainability: Explainability endpoints
- app.routes.monitoring.data_quality: Data quality endpoints
- app.routes.monitoring.lifecycle: Model lifecycle endpoints
- app.routes.monitoring.governance: Governance endpoints
- app.routes.monitoring.analytics: Analytics endpoints
- app.routes.monitoring.integration: Integration endpoints
- app.routes.monitoring.audit: Audit log endpoints
- app.routes.monitoring.middleware: Middleware utilities

This file maintains backward compatibility by re-exporting the router.
"""

# Re-export for backward compatibility
from app.routes.monitoring import router
from app.routes.monitoring.middleware import log_prediction_middleware

__all__ = [
    "router",
    "log_prediction_middleware",
]
