"""
Monitoring routes package
Combines all monitoring route modules into a single router
"""

from fastapi import APIRouter

from app.routes.monitoring.dashboard import router as dashboard_router
from app.routes.monitoring.alerts import router as alerts_router
from app.routes.monitoring.drift import router as drift_router
from app.routes.monitoring.degradation import router as degradation_router
from app.routes.monitoring.baseline import router as baseline_router
from app.routes.monitoring.testing import router as testing_router
from app.routes.monitoring.fairness import router as fairness_router
from app.routes.monitoring.explainability import router as explainability_router
from app.routes.monitoring.data_quality import router as data_quality_router
from app.routes.monitoring.lifecycle import router as lifecycle_router
from app.routes.monitoring.governance import router as governance_router
from app.routes.monitoring.analytics import router as analytics_router
from app.routes.monitoring.integration import router as integration_router
from app.routes.monitoring.audit import router as audit_router

# Create main router
router = APIRouter(tags=["monitoring"])

# Include all sub-routers
router.include_router(dashboard_router)
router.include_router(alerts_router)
router.include_router(drift_router)
router.include_router(degradation_router)
router.include_router(baseline_router)
router.include_router(testing_router)
router.include_router(fairness_router)
router.include_router(explainability_router)
router.include_router(data_quality_router)
router.include_router(lifecycle_router)
router.include_router(governance_router)
router.include_router(analytics_router)
router.include_router(integration_router)
router.include_router(audit_router)

__all__ = ["router"]

