"""
Monitoring and operations service facade for EasyMLOps platform
Facade pattern that delegates to domain-specific monitoring services
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.services.monitoring.performance import PerformanceMonitoringService
from app.services.monitoring.health import SystemHealthService
from app.services.monitoring.alerts import AlertService
from app.services.monitoring.alert_rules import AlertRulesService
from app.services.monitoring.notifications import NotificationService
from app.services.monitoring.alert_management import AlertManagementService
from app.services.monitoring.drift import DriftDetectionService
from app.services.monitoring.degradation import PerformanceDegradationService
from app.services.monitoring.baseline import ModelBaselineService
from app.services.monitoring.versioning import ModelVersioningService
from app.services.monitoring.ab_testing import ABTestingService
from app.services.monitoring.canary import CanaryDeploymentService
from app.services.monitoring.fairness import BiasFairnessService
from app.services.monitoring.explainability import ExplainabilityService
from app.services.monitoring.data_quality import DataQualityService
from app.services.monitoring.lifecycle import ModelLifecycleService
from app.services.monitoring.governance import GovernanceService
from app.services.monitoring.analytics import AnalyticsService
from app.services.monitoring.integration import IntegrationService
from app.services.monitoring.audit import AuditService
from app.services.monitoring.dashboard import DashboardService

from app.schemas.monitoring import (
    PredictionLog, ModelPerformanceMetrics, SystemHealthStatus, Alert,
    ModelDriftDetection, ModelPerformanceHistory, ModelBaseline,
    ModelVersionComparison, ABTest, ABTestMetrics, CanaryDeployment,
    CanaryMetrics, BiasFairnessMetrics, ModelExplanation, DashboardMetrics
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """Facade service that delegates to domain-specific monitoring services"""
    
    def __init__(self):
        self.start_time = time.time()
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 5.0,
            "avg_latency_ms": 1000.0
        }
        
        # Initialize all sub-services
        self.performance = PerformanceMonitoringService()
        self.health = SystemHealthService()
        self.alerts = AlertService()
        self.alert_rules = AlertRulesService()
        self.notifications = NotificationService()
        self.alert_management = AlertManagementService()
        self.drift = DriftDetectionService()
        self.degradation = PerformanceDegradationService()
        self.baseline = ModelBaselineService()
        self.versioning = ModelVersioningService()
        self.ab_testing = ABTestingService()
        self.canary = CanaryDeploymentService()
        self.fairness = BiasFairnessService()
        self.explainability = ExplainabilityService()
        self.data_quality = DataQualityService()
        self.lifecycle = ModelLifecycleService()
        self.governance = GovernanceService()
        self.analytics = AnalyticsService()
        self.integration = IntegrationService()
        self.audit = AuditService()
        self.dashboard = DashboardService()
    
    # Performance Monitoring - delegate to PerformanceMonitoringService
    async def log_prediction(self, *args, **kwargs) -> str:
        """Log individual prediction for performance monitoring"""
        return await self.performance.log_prediction(*args, **kwargs)
    
    async def get_model_performance_metrics(self, *args, **kwargs) -> ModelPerformanceMetrics:
        """Get model performance metrics"""
        return await self.performance.get_model_performance_metrics(*args, **kwargs)
    
    async def store_performance_metrics(self, *args, **kwargs) -> str:
        """Store performance metrics"""
        return await self.performance.store_performance_metrics(*args, **kwargs)
    
    async def get_prediction_logs(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get prediction logs"""
        return await self.performance.get_prediction_logs(*args, **kwargs)
    
    async def get_aggregated_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """Get aggregated metrics"""
        return await self.performance.get_aggregated_metrics(*args, **kwargs)
    
    async def get_deployment_summary(self, *args, **kwargs) -> Dict[str, Any]:
        """Get deployment summary"""
        return await self.performance.get_deployment_summary(*args, **kwargs)
    
    async def calculate_confidence_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """Calculate confidence metrics"""
        return await self.performance.calculate_confidence_metrics(*args, **kwargs)
    
    async def store_confidence_metrics(self, *args, **kwargs) -> str:
        """Store confidence metrics"""
        return await self.performance.store_confidence_metrics(*args, **kwargs)
    
    # System Health - delegate to SystemHealthService
    async def collect_system_health_metrics(self) -> List:
        """Collect system health metrics"""
        return await self.health.collect_system_health_metrics()
    
    async def store_health_metric(self, *args, **kwargs) -> str:
        """Store health metric"""
        return await self.health.store_health_metric(*args, **kwargs)
    
    async def get_system_health_status(self) -> SystemHealthStatus:
        """Get system health status"""
        return await self.health.get_system_health_status()
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check system health"""
        return await self.health.check_system_health()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health"""
        return await self.health.get_system_health()
    
    async def collect_model_resource_usage(self, *args, **kwargs) -> Dict[str, Any]:
        """Collect model resource usage"""
        return await self.health.collect_model_resource_usage(*args, **kwargs)
    
    async def store_model_resource_usage(self, *args, **kwargs) -> str:
        """Store model resource usage"""
        return await self.health.store_model_resource_usage(*args, **kwargs)
    
    async def check_bentoml_system_health(self) -> Tuple[bool, str]:
        """Check BentoML system health"""
        return await self.health.check_bentoml_system_health()
    
    async def start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        return await self.health.start_monitoring_tasks()
    
    # Alerts - delegate to AlertService
    async def check_and_create_alerts(self) -> List[Alert]:
        """Check and create alerts"""
        return await self.alerts.check_and_create_alerts()
    
    async def create_alert(self, *args, **kwargs) -> str:
        """Create alert"""
        return await self.alerts.create_alert(*args, **kwargs)
    
    async def get_alerts(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get alerts"""
        return await self.alerts.get_alerts(*args, **kwargs)
    
    async def resolve_alert(self, *args, **kwargs) -> bool:
        """Resolve alert"""
        return await self.alerts.resolve_alert(*args, **kwargs)
    
    async def acknowledge_alert(self, *args, **kwargs) -> bool:
        """Acknowledge alert"""
        return await self.alerts.acknowledge_alert(*args, **kwargs)
    
    # Alert Rules - delegate to AlertRulesService
    async def create_alert_rule(self, *args, **kwargs) -> str:
        """Create alert rule"""
        return await self.alert_rules.create_alert_rule(*args, **kwargs)
    
    async def store_alert_rule(self, *args, **kwargs) -> str:
        """Store alert rule"""
        return await self.alert_rules.store_alert_rule(*args, **kwargs)
    
    async def evaluate_alert_rule(self, *args, **kwargs) -> bool:
        """Evaluate alert rule"""
        return await self.alert_rules.evaluate_alert_rule(*args, **kwargs)
    
    # Notifications - delegate to NotificationService
    async def create_notification_channel(self, *args, **kwargs) -> str:
        """Create notification channel"""
        return await self.notifications.create_notification_channel(*args, **kwargs)
    
    async def store_notification_channel(self, *args, **kwargs) -> str:
        """Store notification channel"""
        return await self.notifications.store_notification_channel(*args, **kwargs)
    
    async def send_alert_notification(self, *args, **kwargs) -> bool:
        """Send alert notification"""
        return await self.notifications.send_alert_notification(*args, **kwargs)
    
    async def trigger_webhook(self, *args, **kwargs) -> bool:
        """Trigger webhook"""
        return await self.notifications.trigger_webhook(*args, **kwargs)
    
    # Alert Management - delegate to AlertManagementService
    async def group_alerts(self, *args, **kwargs) -> str:
        """Group alerts"""
        return await self.alert_management.group_alerts(*args, **kwargs)
    
    async def store_alert_group(self, *args, **kwargs) -> str:
        """Store alert group"""
        return await self.alert_management.store_alert_group(*args, **kwargs)
    
    async def create_alert_escalation(self, *args, **kwargs) -> str:
        """Create alert escalation"""
        return await self.alert_management.create_alert_escalation(*args, **kwargs)
    
    async def store_alert_escalation(self, *args, **kwargs) -> str:
        """Store alert escalation"""
        return await self.alert_management.store_alert_escalation(*args, **kwargs)
    
    async def check_and_escalate_alerts(self) -> List[str]:
        """Check and escalate alerts"""
        return await self.alert_management.check_and_escalate_alerts()
    
    # Drift Detection - delegate to DriftDetectionService
    async def detect_feature_drift(self, *args, **kwargs) -> ModelDriftDetection:
        """Detect feature drift"""
        return await self.drift.detect_feature_drift(*args, **kwargs)
    
    async def detect_data_drift(self, *args, **kwargs) -> ModelDriftDetection:
        """Detect data drift"""
        return await self.drift.detect_data_drift(*args, **kwargs)
    
    async def detect_prediction_drift(self, *args, **kwargs) -> ModelDriftDetection:
        """Detect prediction drift"""
        return await self.drift.detect_prediction_drift(*args, **kwargs)
    
    async def store_drift_detection(self, *args, **kwargs) -> str:
        """Store drift detection"""
        return await self.drift.store_drift_detection(*args, **kwargs)
    
    # Performance Degradation - delegate to PerformanceDegradationService
    async def log_prediction_with_ground_truth(self, *args, **kwargs) -> str:
        """Log prediction with ground truth"""
        return await self.degradation.log_prediction_with_ground_truth(*args, **kwargs)
    
    async def calculate_classification_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """Calculate classification metrics"""
        return await self.degradation.calculate_classification_metrics(*args, **kwargs)
    
    async def calculate_regression_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """Calculate regression metrics"""
        return await self.degradation.calculate_regression_metrics(*args, **kwargs)
    
    async def detect_performance_degradation(self, *args, **kwargs) -> ModelPerformanceHistory:
        """Detect performance degradation"""
        return await self.degradation.detect_performance_degradation(*args, **kwargs)
    
    async def store_performance_history(self, *args, **kwargs) -> str:
        """Store performance history"""
        return await self.degradation.store_performance_history(*args, **kwargs)
    
    # Model Baseline - delegate to ModelBaselineService
    async def create_model_baseline(self, *args, **kwargs) -> ModelBaseline:
        """Create model baseline"""
        return await self.baseline.create_model_baseline(*args, **kwargs)
    
    async def store_model_baseline(self, *args, **kwargs) -> str:
        """Store model baseline"""
        return await self.baseline.store_model_baseline(*args, **kwargs)
    
    async def get_active_baseline(self, *args, **kwargs) -> Optional[ModelBaseline]:
        """Get active baseline"""
        return await self.baseline.get_active_baseline(*args, **kwargs)
    
    # Model Versioning - delegate to ModelVersioningService
    async def compare_model_versions(self, *args, **kwargs) -> ModelVersionComparison:
        """Compare model versions"""
        return await self.versioning.compare_model_versions(*args, **kwargs)
    
    async def store_version_comparison(self, *args, **kwargs) -> str:
        """Store version comparison"""
        return await self.versioning.store_version_comparison(*args, **kwargs)
    
    # AB Testing - delegate to ABTestingService
    async def create_ab_test(self, *args, **kwargs) -> ABTest:
        """Create AB test"""
        return await self.ab_testing.create_ab_test(*args, **kwargs)
    
    async def store_ab_test(self, *args, **kwargs) -> str:
        """Store AB test"""
        return await self.ab_testing.store_ab_test(*args, **kwargs)
    
    async def assign_variant(self, *args, **kwargs) -> str:
        """Assign variant"""
        return await self.ab_testing.assign_variant(*args, **kwargs)
    
    async def start_ab_test(self, *args, **kwargs) -> bool:
        """Start AB test"""
        return await self.ab_testing.start_ab_test(*args, **kwargs)
    
    async def stop_ab_test(self, *args, **kwargs) -> bool:
        """Stop AB test"""
        return await self.ab_testing.stop_ab_test(*args, **kwargs)
    
    async def calculate_ab_test_metrics(self, *args, **kwargs) -> ABTestMetrics:
        """Calculate AB test metrics"""
        return await self.ab_testing.calculate_ab_test_metrics(*args, **kwargs)
    
    async def store_ab_test_metrics(self, *args, **kwargs) -> str:
        """Store AB test metrics"""
        return await self.ab_testing.store_ab_test_metrics(*args, **kwargs)
    
    # Canary Deployment - delegate to CanaryDeploymentService
    async def create_canary_deployment(self, *args, **kwargs) -> CanaryDeployment:
        """Create canary deployment"""
        return await self.canary.create_canary_deployment(*args, **kwargs)
    
    async def store_canary_deployment(self, *args, **kwargs) -> str:
        """Store canary deployment"""
        return await self.canary.store_canary_deployment(*args, **kwargs)
    
    async def start_canary_rollout(self, *args, **kwargs) -> bool:
        """Start canary rollout"""
        return await self.canary.start_canary_rollout(*args, **kwargs)
    
    async def check_canary_health(self, *args, **kwargs) -> Tuple[bool, str, Optional[str]]:
        """Check canary health"""
        return await self.canary.check_canary_health(*args, **kwargs)
    
    async def advance_canary_rollout(self, *args, **kwargs) -> bool:
        """Advance canary rollout"""
        return await self.canary.advance_canary_rollout(*args, **kwargs)
    
    async def rollback_canary(self, *args, **kwargs) -> bool:
        """Rollback canary"""
        return await self.canary.rollback_canary(*args, **kwargs)
    
    async def calculate_canary_metrics(self, *args, **kwargs) -> CanaryMetrics:
        """Calculate canary metrics"""
        return await self.canary.calculate_canary_metrics(*args, **kwargs)
    
    async def store_canary_metrics(self, *args, **kwargs) -> str:
        """Store canary metrics"""
        return await self.canary.store_canary_metrics(*args, **kwargs)
    
    # Bias and Fairness - delegate to BiasFairnessService
    async def configure_protected_attribute(self, *args, **kwargs) -> str:
        """Configure protected attribute"""
        return await self.fairness.configure_protected_attribute(*args, **kwargs)
    
    async def store_protected_attribute_config(self, *args, **kwargs) -> str:
        """Store protected attribute config"""
        return await self.fairness.store_protected_attribute_config(*args, **kwargs)
    
    async def calculate_fairness_metrics(self, *args, **kwargs) -> BiasFairnessMetrics:
        """Calculate fairness metrics"""
        return await self.fairness.calculate_fairness_metrics(*args, **kwargs)
    
    async def calculate_demographic_distribution(self, *args, **kwargs) -> Dict[str, Any]:
        """Calculate demographic distribution"""
        return await self.fairness.calculate_demographic_distribution(*args, **kwargs)
    
    async def store_bias_fairness_metrics(self, *args, **kwargs) -> str:
        """Store bias fairness metrics"""
        return await self.fairness.store_bias_fairness_metrics(*args, **kwargs)
    
    async def store_demographic_distribution(self, *args, **kwargs) -> str:
        """Store demographic distribution"""
        return await self.fairness.store_demographic_distribution(*args, **kwargs)
    
    # Explainability - delegate to ExplainabilityService
    async def generate_shap_explanation(self, *args, **kwargs) -> ModelExplanation:
        """Generate SHAP explanation"""
        return await self.explainability.generate_shap_explanation(*args, **kwargs)
    
    async def generate_lime_explanation(self, *args, **kwargs) -> ModelExplanation:
        """Generate LIME explanation"""
        return await self.explainability.generate_lime_explanation(*args, **kwargs)
    
    async def calculate_global_feature_importance(self, *args, **kwargs) -> Dict[str, Any]:
        """Calculate global feature importance"""
        return await self.explainability.calculate_global_feature_importance(*args, **kwargs)
    
    async def store_explanation(self, *args, **kwargs) -> str:
        """Store explanation"""
        return await self.explainability.store_explanation(*args, **kwargs)
    
    # Data Quality - delegate to DataQualityService
    async def detect_outliers(self, *args, **kwargs) -> Dict[str, Any]:
        """Detect outliers"""
        return await self.data_quality.detect_outliers(*args, **kwargs)
    
    async def calculate_data_quality_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        return await self.data_quality.calculate_data_quality_metrics(*args, **kwargs)
    
    async def detect_anomaly(self, *args, **kwargs) -> Dict[str, Any]:
        """Detect anomaly"""
        return await self.data_quality.detect_anomaly(*args, **kwargs)
    
    # Model Lifecycle - delegate to ModelLifecycleService
    async def configure_retraining_trigger(self, *args, **kwargs) -> str:
        """Configure retraining trigger"""
        return await self.lifecycle.configure_retraining_trigger(*args, **kwargs)
    
    async def create_retraining_job(self, *args, **kwargs) -> str:
        """Create retraining job"""
        return await self.lifecycle.create_retraining_job(*args, **kwargs)
    
    async def generate_model_card(self, *args, **kwargs) -> Dict[str, Any]:
        """Generate model card"""
        return await self.lifecycle.generate_model_card(*args, **kwargs)
    
    async def get_model_card(self, *args, **kwargs) -> Dict[str, Any]:
        """Get model card"""
        return await self.lifecycle.get_model_card(*args, **kwargs)
    
    # Governance - delegate to GovernanceService
    async def create_data_lineage(self, *args, **kwargs) -> str:
        """Create data lineage"""
        return await self.governance.create_data_lineage(*args, **kwargs)
    
    async def create_governance_workflow(self, *args, **kwargs) -> str:
        """Create governance workflow"""
        return await self.governance.create_governance_workflow(*args, **kwargs)
    
    async def create_compliance_record(self, *args, **kwargs) -> str:
        """Create compliance record"""
        return await self.governance.create_compliance_record(*args, **kwargs)
    
    async def create_data_retention_policy(self, *args, **kwargs) -> str:
        """Create data retention policy"""
        return await self.governance.create_data_retention_policy(*args, **kwargs)
    
    # Analytics - delegate to AnalyticsService
    async def analyze_time_series_trend(self, *args, **kwargs) -> Dict[str, Any]:
        """Analyze time series trend"""
        return await self.analytics.analyze_time_series_trend(*args, **kwargs)
    
    async def create_comparative_analytics(self, *args, **kwargs) -> str:
        """Create comparative analytics"""
        return await self.analytics.create_comparative_analytics(*args, **kwargs)
    
    async def create_custom_dashboard(self, *args, **kwargs) -> str:
        """Create custom dashboard"""
        return await self.analytics.create_custom_dashboard(*args, **kwargs)
    
    async def create_automated_report(self, *args, **kwargs) -> str:
        """Create automated report"""
        return await self.analytics.create_automated_report(*args, **kwargs)
    
    # Integration - delegate to IntegrationService
    async def create_external_integration(self, *args, **kwargs) -> str:
        """Create external integration"""
        return await self.integration.create_external_integration(*args, **kwargs)
    
    async def create_webhook_config(self, *args, **kwargs) -> str:
        """Create webhook config"""
        return await self.integration.create_webhook_config(*args, **kwargs)
    
    async def create_sampling_config(self, *args, **kwargs) -> str:
        """Create sampling config"""
        return await self.integration.create_sampling_config(*args, **kwargs)
    
    async def create_metric_aggregation_config(self, *args, **kwargs) -> str:
        """Create metric aggregation config"""
        return await self.integration.create_metric_aggregation_config(*args, **kwargs)
    
    async def should_sample(self, *args, **kwargs) -> bool:
        """Check if should sample"""
        return await self.integration.should_sample(*args, **kwargs)
    
    # Audit - delegate to AuditService
    async def log_audit_event(self, *args, **kwargs) -> str:
        """Log audit event"""
        return await self.audit.log_audit_event(*args, **kwargs)
    
    # Dashboard - delegate to DashboardService
    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get dashboard metrics"""
        return await self.dashboard.get_dashboard_metrics()
    
    # Helper methods - delegate to base service methods (available on all sub-services)
    def _extract_confidence_score(self, *args, **kwargs) -> Optional[float]:
        """Extract confidence score"""
        return self.performance._extract_confidence_score(*args, **kwargs)
    
    def _extract_confidence_scores(self, *args, **kwargs) -> Optional[Dict[str, float]]:
        """Extract confidence scores"""
        return self.performance._extract_confidence_scores(*args, **kwargs)
    
    def _extract_uncertainty(self, *args, **kwargs) -> Optional[float]:
        """Extract uncertainty"""
        return self.performance._extract_uncertainty(*args, **kwargs)
    
    def _extract_prediction_interval(self, *args, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        """Extract prediction interval"""
        return self.performance._extract_prediction_interval(*args, **kwargs)
    
    def _extract_prediction_value(self, *args, **kwargs) -> Optional[float]:
        """Extract prediction value"""
        return self.performance._extract_prediction_value(*args, **kwargs)
    
    def _calculate_psi(self, *args, **kwargs) -> float:
        """Calculate PSI"""
        return self.drift._calculate_psi(*args, **kwargs)
    
    def _ks_test(self, *args, **kwargs) -> Tuple[float, float]:
        """KS test"""
        return self.drift._ks_test(*args, **kwargs)
    
    async def _check_database_health(self, *args, **kwargs) -> Tuple[bool, str]:
        """Check database health"""
        return await self.health._check_database_health(*args, **kwargs)


# Global monitoring service instance
monitoring_service = MonitoringService()

# Module-level functions for backward compatibility
async def log_prediction(*args, **kwargs):
    """Module-level function for logging predictions"""
    return await monitoring_service.log_prediction(*args, **kwargs)

