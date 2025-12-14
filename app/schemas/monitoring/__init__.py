"""
Monitoring schemas package
Re-exports all monitoring schemas from sub-modules
"""

# Base schemas
from app.schemas.monitoring.base import (
    MetricType,
    AlertSeverity,
    SystemComponent,
    SystemStatus,
    PredictionLog,
    ModelPerformanceMetrics,
    SystemHealthMetric,
    SystemHealthStatus,
    AuditLog,
    ModelUsageAnalytics,
    DashboardMetrics,
)

# Alert schemas
from app.schemas.monitoring.alerts import (
    AlertCondition,
    AlertRule,
    NotificationChannelType,
    NotificationChannel,
    AlertGroup,
    EscalationTriggerCondition,
    AlertEscalation,
    Alert,
)

# Drift schemas
from app.schemas.monitoring.drift import (
    DriftType,
    DriftSeverity,
    ModelDriftDetection,
    ModelPerformanceHistory,
    ModelConfidenceMetrics,
)

# Model management schemas
from app.schemas.monitoring.models import (
    ModelBaseline,
    ModelVersionComparison,
    ModelResourceUsage,
)

# Testing schemas
from app.schemas.monitoring.testing import (
    ABTestStatus,
    ABTest,
    ABTestMetrics,
    ABTestComparison,
    CanaryDeploymentStatus,
    CanaryDeployment,
    CanaryMetrics,
)

# Fairness schemas
from app.schemas.monitoring.fairness import (
    ProtectedAttributeType,
    AnonymizationMethod,
    ProtectedAttributeConfig,
    BiasFairnessMetrics,
    DemographicDistribution,
)

# Explainability schemas
from app.schemas.monitoring.explainability import (
    ExplanationType,
    ImportanceType,
    ModelExplanation,
    FeatureImportance,
)

# Data quality schemas
from app.schemas.monitoring.data_quality import (
    OutlierDetectionMethod,
    OutlierType,
    AnomalyType,
    OutlierDetection,
    AnomalyDetection,
    DataQualityMetrics,
)

# Lifecycle schemas
from app.schemas.monitoring.lifecycle import (
    RetrainingTriggerType,
    RetrainingJobStatus,
    ReplacementStatus,
    RetrainingJob,
    RetrainingTriggerConfig,
    ModelCard,
)

# Governance schemas
from app.schemas.monitoring.governance import (
    LineageType,
    RelationshipType,
    DataLineage,
    WorkflowType,
    WorkflowStatus,
    GovernanceWorkflow,
    ComplianceType,
    ComplianceRecordType,
    ComplianceRecordStatus,
    ComplianceRecord,
    DataRetentionPolicy,
)

# Analytics schemas
from app.schemas.monitoring.analytics import (
    AnalysisType,
    TrendDirection,
    TimeSeriesAnalysis,
    ComparisonType,
    ComparativeAnalytics,
    CustomDashboard,
    ReportType,
    ScheduleType,
    AutomatedReport,
)

# Integration schemas
from app.schemas.monitoring.integration import (
    IntegrationType,
    ExternalIntegration,
    WebhookConfig,
    SamplingStrategy,
    SamplingConfig,
    AggregationMethod,
    MetricAggregationConfig,
)

__all__ = [
    # Base
    "MetricType",
    "AlertSeverity",
    "SystemComponent",
    "SystemStatus",
    "PredictionLog",
    "ModelPerformanceMetrics",
    "SystemHealthMetric",
    "SystemHealthStatus",
    "AuditLog",
    "ModelUsageAnalytics",
    "DashboardMetrics",
    # Alerts
    "AlertCondition",
    "AlertRule",
    "NotificationChannelType",
    "NotificationChannel",
    "AlertGroup",
    "EscalationTriggerCondition",
    "AlertEscalation",
    "Alert",
    # Drift
    "DriftType",
    "DriftSeverity",
    "ModelDriftDetection",
    "ModelPerformanceHistory",
    "ModelConfidenceMetrics",
    # Models
    "ModelBaseline",
    "ModelVersionComparison",
    "ModelResourceUsage",
    # Testing
    "ABTestStatus",
    "ABTest",
    "ABTestMetrics",
    "ABTestComparison",
    "CanaryDeploymentStatus",
    "CanaryDeployment",
    "CanaryMetrics",
    # Fairness
    "ProtectedAttributeType",
    "AnonymizationMethod",
    "ProtectedAttributeConfig",
    "BiasFairnessMetrics",
    "DemographicDistribution",
    # Explainability
    "ExplanationType",
    "ImportanceType",
    "ModelExplanation",
    "FeatureImportance",
    # Data Quality
    "OutlierDetectionMethod",
    "OutlierType",
    "AnomalyType",
    "OutlierDetection",
    "AnomalyDetection",
    "DataQualityMetrics",
    # Lifecycle
    "RetrainingTriggerType",
    "RetrainingJobStatus",
    "ReplacementStatus",
    "RetrainingJob",
    "RetrainingTriggerConfig",
    "ModelCard",
    # Governance
    "LineageType",
    "RelationshipType",
    "DataLineage",
    "WorkflowType",
    "WorkflowStatus",
    "GovernanceWorkflow",
    "ComplianceType",
    "ComplianceRecordType",
    "ComplianceRecordStatus",
    "ComplianceRecord",
    "DataRetentionPolicy",
    # Analytics
    "AnalysisType",
    "TrendDirection",
    "TimeSeriesAnalysis",
    "ComparisonType",
    "ComparativeAnalytics",
    "CustomDashboard",
    "ReportType",
    "ScheduleType",
    "AutomatedReport",
    # Integration
    "IntegrationType",
    "ExternalIntegration",
    "WebhookConfig",
    "SamplingStrategy",
    "SamplingConfig",
    "AggregationMethod",
    "MetricAggregationConfig",
]

