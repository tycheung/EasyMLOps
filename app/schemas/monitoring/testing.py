"""
A/B Testing and Canary Deployment schemas
Contains schemas for A/B testing and canary deployment configurations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ABTestStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABTest(BaseModel):
    """Schema for A/B test configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique test ID")
    test_name: str = Field(..., description="Test name/identifier")
    description: Optional[str] = Field(None, description="Test description")
    
    # Test configuration
    model_name: str = Field(..., description="Base model name")
    variant_a_model_id: str = Field(..., description="Control variant model ID")
    variant_b_model_id: str = Field(..., description="Treatment variant model ID")
    variant_a_deployment_id: Optional[str] = Field(None, description="Control variant deployment ID")
    variant_b_deployment_id: Optional[str] = Field(None, description="Treatment variant deployment ID")
    
    # Traffic splitting
    variant_a_percentage: float = Field(50.0, description="Traffic percentage for variant A", ge=0, le=100)
    variant_b_percentage: float = Field(50.0, description="Traffic percentage for variant B", ge=0, le=100)
    use_sticky_sessions: bool = Field(False, description="Use sticky session routing")
    
    # Test status
    status: ABTestStatus = Field(ABTestStatus.DRAFT, description="Test status")
    start_time: Optional[datetime] = Field(None, description="Test start time")
    end_time: Optional[datetime] = Field(None, description="Test end time")
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start time")
    scheduled_end: Optional[datetime] = Field(None, description="Scheduled end time")
    
    # Test criteria
    min_sample_size: Optional[int] = Field(None, description="Minimum samples per variant", ge=1)
    significance_level: float = Field(0.05, description="Statistical significance threshold", ge=0, le=1)
    primary_metric: str = Field("accuracy", description="Primary metric for comparison")
    
    # Test results
    variant_a_samples: int = Field(0, description="Number of samples for variant A")
    variant_b_samples: int = Field(0, description="Number of samples for variant B")
    winner: Optional[str] = Field(None, description="Test winner: variant_a, variant_b, no_winner, inconclusive")
    conclusion_reason: Optional[str] = Field(None, description="Reason for conclusion")
    
    # Statistical results
    p_value: Optional[float] = Field(None, description="Statistical p-value", ge=0, le=1)
    confidence_interval_lower: Optional[float] = Field(None, description="Lower bound of confidence interval")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper bound of confidence interval")
    is_statistically_significant: bool = Field(False, description="Whether result is statistically significant")
    
    # Additional configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional test configuration")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created the test")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Test creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "test_name": "model_v2_ab_test",
                "model_name": "house_price_predictor",
                "variant_a_model_id": "model_v1_id",
                "variant_b_model_id": "model_v2_id",
                "variant_a_percentage": 50.0,
                "variant_b_percentage": 50.0,
                "primary_metric": "accuracy"
            }
        }
    }


class ABTestMetrics(BaseModel):
    """Schema for A/B test metrics per variant"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique metrics ID")
    test_id: str = Field(..., description="A/B test ID")
    variant: str = Field(..., description="Variant: variant_a or variant_b")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Time window
    time_window_start: datetime = Field(..., description="Metrics window start")
    time_window_end: datetime = Field(..., description="Metrics window end")
    
    # Performance metrics
    total_requests: int = Field(0, description="Total requests")
    successful_requests: int = Field(0, description="Successful requests")
    failed_requests: int = Field(0, description="Failed requests")
    avg_latency_ms: Optional[float] = Field(None, description="Average latency", ge=0)
    p50_latency_ms: Optional[float] = Field(None, description="P50 latency", ge=0)
    p95_latency_ms: Optional[float] = Field(None, description="P95 latency", ge=0)
    p99_latency_ms: Optional[float] = Field(None, description="P99 latency", ge=0)
    
    # Classification metrics
    accuracy: Optional[float] = Field(None, description="Accuracy", ge=0, le=1)
    precision: Optional[float] = Field(None, description="Precision", ge=0, le=1)
    recall: Optional[float] = Field(None, description="Recall", ge=0, le=1)
    f1_score: Optional[float] = Field(None, description="F1 score", ge=0, le=1)
    auc_roc: Optional[float] = Field(None, description="AUC-ROC", ge=0, le=1)
    
    # Regression metrics
    mae: Optional[float] = Field(None, description="MAE", ge=0)
    mse: Optional[float] = Field(None, description="MSE", ge=0)
    rmse: Optional[float] = Field(None, description="RMSE", ge=0)
    r2_score: Optional[float] = Field(None, description="R² score")
    
    # Confidence metrics
    avg_confidence: Optional[float] = Field(None, description="Average confidence", ge=0, le=1)
    low_confidence_rate: Optional[float] = Field(None, description="Low confidence rate", ge=0, le=100)
    
    # Error metrics
    error_rate: Optional[float] = Field(None, description="Error rate", ge=0, le=100)
    
    # Additional metrics
    additional_metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class ABTestComparison(BaseModel):
    """Schema for A/B test comparison results"""
    test_id: str = Field(..., description="A/B test ID")
    variant_a_metrics: ABTestMetrics = Field(..., description="Variant A metrics")
    variant_b_metrics: ABTestMetrics = Field(..., description="Variant B metrics")
    
    # Comparison deltas
    accuracy_delta: Optional[float] = Field(None, description="Difference in accuracy (B - A)")
    precision_delta: Optional[float] = Field(None, description="Difference in precision")
    recall_delta: Optional[float] = Field(None, description="Difference in recall")
    f1_delta: Optional[float] = Field(None, description="Difference in F1")
    mae_delta: Optional[float] = Field(None, description="Difference in MAE")
    r2_delta: Optional[float] = Field(None, description="Difference in R²")
    latency_delta_ms: Optional[float] = Field(None, description="Difference in average latency")
    
    # Statistical significance
    p_value: Optional[float] = Field(None, description="Statistical p-value", ge=0, le=1)
    confidence_interval_lower: Optional[float] = Field(None, description="Lower CI bound")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper CI bound")
    is_statistically_significant: bool = Field(False, description="Whether difference is significant")
    
    # Winner determination
    winner: Optional[str] = Field(None, description="Winner: variant_a, variant_b, or no_winner")
    recommendation: Optional[str] = Field(None, description="Recommendation based on results")
    
    # Additional comparison data
    comparison_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed comparison")


class CanaryDeploymentStatus(str, Enum):
    PENDING = "pending"
    ROLLING_OUT = "rolling_out"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class CanaryDeployment(BaseModel):
    """Schema for canary deployment configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique canary deployment ID")
    deployment_name: str = Field(..., description="Canary deployment name")
    model_id: str = Field(..., description="Model ID")
    production_deployment_id: str = Field(..., description="Production deployment ID")
    canary_deployment_id: str = Field(..., description="Canary deployment ID")
    
    # Rollout configuration
    current_traffic_percentage: float = Field(0.0, description="Current canary traffic percentage", ge=0, le=100)
    target_traffic_percentage: float = Field(100.0, description="Target canary traffic percentage", ge=0, le=100)
    rollout_schedule: Optional[Dict[str, Any]] = Field(None, description="Gradual rollout schedule")
    rollout_step_size: Optional[float] = Field(10.0, description="Percentage increase per step", ge=0, le=100)
    rollout_step_duration_minutes: Optional[int] = Field(60, description="Time between steps in minutes", ge=1)
    
    # Health and rollback configuration
    max_error_rate_threshold: Optional[float] = Field(5.0, description="Max error rate % before rollback", ge=0, le=100)
    max_latency_increase_pct: Optional[float] = Field(50.0, description="Max latency increase % before rollback", ge=0)
    min_health_check_duration_minutes: Optional[int] = Field(5, description="Min time before health check", ge=1)
    health_check_window_minutes: Optional[int] = Field(15, description="Window for health metrics", ge=1)
    
    # Status
    status: CanaryDeploymentStatus = Field(CanaryDeploymentStatus.PENDING, description="Canary deployment status")
    started_at: Optional[datetime] = Field(None, description="Rollout start time")
    completed_at: Optional[datetime] = Field(None, description="Rollout completion time")
    rolled_back_at: Optional[datetime] = Field(None, description="Rollback time")
    
    # Rollout progress
    current_step: int = Field(0, description="Current rollout step")
    total_steps: Optional[int] = Field(None, description="Total rollout steps")
    next_step_time: Optional[datetime] = Field(None, description="Next step scheduled time")
    
    # Health metrics
    canary_error_rate: Optional[float] = Field(None, description="Canary error rate", ge=0, le=100)
    production_error_rate: Optional[float] = Field(None, description="Production error rate", ge=0, le=100)
    canary_avg_latency_ms: Optional[float] = Field(None, description="Canary average latency", ge=0)
    production_avg_latency_ms: Optional[float] = Field(None, description="Production average latency", ge=0)
    health_status: Optional[str] = Field(None, description="Health status: healthy, degraded, unhealthy")
    last_health_check: Optional[datetime] = Field(None, description="Last health check time")
    
    # Rollback information
    rollback_reason: Optional[str] = Field(None, description="Reason for rollback")
    rollback_triggered_by: Optional[str] = Field(None, description="What triggered rollback")
    
    # Additional configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created canary")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "deployment_name": "model_v2_canary",
                "model_id": "abc123",
                "production_deployment_id": "prod_deploy_123",
                "canary_deployment_id": "canary_deploy_456",
                "current_traffic_percentage": 10.0,
                "target_traffic_percentage": 100.0,
                "rollout_step_size": 10.0
            }
        }
    }


class CanaryMetrics(BaseModel):
    """Schema for canary deployment metrics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique metrics ID")
    canary_deployment_id: str = Field(..., description="Canary deployment ID")
    
    # Time window
    time_window_start: datetime = Field(..., description="Metrics window start")
    time_window_end: datetime = Field(..., description="Metrics window end")
    
    # Canary metrics
    canary_total_requests: int = Field(0, description="Canary total requests")
    canary_successful_requests: int = Field(0, description="Canary successful requests")
    canary_failed_requests: int = Field(0, description="Canary failed requests")
    canary_error_rate: Optional[float] = Field(None, description="Canary error rate", ge=0, le=100)
    canary_avg_latency_ms: Optional[float] = Field(None, description="Canary average latency", ge=0)
    
    # Production metrics
    production_total_requests: int = Field(0, description="Production total requests")
    production_successful_requests: int = Field(0, description="Production successful requests")
    production_failed_requests: int = Field(0, description="Production failed requests")
    production_error_rate: Optional[float] = Field(None, description="Production error rate", ge=0, le=100)
    production_avg_latency_ms: Optional[float] = Field(None, description="Production average latency", ge=0)
    
    # Comparison
    error_rate_delta: Optional[float] = Field(None, description="Error rate difference (canary - production)")
    latency_delta_ms: Optional[float] = Field(None, description="Latency difference (canary - production)")
    latency_increase_pct: Optional[float] = Field(None, description="Latency increase percentage")
    
    # Health assessment
    health_status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    health_check_passed: bool = Field(False, description="Whether health check passed")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")

