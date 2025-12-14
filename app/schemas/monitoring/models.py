"""
Model management schemas
Contains schemas for model baselines, version comparisons, and resource usage
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class ModelBaseline(BaseModel):
    """Schema for model performance baseline"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique baseline ID")
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Baseline type
    baseline_type: str = Field(..., description="Type of baseline: performance, classification, regression, latency")
    
    # Performance baselines
    baseline_accuracy: Optional[float] = Field(None, description="Baseline accuracy", ge=0, le=1)
    baseline_precision: Optional[float] = Field(None, description="Baseline precision", ge=0, le=1)
    baseline_recall: Optional[float] = Field(None, description="Baseline recall", ge=0, le=1)
    baseline_f1: Optional[float] = Field(None, description="Baseline F1 score", ge=0, le=1)
    baseline_auc_roc: Optional[float] = Field(None, description="Baseline AUC-ROC", ge=0, le=1)
    baseline_mae: Optional[float] = Field(None, description="Baseline MAE", ge=0)
    baseline_mse: Optional[float] = Field(None, description="Baseline MSE", ge=0)
    baseline_rmse: Optional[float] = Field(None, description="Baseline RMSE", ge=0)
    baseline_r2: Optional[float] = Field(None, description="Baseline R² score")
    baseline_p50_latency_ms: Optional[float] = Field(None, description="Baseline P50 latency", ge=0)
    baseline_p95_latency_ms: Optional[float] = Field(None, description="Baseline P95 latency", ge=0)
    baseline_p99_latency_ms: Optional[float] = Field(None, description="Baseline P99 latency", ge=0)
    baseline_avg_latency_ms: Optional[float] = Field(None, description="Baseline average latency", ge=0)
    baseline_avg_confidence: Optional[float] = Field(None, description="Baseline average confidence", ge=0, le=1)
    baseline_low_confidence_rate: Optional[float] = Field(None, description="Baseline low confidence rate", ge=0, le=100)
    
    # Sample information
    baseline_sample_count: int = Field(..., description="Number of samples used for baseline")
    baseline_time_window_start: datetime = Field(..., description="Baseline time window start")
    baseline_time_window_end: datetime = Field(..., description="Baseline time window end")
    
    # Metadata
    is_active: bool = Field(True, description="Whether this is the active baseline")
    is_production: bool = Field(False, description="Whether this is a production baseline")
    created_by: Optional[str] = Field(None, description="User/system that created baseline")
    description: Optional[str] = Field(None, description="Baseline description")
    additional_metrics: Optional[Dict[str, Any]] = Field(None, description="Additional baseline metrics")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Baseline creation timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "model_name": "house_price_predictor",
                "model_version": "1.0.0",
                "baseline_type": "regression",
                "baseline_mae": 15000.0,
                "baseline_r2": 0.85,
                "baseline_sample_count": 1000
            }
        }
    }


class ModelVersionComparison(BaseModel):
    """Schema for model version comparison results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique comparison ID")
    model_name: str = Field(..., description="Model name")
    baseline_version: str = Field(..., description="Baseline model version")
    comparison_version: str = Field(..., description="Version being compared")
    baseline_model_id: str = Field(..., description="Baseline model ID")
    comparison_model_id: str = Field(..., description="Comparison model ID")
    
    # Comparison time window
    comparison_window_start: datetime = Field(..., description="Comparison window start")
    comparison_window_end: datetime = Field(..., description="Comparison window end")
    
    # Performance deltas
    accuracy_delta: Optional[float] = Field(None, description="Change in accuracy")
    precision_delta: Optional[float] = Field(None, description="Change in precision")
    recall_delta: Optional[float] = Field(None, description="Change in recall")
    f1_delta: Optional[float] = Field(None, description="Change in F1 score")
    auc_roc_delta: Optional[float] = Field(None, description="Change in AUC-ROC")
    mae_delta: Optional[float] = Field(None, description="Change in MAE")
    mse_delta: Optional[float] = Field(None, description="Change in MSE")
    rmse_delta: Optional[float] = Field(None, description="Change in RMSE")
    r2_delta: Optional[float] = Field(None, description="Change in R² score")
    p50_latency_delta_ms: Optional[float] = Field(None, description="Change in P50 latency")
    p95_latency_delta_ms: Optional[float] = Field(None, description="Change in P95 latency")
    p99_latency_delta_ms: Optional[float] = Field(None, description="Change in P99 latency")
    avg_latency_delta_ms: Optional[float] = Field(None, description="Change in average latency")
    avg_confidence_delta: Optional[float] = Field(None, description="Change in average confidence")
    low_confidence_rate_delta: Optional[float] = Field(None, description="Change in low confidence rate")
    
    # Statistical significance
    p_value: Optional[float] = Field(None, description="Statistical p-value", ge=0, le=1)
    is_statistically_significant: bool = Field(False, description="Whether difference is statistically significant")
    
    # Overall assessment
    performance_improved: bool = Field(False, description="Whether performance improved")
    performance_degraded: bool = Field(False, description="Whether performance degraded")
    performance_regression_severity: Optional[str] = Field(None, description="Severity of regression")
    comparison_summary: Optional[str] = Field(None, description="Human-readable summary")
    recommendation: Optional[str] = Field(None, description="Recommendation: promote, rollback, investigate, no_change")
    comparison_details: Optional[Dict[str, Any]] = Field(None, description="Detailed comparison metrics")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Comparison timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name": "house_price_predictor",
                "baseline_version": "1.0.0",
                "comparison_version": "1.1.0",
                "accuracy_delta": 0.02,
                "performance_improved": True,
                "recommendation": "promote"
            }
        }
    }


class ModelResourceUsage(BaseModel):
    """Schema for model-specific resource usage metrics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique resource usage record ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Resource metrics
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage", ge=0, le=100)
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB", ge=0)
    gpu_usage_percent: Optional[float] = Field(None, description="GPU usage percentage", ge=0, le=100)
    gpu_memory_usage_mb: Optional[float] = Field(None, description="GPU memory usage in MB", ge=0)
    
    # Network I/O
    network_bytes_sent: Optional[float] = Field(None, description="Network bytes sent", ge=0)
    network_bytes_recv: Optional[float] = Field(None, description="Network bytes received", ge=0)
    
    # Performance correlation
    avg_latency_ms: Optional[float] = Field(None, description="Average latency during this period")
    requests_per_second: Optional[float] = Field(None, description="Requests per second during this period")
    
    # Time window
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
    time_window_start: datetime = Field(..., description="Start of measurement window")
    time_window_end: datetime = Field(..., description="End of measurement window")
    
    # Additional metadata
    host: Optional[str] = Field(None, description="Host/server name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "cpu_usage_percent": 45.5,
                "memory_usage_mb": 1024.0,
                "gpu_usage_percent": 78.2,
                "gpu_memory_usage_mb": 4096.0,
                "avg_latency_ms": 45.2,
                "requests_per_second": 10.5
            }
        }
    }

