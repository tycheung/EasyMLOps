"""
Drift detection schemas
Contains schemas for model drift detection, performance history, and confidence metrics
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class DriftType(str, Enum):
    """Types of drift that can be detected"""
    FEATURE = "feature"
    PREDICTION = "prediction"
    DATA = "data"


class DriftSeverity(str, Enum):
    """Severity levels for detected drift"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelDriftDetection(BaseModel):
    """Schema for model drift detection results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique drift detection ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Drift detection metadata
    drift_type: DriftType = Field(..., description="Type of drift detected")
    detection_method: str = Field(..., description="Method used for detection")
    
    # Baseline and current data windows
    baseline_window_start: datetime = Field(..., description="Baseline period start")
    baseline_window_end: datetime = Field(..., description="Baseline period end")
    current_window_start: datetime = Field(..., description="Current period start")
    current_window_end: datetime = Field(..., description="Current period end")
    
    # Drift detection results
    drift_detected: bool = Field(False, description="Whether drift was detected")
    drift_score: Optional[float] = Field(None, description="Overall drift score", ge=0)
    drift_severity: Optional[DriftSeverity] = Field(None, description="Severity of detected drift")
    p_value: Optional[float] = Field(None, description="Statistical p-value", ge=0, le=1)
    
    # Feature-specific drift
    feature_drift_scores: Optional[Dict[str, float]] = Field(None, description="Per-feature drift scores")
    feature_drift_details: Optional[Dict[str, Any]] = Field(None, description="Detailed per-feature drift information")
    
    # Prediction drift metrics
    prediction_mean_shift: Optional[float] = Field(None, description="Mean shift in predictions")
    prediction_variance_shift: Optional[float] = Field(None, description="Variance shift in predictions")
    prediction_distribution_shift: Optional[float] = Field(None, description="Distribution shift score")
    
    # Data quality metrics
    data_quality_metrics: Optional[Dict[str, Any]] = Field(None, description="Data quality metrics")
    schema_changes: Optional[Dict[str, Any]] = Field(None, description="Schema change information")
    
    # Thresholds
    drift_threshold: Optional[float] = Field(None, description="Threshold used for detection")
    
    # Alert information
    alert_triggered: bool = Field(False, description="Whether an alert was triggered")
    alert_id: Optional[str] = Field(None, description="Alert ID if alert was created")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "drift_type": "feature",
                "detection_method": "ks_test",
                "drift_detected": True,
                "drift_score": 0.75,
                "drift_severity": "high",
                "p_value": 0.001,
                "feature_drift_scores": {"feature1": 0.8, "feature2": 0.3}
            }
        }
    }


class ModelPerformanceHistory(BaseModel):
    """Schema for model performance metrics with ground truth"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique performance history ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Time window
    time_window_start: datetime = Field(..., description="Start of measurement window")
    time_window_end: datetime = Field(..., description="End of measurement window")
    
    # Model type
    model_type: str = Field(..., description="Model type: classification or regression")
    
    # Classification metrics
    accuracy: Optional[float] = Field(None, description="Accuracy score", ge=0, le=1)
    precision: Optional[float] = Field(None, description="Precision score", ge=0, le=1)
    recall: Optional[float] = Field(None, description="Recall score", ge=0, le=1)
    f1_score: Optional[float] = Field(None, description="F1 score", ge=0, le=1)
    auc_roc: Optional[float] = Field(None, description="AUC-ROC score", ge=0, le=1)
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")
    
    # Regression metrics
    mae: Optional[float] = Field(None, description="Mean Absolute Error", ge=0)
    mse: Optional[float] = Field(None, description="Mean Squared Error", ge=0)
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error", ge=0)
    r2_score: Optional[float] = Field(None, description="R² score")
    
    # Sample counts
    total_samples: int = Field(..., description="Total number of samples")
    samples_with_ground_truth: int = Field(..., description="Number of samples with ground truth")
    
    # Baseline comparison
    baseline_accuracy: Optional[float] = Field(None, description="Baseline accuracy")
    baseline_f1: Optional[float] = Field(None, description="Baseline F1 score")
    baseline_mae: Optional[float] = Field(None, description="Baseline MAE")
    baseline_r2: Optional[float] = Field(None, description="Baseline R² score")
    
    # Performance delta
    accuracy_delta: Optional[float] = Field(None, description="Change in accuracy from baseline")
    f1_delta: Optional[float] = Field(None, description="Change in F1 from baseline")
    mae_delta: Optional[float] = Field(None, description="Change in MAE from baseline")
    r2_delta: Optional[float] = Field(None, description="Change in R² from baseline")
    
    # Degradation detection
    performance_degraded: bool = Field(False, description="Whether performance has degraded")
    degradation_severity: Optional[str] = Field(None, description="Severity of degradation")
    degradation_threshold: Optional[float] = Field(None, description="Threshold used for detection")
    p_value: Optional[float] = Field(None, description="Statistical p-value", ge=0, le=1)
    
    # Additional metrics
    per_class_metrics: Optional[Dict[str, Any]] = Field(None, description="Per-class performance metrics")
    residual_stats: Optional[Dict[str, Any]] = Field(None, description="Residual statistics")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "model_type": "classification",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "total_samples": 1000,
                "samples_with_ground_truth": 950,
                "performance_degraded": False
            }
        }
    }


class ModelConfidenceMetrics(BaseModel):
    """Schema for confidence and uncertainty metrics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique confidence metrics ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Time window
    time_window_start: datetime = Field(..., description="Start of measurement window")
    time_window_end: datetime = Field(..., description="End of measurement window")
    
    # Confidence metrics
    avg_confidence: Optional[float] = Field(None, description="Average confidence score", ge=0, le=1)
    min_confidence: Optional[float] = Field(None, description="Minimum confidence", ge=0, le=1)
    max_confidence: Optional[float] = Field(None, description="Maximum confidence", ge=0, le=1)
    median_confidence: Optional[float] = Field(None, description="Median confidence", ge=0, le=1)
    std_dev_confidence: Optional[float] = Field(None, description="Standard deviation of confidence", ge=0)
    confidence_distribution: Optional[Dict[str, Any]] = Field(None, description="Histogram of confidence scores")
    
    # Low confidence tracking
    low_confidence_count: int = Field(0, description="Count of predictions below threshold")
    low_confidence_threshold: Optional[float] = Field(None, description="Threshold used for low confidence", ge=0, le=1)
    low_confidence_percentage: Optional[float] = Field(None, description="Percentage of low confidence predictions", ge=0, le=100)
    
    # Confidence calibration
    calibration_error: Optional[float] = Field(None, description="Expected Calibration Error (ECE)", ge=0, le=1)
    brier_score: Optional[float] = Field(None, description="Brier score for calibration", ge=0, le=1)
    confidence_accuracy_correlation: Optional[float] = Field(None, description="Correlation between confidence and accuracy", ge=-1, le=1)
    
    # Uncertainty metrics
    avg_uncertainty: Optional[float] = Field(None, description="Average uncertainty score", ge=0)
    high_uncertainty_count: int = Field(0, description="Count of high uncertainty predictions")
    high_uncertainty_threshold: Optional[float] = Field(None, description="Threshold for high uncertainty")
    uncertainty_distribution: Optional[Dict[str, Any]] = Field(None, description="Distribution of uncertainty scores")
    
    # Prediction intervals
    avg_interval_width: Optional[float] = Field(None, description="Average width of prediction intervals", ge=0)
    coverage_rate: Optional[float] = Field(None, description="Percentage of true values within intervals", ge=0, le=100)
    
    # Sample counts
    total_samples: int = Field(..., description="Total number of samples")
    samples_with_confidence: int = Field(..., description="Number of samples with confidence scores")
    samples_with_ground_truth: int = Field(0, description="Number of samples with ground truth")
    
    # Additional metrics
    per_class_confidence: Optional[Dict[str, Any]] = Field(None, description="Per-class confidence metrics")
    confidence_trend: Optional[Dict[str, Any]] = Field(None, description="Confidence trend over time")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "avg_confidence": 0.85,
                "min_confidence": 0.12,
                "max_confidence": 0.99,
                "low_confidence_count": 15,
                "low_confidence_percentage": 5.0,
                "total_samples": 300,
                "samples_with_confidence": 300
            }
        }
    }

