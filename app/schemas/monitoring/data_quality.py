"""
Data quality and outlier detection schemas
Contains schemas for outlier detection, anomaly detection, and data quality metrics
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class OutlierDetectionMethod(str, Enum):
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL = "statistical"


class OutlierType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    TEMPORAL = "temporal"


class AnomalyType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    TEMPORAL = "temporal"
    PATTERN = "pattern"


class OutlierDetection(BaseModel):
    """Schema for outlier detection results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique detection ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    prediction_log_id: Optional[str] = Field(None, description="Prediction log ID")
    
    # Detection metadata
    detection_method: OutlierDetectionMethod = Field(..., description="Detection method used")
    outlier_type: OutlierType = Field(..., description="Type of outlier")
    
    # Input/output data
    input_data: Optional[Dict[str, Any]] = Field(None, description="Original input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Model output")
    prediction: Optional[Any] = Field(None, description="Prediction value")
    
    # Outlier scores
    outlier_score: Optional[float] = Field(None, description="Overall outlier score (0-1)", ge=0, le=1)
    z_score: Optional[float] = Field(None, description="Z-score")
    iqr_score: Optional[float] = Field(None, description="IQR-based score")
    isolation_forest_score: Optional[float] = Field(None, description="Isolation Forest score")
    
    # Feature-level scores
    feature_outlier_scores: Dict[str, float] = Field(default_factory=dict, description="Feature-level outlier scores")
    outlier_features: List[str] = Field(default_factory=list, description="Features that are outliers")
    
    # Thresholds
    z_score_threshold: float = Field(3.0, description="Z-score threshold", ge=0)
    iqr_multiplier: float = Field(1.5, description="IQR multiplier", ge=0)
    
    # Detection result
    is_outlier: bool = Field(False, description="Whether this is an outlier")
    severity: Optional[str] = Field(None, description="Outlier severity")
    
    # Timestamps
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")


class AnomalyDetection(BaseModel):
    """Schema for anomaly detection results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique detection ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    prediction_log_id: Optional[str] = Field(None, description="Prediction log ID")
    
    # Anomaly metadata
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    detection_method: str = Field(..., description="Detection method")
    
    # Anomaly data
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    baseline_data: Optional[Dict[str, Any]] = Field(None, description="Baseline for comparison")
    
    # Anomaly scores
    anomaly_score: float = Field(..., description="Anomaly score (0-1)", ge=0, le=1)
    confidence: Optional[float] = Field(None, description="Detection confidence (0-1)", ge=0, le=1)
    
    # Anomaly characteristics
    anomaly_features: List[str] = Field(default_factory=list, description="Features contributing to anomaly")
    anomaly_reason: Optional[str] = Field(None, description="Human-readable reason")
    
    # Detection result
    is_anomaly: bool = Field(False, description="Whether this is an anomaly")
    severity: Optional[str] = Field(None, description="Anomaly severity")
    alert_triggered: bool = Field(False, description="Whether alert was triggered")
    alert_id: Optional[str] = Field(None, description="Alert ID if triggered")
    
    # Timestamps
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")


class DataQualityMetrics(BaseModel):
    """Schema for data quality metrics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique metrics ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Time window
    time_window_start: datetime = Field(..., description="Metrics window start")
    time_window_end: datetime = Field(..., description="Metrics window end")
    
    # Data quality scores (0-1, higher is better)
    completeness_score: Optional[float] = Field(None, description="Completeness score", ge=0, le=1)
    consistency_score: Optional[float] = Field(None, description="Consistency score", ge=0, le=1)
    validity_score: Optional[float] = Field(None, description="Validity score", ge=0, le=1)
    overall_quality_score: Optional[float] = Field(None, description="Overall quality score", ge=0, le=1)
    
    # Quality metrics
    total_samples: int = Field(0, description="Total samples analyzed")
    valid_samples: int = Field(0, description="Valid samples")
    invalid_samples: int = Field(0, description="Invalid samples")
    missing_value_count: int = Field(0, description="Missing value count")
    schema_violations: int = Field(0, description="Schema violation count")
    range_violations: int = Field(0, description="Range violation count")
    type_violations: int = Field(0, description="Type violation count")
    
    # Per-feature quality metrics
    feature_quality_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-feature quality metrics")
    
    # Quality trends
    quality_trend: Optional[str] = Field(None, description="Quality trend: improving, stable, degrading")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")

