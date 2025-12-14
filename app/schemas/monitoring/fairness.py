"""
Bias and fairness monitoring schemas
Contains schemas for protected attributes, bias metrics, and demographic distribution
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ProtectedAttributeType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BINARY = "binary"


class AnonymizationMethod(str, Enum):
    HASH = "hash"
    K_ANONYMITY = "k_anonymity"
    DIFFERENTIAL_PRIVACY = "differential_privacy"


class ProtectedAttributeConfig(BaseModel):
    """Schema for protected attribute configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique config ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Protected attribute configuration
    attribute_name: str = Field(..., description="Protected attribute name (e.g., 'gender', 'age_group')")
    attribute_type: ProtectedAttributeType = Field(..., description="Attribute type")
    attribute_values: Optional[List[str]] = Field(None, description="Possible values for categorical")
    attribute_ranges: Optional[Dict[str, Any]] = Field(None, description="Ranges for numerical attributes")
    
    # Privacy settings
    use_privacy_preserving: bool = Field(False, description="Use privacy-preserving tracking")
    anonymization_method: Optional[AnonymizationMethod] = Field(None, description="Anonymization method")
    
    # Tracking settings
    is_active: bool = Field(True, description="Whether tracking is active")
    tracking_enabled: bool = Field(True, description="Whether tracking is enabled")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created config")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class BiasFairnessMetrics(BaseModel):
    """Schema for bias and fairness metrics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique metrics ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Time window
    time_window_start: datetime = Field(..., description="Metrics window start")
    time_window_end: datetime = Field(..., description="Metrics window end")
    
    # Protected attribute
    protected_attribute: str = Field(..., description="Protected attribute name")
    protected_attribute_values: Optional[List[str]] = Field(None, description="Possible attribute values")
    
    # Demographic parity metrics
    demographic_parity_score: Optional[float] = Field(None, description="Demographic parity score (0-1)", ge=0, le=1)
    demographic_parity_ratio: Optional[float] = Field(None, description="Demographic parity ratio", ge=0)
    demographic_parity_difference: Optional[float] = Field(None, description="Demographic parity difference")
    
    # Equalized odds metrics
    equalized_odds_score: Optional[float] = Field(None, description="Equalized odds score (0-1)", ge=0, le=1)
    true_positive_rate_difference: Optional[float] = Field(None, description="TPR difference between groups")
    false_positive_rate_difference: Optional[float] = Field(None, description="FPR difference between groups")
    
    # Equal opportunity metrics
    equal_opportunity_score: Optional[float] = Field(None, description="Equal opportunity score (0-1)", ge=0, le=1)
    equal_opportunity_difference: Optional[float] = Field(None, description="Equal opportunity difference")
    
    # Group-based metrics
    group_metrics: Dict[str, Any] = Field(default_factory=dict, description="Per-group metrics")
    
    # Bias scores
    overall_bias_score: Optional[float] = Field(None, description="Overall bias score (0-1)", ge=0, le=1)
    prediction_bias_by_group: Dict[str, float] = Field(default_factory=dict, description="Bias scores per group")
    feature_bias_scores: Dict[str, float] = Field(default_factory=dict, description="Bias scores per feature")
    
    # Statistical significance
    p_value: Optional[float] = Field(None, description="Statistical p-value", ge=0, le=1)
    is_statistically_significant: bool = Field(False, description="Whether difference is significant")
    
    # Fairness thresholds
    fairness_threshold: float = Field(0.8, description="Minimum fairness score", ge=0, le=1)
    bias_threshold: float = Field(0.2, description="Maximum acceptable bias", ge=0, le=1)
    
    # Alert information
    fairness_violation_detected: bool = Field(False, description="Whether fairness violation detected")
    bias_alert_triggered: bool = Field(False, description="Whether bias alert triggered")
    alert_id: Optional[str] = Field(None, description="Alert ID if triggered")
    
    # Additional metadata
    sample_size: int = Field(0, description="Number of samples analyzed")
    positive_class_rate: Optional[float] = Field(None, description="Overall positive prediction rate", ge=0, le=1)
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class DemographicDistribution(BaseModel):
    """Schema for demographic distribution tracking"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique distribution ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Time window
    time_window_start: datetime = Field(..., description="Distribution window start")
    time_window_end: datetime = Field(..., description="Distribution window end")
    
    # Protected attribute
    protected_attribute: str = Field(..., description="Protected attribute name")
    
    # Distribution data
    group_distribution: Dict[str, int] = Field(..., description="Group value to count mapping")
    group_percentages: Dict[str, float] = Field(..., description="Group value to percentage mapping")
    total_samples: int = Field(0, description="Total number of samples")
    
    # Prediction distribution by group
    prediction_distribution_by_group: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Prediction distribution per group")
    positive_rate_by_group: Dict[str, float] = Field(default_factory=dict, description="Positive rate per group")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Distribution timestamp")

