"""
Model lifecycle management schemas
Contains schemas for retraining jobs, triggers, and model cards
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class RetrainingTriggerType(str, Enum):
    PERFORMANCE = "performance"
    DRIFT = "drift"
    TIME = "time"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"


class RetrainingJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReplacementStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REPLACED = "replaced"


class RetrainingJob(BaseModel):
    """Schema for retraining job"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job ID")
    model_id: str = Field(..., description="Model ID")
    base_model_id: Optional[str] = Field(None, description="Original model being retrained")
    
    # Job metadata
    job_name: str = Field(..., description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    
    # Trigger information
    trigger_type: RetrainingTriggerType = Field(..., description="Type of trigger")
    trigger_reason: Optional[str] = Field(None, description="Reason for retraining")
    triggered_by: Optional[str] = Field(None, description="User or system that triggered")
    
    # Job configuration
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    data_source: Optional[str] = Field(None, description="Training data source")
    data_window_start: Optional[datetime] = Field(None, description="Data window start")
    data_window_end: Optional[datetime] = Field(None, description="Data window end")
    
    # Job status
    status: RetrainingJobStatus = Field(RetrainingJobStatus.PENDING, description="Job status")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled time")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    failed_at: Optional[datetime] = Field(None, description="Failure time")
    
    # Retraining results
    new_model_id: Optional[str] = Field(None, description="Newly trained model ID")
    performance_comparison: Dict[str, Any] = Field(default_factory=dict, description="Performance comparison")
    improvement_detected: bool = Field(False, description="Whether improvement was detected")
    auto_replace_enabled: bool = Field(False, description="Whether auto-replacement is enabled")
    replacement_status: Optional[ReplacementStatus] = Field(None, description="Replacement status")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message")
    error_traceback: Optional[str] = Field(None, description="Error traceback")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class RetrainingTriggerConfig(BaseModel):
    """Schema for retraining trigger configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique config ID")
    model_id: str = Field(..., description="Model ID")
    
    # Trigger configuration
    trigger_type: RetrainingTriggerType = Field(..., description="Type of trigger")
    is_enabled: bool = Field(True, description="Whether trigger is enabled")
    
    # Performance degradation trigger
    performance_threshold: Optional[float] = Field(None, description="Performance threshold")
    performance_metric: Optional[str] = Field(None, description="Performance metric name")
    degradation_window_hours: Optional[int] = Field(None, description="Degradation window in hours", ge=1)
    
    # Drift trigger
    drift_threshold: Optional[float] = Field(None, description="Drift score threshold", ge=0, le=1)
    drift_type: Optional[str] = Field(None, description="Drift type")
    
    # Time-based trigger
    retraining_interval_days: Optional[int] = Field(None, description="Days between retraining", ge=1)
    last_retraining_date: Optional[datetime] = Field(None, description="Last retraining date")
    
    # Data volume trigger
    min_data_samples: Optional[int] = Field(None, description="Minimum samples for retraining", ge=1)
    data_window_days: Optional[int] = Field(None, description="Data collection window in days", ge=1)
    
    # Auto-replacement settings
    auto_replace_on_improvement: bool = Field(False, description="Auto-replace on improvement")
    min_improvement_threshold: Optional[float] = Field(None, description="Minimum improvement threshold", ge=0)
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created config")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class ModelCard(BaseModel):
    """Schema for model card"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique card ID")
    model_id: str = Field(..., description="Model ID")
    model_version: Optional[str] = Field(None, description="Model version")
    
    # Model card content
    card_content: Dict[str, Any] = Field(..., description="Full model card JSON")
    card_version: str = Field("1.0", description="Card version")
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    # Training information
    training_data_info: Dict[str, Any] = Field(default_factory=dict, description="Training dataset information")
    training_date: Optional[datetime] = Field(None, description="Training date")
    training_duration_hours: Optional[float] = Field(None, description="Training duration", ge=0)
    training_parameters: Dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    
    # Model information
    model_architecture: Optional[str] = Field(None, description="Model architecture description")
    model_limitations: Optional[str] = Field(None, description="Model limitations")
    usage_guidelines: Optional[str] = Field(None, description="Usage guidelines")
    ethical_considerations: Optional[str] = Field(None, description="Ethical considerations")
    
    # Evaluation information
    evaluation_metrics: Dict[str, Any] = Field(default_factory=dict, description="Evaluation metrics")
    evaluation_dataset_info: Dict[str, Any] = Field(default_factory=dict, description="Evaluation dataset info")
    evaluation_date: Optional[datetime] = Field(None, description="Evaluation date")
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Tags")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created card")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

