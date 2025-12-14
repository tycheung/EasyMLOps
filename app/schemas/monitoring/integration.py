"""
Integration and scalability schemas
Contains schemas for external integrations, webhooks, sampling, and metric aggregation
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class IntegrationType(str, Enum):
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEWRELIC = "newrelic"
    MLFLOW = "mlflow"
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


class ExternalIntegration(BaseModel):
    """Schema for external integration configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique integration ID")
    
    # Integration metadata
    integration_type: IntegrationType = Field(..., description="Type of integration")
    integration_name: str = Field(..., description="Integration name")
    description: Optional[str] = Field(None, description="Integration description")
    
    # Configuration
    config: Dict[str, Any] = Field(..., description="Integration-specific configuration")
    endpoint_url: Optional[str] = Field(None, description="Endpoint URL")
    api_key: Optional[str] = Field(None, description="API key")
    auth_config: Dict[str, Any] = Field(default_factory=dict, description="Authentication configuration")
    
    # Status
    is_active: bool = Field(True, description="Whether integration is active")
    last_successful_sync: Optional[datetime] = Field(None, description="Last successful sync")
    last_sync_error: Optional[str] = Field(None, description="Last sync error")
    sync_frequency_seconds: Optional[int] = Field(None, description="Sync frequency", ge=1)
    
    # Scope
    model_ids: List[str] = Field(default_factory=list, description="Model IDs to sync")
    metric_types: List[str] = Field(default_factory=list, description="Metric types to sync")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created integration")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class WebhookConfig(BaseModel):
    """Schema for webhook configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique webhook ID")
    
    # Webhook metadata
    webhook_name: str = Field(..., description="Webhook name")
    webhook_url: str = Field(..., description="Webhook URL")
    description: Optional[str] = Field(None, description="Webhook description")
    
    # Event triggers
    trigger_events: List[str] = Field(..., description="Events that trigger webhook")
    
    # Configuration
    http_method: str = Field("POST", description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    payload_template: Optional[str] = Field(None, description="Custom payload template")
    timeout_seconds: int = Field(30, description="Request timeout", ge=1, le=300)
    retry_config: Dict[str, Any] = Field(default_factory=dict, description="Retry configuration")
    
    # Security
    secret_key: Optional[str] = Field(None, description="Webhook signature secret")
    auth_type: Optional[str] = Field(None, description="Authentication type")
    auth_config: Dict[str, Any] = Field(default_factory=dict, description="Authentication configuration")
    
    # Status
    is_active: bool = Field(True, description="Whether webhook is active")
    last_triggered_at: Optional[datetime] = Field(None, description="Last trigger time")
    last_success: Optional[bool] = Field(None, description="Last request success")
    last_error: Optional[str] = Field(None, description="Last error message")
    total_triggers: int = Field(0, description="Total triggers", ge=0)
    success_count: int = Field(0, description="Success count", ge=0)
    failure_count: int = Field(0, description="Failure count", ge=0)
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created webhook")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class SamplingStrategy(str, Enum):
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    ADAPTIVE = "adaptive"


class SamplingConfig(BaseModel):
    """Schema for sampling configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique config ID")
    
    # Sampling metadata
    config_name: str = Field(..., description="Config name")
    resource_type: str = Field(..., description="Resource type")
    
    # Sampling strategy
    sampling_strategy: SamplingStrategy = Field(..., description="Sampling strategy")
    sampling_rate: float = Field(..., description="Sampling rate", ge=0.0, le=1.0)
    
    # Conditions
    min_volume_threshold: Optional[int] = Field(None, description="Min volume threshold", ge=1)
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Additional conditions")
    
    # Scope
    model_id: Optional[str] = Field(None, description="Model ID (null = all)")
    deployment_id: Optional[str] = Field(None, description="Deployment ID (null = all)")
    
    # Status
    is_active: bool = Field(True, description="Whether config is active")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created config")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class AggregationMethod(str, Enum):
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    PERCENTILE = "percentile"


class MetricAggregationConfig(BaseModel):
    """Schema for metric aggregation configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique config ID")
    
    # Aggregation metadata
    config_name: str = Field(..., description="Config name")
    metric_type: str = Field(..., description="Metric type")
    
    # Aggregation settings
    aggregation_window_seconds: int = Field(..., description="Aggregation window", ge=1)
    aggregation_method: AggregationMethod = Field(..., description="Aggregation method")
    percentile: Optional[float] = Field(None, description="Percentile for percentile method", ge=0, le=100)
    
    # Retention
    raw_data_retention_hours: Optional[int] = Field(None, description="Raw data retention", ge=1)
    aggregated_data_retention_days: Optional[int] = Field(None, description="Aggregated data retention", ge=1)
    
    # Scope
    model_id: Optional[str] = Field(None, description="Model ID (null = all)")
    
    # Status
    is_active: bool = Field(True, description="Whether config is active")
    last_aggregation_at: Optional[datetime] = Field(None, description="Last aggregation time")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created config")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

