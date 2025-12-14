"""
Alert and notification schemas
Contains schemas for alert rules, notifications, and alert management
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid

from app.schemas.monitoring.base import AlertSeverity, SystemComponent


class AlertCondition(str, Enum):
    GT = "gt"  # Greater than
    LT = "lt"  # Less than
    EQ = "eq"  # Equal to
    GTE = "gte"  # Greater than or equal
    LTE = "lte"  # Less than or equal
    BETWEEN = "between"  # Between two values
    NOT_BETWEEN = "not_between"  # Not between two values


class AlertRule(BaseModel):
    """Schema for custom alert rule"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique rule ID")
    
    # Rule metadata
    rule_name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    
    # Rule configuration
    metric_name: str = Field(..., description="Metric to monitor")
    condition: AlertCondition = Field(..., description="Condition type")
    threshold_value: Optional[float] = Field(None, description="Threshold value")
    threshold_min: Optional[float] = Field(None, description="Min threshold (for between)")
    threshold_max: Optional[float] = Field(None, description="Max threshold (for between)")
    
    # Alert configuration
    severity: AlertSeverity = Field(AlertSeverity.WARNING, description="Alert severity")
    component: SystemComponent = Field(..., description="System component")
    
    # Frequency controls
    min_interval_seconds: Optional[int] = Field(None, description="Min interval between alerts", ge=1)
    max_alerts_per_hour: Optional[int] = Field(None, description="Max alerts per hour", ge=1)
    cooldown_period_seconds: Optional[int] = Field(None, description="Cooldown period", ge=1)
    
    # Scope
    model_id: Optional[str] = Field(None, description="Model ID (null = all)")
    deployment_id: Optional[str] = Field(None, description="Deployment ID (null = all)")
    
    # Status
    is_active: bool = Field(True, description="Whether rule is active")
    last_triggered_at: Optional[datetime] = Field(None, description="Last trigger time")
    trigger_count: int = Field(0, description="Total trigger count", ge=0)
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created rule")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class NotificationChannelType(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"


class NotificationChannel(BaseModel):
    """Schema for notification channel configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique channel ID")
    
    # Channel metadata
    channel_name: str = Field(..., description="Channel name")
    channel_type: NotificationChannelType = Field(..., description="Channel type")
    description: Optional[str] = Field(None, description="Channel description")
    
    # Configuration
    config: Dict[str, Any] = Field(..., description="Channel-specific configuration")
    
    # Email configuration
    email_recipients: List[str] = Field(default_factory=list, description="Email recipients")
    email_template: Optional[str] = Field(None, description="Email template")
    
    # Slack configuration
    slack_webhook_url: Optional[str] = Field(None, description="Slack webhook URL")
    slack_channel: Optional[str] = Field(None, description="Slack channel")
    
    # PagerDuty configuration
    pagerduty_integration_key: Optional[str] = Field(None, description="PagerDuty integration key")
    pagerduty_service_id: Optional[str] = Field(None, description="PagerDuty service ID")
    
    # Webhook configuration
    webhook_url: Optional[str] = Field(None, description="Webhook URL")
    webhook_headers: Dict[str, str] = Field(default_factory=dict, description="Webhook headers")
    
    # SMS configuration
    sms_provider: Optional[str] = Field(None, description="SMS provider")
    sms_recipients: List[str] = Field(default_factory=list, description="SMS recipients")
    sms_config: Dict[str, Any] = Field(default_factory=dict, description="SMS configuration")
    
    # Alert filtering
    severity_filter: List[str] = Field(default_factory=list, description="Severity filter")
    component_filter: List[str] = Field(default_factory=list, description="Component filter")
    
    # Status
    is_active: bool = Field(True, description="Whether channel is active")
    last_sent_at: Optional[datetime] = Field(None, description="Last sent time")
    total_sent: int = Field(0, description="Total sent count", ge=0)
    failure_count: int = Field(0, description="Failure count", ge=0)
    last_error: Optional[str] = Field(None, description="Last error message")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created channel")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class AlertGroup(BaseModel):
    """Schema for alert grouping"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique group ID")
    
    # Group metadata
    group_key: str = Field(..., description="Group key")
    group_type: str = Field(..., description="Group type")
    
    # Grouped alerts
    alert_ids: List[str] = Field(..., description="Alert IDs in group")
    alert_count: int = Field(0, description="Alert count", ge=0)
    
    # Group status
    is_active: bool = Field(True, description="Whether group is active")
    first_alert_at: datetime = Field(..., description="First alert time")
    last_alert_at: datetime = Field(..., description="Last alert time")
    
    # Resolution
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")
    resolved_by: Optional[str] = Field(None, description="Who resolved")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class EscalationTriggerCondition(str, Enum):
    UNACKNOWLEDGED_DURATION = "unacknowledged_duration"
    SEVERITY = "severity"
    COUNT = "count"


class AlertEscalation(BaseModel):
    """Schema for alert escalation rule"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique escalation ID")
    
    # Escalation metadata
    escalation_name: str = Field(..., description="Escalation name")
    description: Optional[str] = Field(None, description="Escalation description")
    
    # Escalation rules
    trigger_condition: EscalationTriggerCondition = Field(..., description="Trigger condition")
    trigger_value: Dict[str, Any] = Field(..., description="Condition value")
    
    # Escalation actions
    escalation_actions: List[str] = Field(..., description="Actions to take")
    notification_channels: List[str] = Field(default_factory=list, description="Additional channels")
    
    # Scope
    severity_filter: List[str] = Field(default_factory=list, description="Severity filter")
    component_filter: List[str] = Field(default_factory=list, description="Component filter")
    
    # Status
    is_active: bool = Field(True, description="Whether escalation is active")
    escalation_count: int = Field(0, description="Escalation count", ge=0)
    last_escalated_at: Optional[datetime] = Field(None, description="Last escalation time")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created escalation")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class Alert(BaseModel):
    """Schema for system alerts"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique alert ID")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    component: SystemComponent = Field(..., description="Component that triggered alert")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed alert description")
    
    # Alert metadata
    triggered_at: datetime = Field(default_factory=datetime.utcnow, description="When alert was triggered")
    resolved_at: Optional[datetime] = Field(None, description="When alert was resolved")
    acknowledged_at: Optional[datetime] = Field(None, description="When alert was acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged the alert")
    
    # Alert context
    metric_value: Optional[float] = Field(None, description="Metric value that triggered alert")
    threshold_value: Optional[float] = Field(None, description="Threshold that was exceeded")
    affected_models: List[str] = Field(default_factory=list, description="Models affected by this alert")
    
    # Status
    is_active: bool = Field(True, description="Whether alert is currently active")
    is_acknowledged: bool = Field(False, description="Whether alert has been acknowledged")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "severity": "warning",
                "component": "api_server",
                "title": "High CPU Usage",
                "description": "API server CPU usage exceeded 80% for 5 minutes",
                "metric_value": 85.5,
                "threshold_value": 80.0,
                "is_active": True,
                "is_acknowledged": False
            }
        }
    }

