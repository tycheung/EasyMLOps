"""
Monitoring and operations schemas for EasyMLOps platform
Defines data validation schemas for performance monitoring, system health, and audit trails
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class MetricType(str, Enum):
    """Types of metrics that can be tracked"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    DISK_USAGE_MODELS = "disk_usage_models"
    REQUEST_COUNT = "request_count"
    SUCCESS_RATE = "success_rate"
    DB_CONNECTION_STATUS = "db_connection_status"
    BENTOML_SERVICE_STATUS = "bentoml_service_status"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemComponent(str, Enum):
    """System components that can be monitored"""
    API_SERVER = "api_server"
    DATABASE = "database"
    MODEL_SERVICE = "model_service"
    BENTOML = "bentoml"
    STORAGE = "storage"
    SYSTEM = "system"


class SystemStatus(str, Enum):
    """Overall system or component status"""
    OPERATIONAL = "operational"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


# Model Performance Monitoring Schemas
class PredictionLog(BaseModel):
    """Schema for logging individual predictions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique prediction log ID")
    model_id: str = Field(..., description="Model ID that made the prediction")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request ID")
    
    # Request details
    input_data: Dict[str, Any] = Field(..., description="Input data sent to model")
    output_data: Any = Field(..., description="Model prediction output")
    
    # Performance metrics
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    # Request metadata
    user_agent: Optional[str] = Field(None, description="Client user agent")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    api_endpoint: str = Field(..., description="API endpoint used")
    
    # Status information
    success: bool = Field(..., description="Whether prediction was successful")
    error_message: Optional[str] = Field(None, description="Error message if prediction failed")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "input_data": {"bedrooms": 3, "bathrooms": 2.5, "sqft": 2000},
                "output_data": {"prediction": 425000.0, "confidence": 0.87},
                "latency_ms": 45.2,
                "api_endpoint": "/models/abc123-def456/predict",
                "success": True
            }
        }
    }


class ModelPerformanceMetrics(BaseModel):
    """Schema for aggregated model performance metrics"""
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    time_window_start: datetime = Field(..., description="Start of measurement window")
    time_window_end: datetime = Field(..., description="End of measurement window")
    
    # Throughput metrics
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    requests_per_minute: float = Field(..., description="Average requests per minute")
    
    # Latency metrics
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    p50_latency_ms: float = Field(..., description="50th percentile latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    max_latency_ms: float = Field(..., description="Maximum latency")
    
    # Quality metrics
    success_rate: float = Field(..., description="Success rate percentage", ge=0, le=100)
    error_rate: float = Field(..., description="Error rate percentage", ge=0, le=100)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "total_requests": 1500,
                "successful_requests": 1485,
                "failed_requests": 15,
                "requests_per_minute": 25.5,
                "avg_latency_ms": 45.2,
                "p95_latency_ms": 78.5,
                "success_rate": 99.0,
                "error_rate": 1.0
            }
        }
    }


# System Health Monitoring Schemas
class SystemHealthMetric(BaseModel):
    """Schema for individual system component health metrics or status"""
    component: SystemComponent = Field(..., description="System component")
    status: SystemStatus = Field(..., description="Status of the component")
    message: str = Field(..., description="Health message or details")
    
    # Optional fields for specific metrics if this schema is also used for that
    metric_type: Optional[MetricType] = Field(None, description="Type of metric, if applicable")
    value: Optional[float] = Field(None, description="Metric value, if applicable")
    unit: Optional[str] = Field(None, description="Unit of measurement, if applicable")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
    host: Optional[str] = Field(None, description="Host/server name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Additional metric tags")

    model_config = {
        "json_schema_extra": {
            "example": {
                "component": "api_server",
                "status": "operational",
                "message": "API server is responsive.",
                "metric_type": "cpu_usage",
                "value": 45.5,
                "unit": "percent",
                "host": "easymlops-server-01",
                "tags": {"environment": "production", "region": "us-east-1"}
            }
        }
    }


class SystemHealthStatus(BaseModel):
    """Schema for overall system health status"""
    overall_status: SystemStatus = Field(..., description="Overall system status")
    components: List[SystemHealthMetric] = Field(..., description="Status of individual components")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "overall_status": "operational",
                "components": [
                    {"component": "api_server", "status": "operational", "message": "API server is responsive."},
                    {"component": "database", "status": "unhealthy", "message": "Database connection failed."}
                ]
            }
        }
    }


# Alert Management Schemas
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


# Audit & Logging Schemas
class AuditLog(BaseModel):
    """Schema for audit trail logging"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique audit log ID")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    session_id: Optional[str] = Field(None, description="Session ID")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="ID of affected resource")
    
    # Action details
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When action was performed")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    
    # Change tracking
    old_values: Optional[Dict[str, Any]] = Field(None, description="Previous values before change")
    new_values: Optional[Dict[str, Any]] = Field(None, description="New values after change")
    
    # Additional context
    success: bool = Field(..., description="Whether action was successful")
    error_message: Optional[str] = Field(None, description="Error message if action failed")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user123",
                "action": "model_deployed",
                "resource_type": "model",
                "resource_id": "abc123-def456",
                "ip_address": "192.168.1.100",
                "new_values": {"status": "deployed", "endpoint": "/predict/abc123"},
                "success": True
            }
        }
    }


# Analytics and Reporting Schemas
class ModelUsageAnalytics(BaseModel):
    """Schema for model usage analytics"""
    model_id: str = Field(..., description="Model ID")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")
    
    # Usage statistics
    total_predictions: int = Field(..., description="Total number of predictions")
    unique_users: int = Field(..., description="Number of unique users")
    avg_daily_requests: float = Field(..., description="Average requests per day")
    peak_requests_per_hour: int = Field(..., description="Peak requests in a single hour")
    
    # Performance trends
    avg_response_time_trend: List[float] = Field(..., description="Response time trend over period")
    error_rate_trend: List[float] = Field(..., description="Error rate trend over period")
    usage_pattern: Dict[str, int] = Field(..., description="Usage patterns by hour/day")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "abc123-def456",
                "total_predictions": 50000,
                "unique_users": 125,
                "avg_daily_requests": 1667,
                "peak_requests_per_hour": 450,
                "usage_pattern": {"monday": 8500, "tuesday": 7200, "wednesday": 9100}
            }
        }
    }


# Dashboard and Reporting Schemas
class DashboardMetrics(BaseModel):
    """Schema for dashboard overview metrics"""
    
    # Overall platform metrics
    total_models: int = Field(..., description="Total number of models")
    active_deployments: int = Field(..., description="Number of active deployments")
    total_predictions_today: int = Field(..., description="Total predictions made today")
    avg_response_time_today: float = Field(..., description="Average response time today")
    
    # System health summary
    system_status: str = Field(..., description="Overall system status")
    active_alerts: int = Field(..., description="Number of active alerts")
    cpu_usage: float = Field(..., description="Current CPU usage percentage")
    memory_usage: float = Field(..., description="Current memory usage percentage")
    
    # Model performance summary
    most_used_model: Optional[Dict[str, Any]] = Field(None, description="Most frequently used model")
    fastest_model: Optional[Dict[str, Any]] = Field(None, description="Fastest responding model")
    recent_deployments: List[Dict[str, Any]] = Field(..., description="Recently deployed models")
    
    # Trend data
    request_trend_24h: List[int] = Field(..., description="Request count trend over last 24 hours")
    error_trend_24h: List[float] = Field(..., description="Error rate trend over last 24 hours")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last metrics update")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_models": 15,
                "active_deployments": 12,
                "total_predictions_today": 5420,
                "avg_response_time_today": 42.5,
                "system_status": "healthy",
                "active_alerts": 1,
                "cpu_usage": 45.5,
                "memory_usage": 62.3,
                "request_trend_24h": [120, 135, 98, 156, 189, 210, 195],
                "error_trend_24h": [0.5, 0.8, 0.3, 1.2, 0.9, 0.6, 0.4]
            }
        }
    } 