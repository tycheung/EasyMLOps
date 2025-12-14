"""
Advanced analytics and reporting schemas
Contains schemas for time-series analysis, comparative analytics, dashboards, and automated reports
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class AnalysisType(str, Enum):
    TREND = "trend"
    SEASONALITY = "seasonality"
    FORECAST = "forecast"
    ANOMALY = "anomaly"


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class TimeSeriesAnalysis(BaseModel):
    """Schema for time-series analysis results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique analysis ID")
    model_id: Optional[str] = Field(None, description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    
    # Analysis metadata
    analysis_type: AnalysisType = Field(..., description="Type of analysis")
    metric_name: str = Field(..., description="Metric being analyzed")
    
    # Time window
    time_window_start: datetime = Field(..., description="Analysis window start")
    time_window_end: datetime = Field(..., description="Analysis window end")
    
    # Analysis results
    trend_direction: Optional[TrendDirection] = Field(None, description="Trend direction")
    trend_slope: Optional[float] = Field(None, description="Trend slope")
    trend_strength: Optional[float] = Field(None, description="Trend strength (R²)", ge=0, le=1)
    seasonality_detected: bool = Field(False, description="Whether seasonality was detected")
    seasonality_period: Optional[str] = Field(None, description="Seasonality period")
    seasonality_strength: Optional[float] = Field(None, description="Seasonality strength", ge=0, le=1)
    
    # Forecasting
    forecast_values: List[float] = Field(default_factory=list, description="Forecasted values")
    forecast_confidence_intervals: Dict[str, List[float]] = Field(default_factory=dict, description="Confidence intervals")
    forecast_horizon: Optional[int] = Field(None, description="Forecast horizon (periods ahead)", ge=1)
    
    # Anomaly detection
    anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    anomaly_count: int = Field(0, description="Number of anomalies", ge=0)
    
    # Analysis data
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw time-series data")
    processed_data: Dict[str, Any] = Field(default_factory=dict, description="Processed data")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class ComparisonType(str, Enum):
    MODEL_COMPARISON = "model_comparison"
    BENCHMARK = "benchmark"
    COST_BENEFIT = "cost_benefit"
    ROI = "roi"


class ComparativeAnalytics(BaseModel):
    """Schema for comparative analytics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique comparison ID")
    
    # Comparison metadata
    comparison_type: ComparisonType = Field(..., description="Type of comparison")
    comparison_name: str = Field(..., description="Comparison name")
    
    # Entities being compared
    entity_ids: List[str] = Field(..., description="IDs of entities being compared")
    entity_types: List[str] = Field(..., description="Types of entities")
    entity_names: List[str] = Field(default_factory=list, description="Names for display")
    
    # Comparison metrics
    comparison_metrics: Dict[str, Any] = Field(..., description="Metrics being compared")
    comparison_results: Dict[str, Any] = Field(..., description="Comparison results")
    winner_id: Optional[str] = Field(None, description="ID of best performing entity")
    winner_reason: Optional[str] = Field(None, description="Reason for winner selection")
    
    # Cost-benefit analysis
    cost_data: Dict[str, Any] = Field(default_factory=dict, description="Cost information")
    benefit_data: Dict[str, Any] = Field(default_factory=dict, description="Benefit information")
    roi_calculation: Dict[str, Any] = Field(default_factory=dict, description="ROI calculations")
    
    # Benchmarking
    benchmark_metrics: Dict[str, Any] = Field(default_factory=dict, description="Benchmark values")
    benchmark_source: Optional[str] = Field(None, description="Source of benchmark")
    
    # Time window
    time_window_start: datetime = Field(..., description="Comparison window start")
    time_window_end: datetime = Field(..., description="Comparison window end")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created comparison")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class CustomDashboard(BaseModel):
    """Schema for custom dashboards"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique dashboard ID")
    
    # Dashboard metadata
    dashboard_name: str = Field(..., description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    
    # Dashboard configuration
    dashboard_config: Dict[str, Any] = Field(..., description="Dashboard layout and widgets")
    selected_metrics: List[str] = Field(..., description="Selected metrics")
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="Visualization settings")
    
    # Sharing
    is_shared: bool = Field(False, description="Whether dashboard is shared")
    shared_with: List[str] = Field(default_factory=list, description="Users dashboard is shared with")
    share_token: Optional[str] = Field(None, description="Token for public sharing")
    
    # Real-time settings
    auto_refresh_enabled: bool = Field(False, description="Whether auto-refresh is enabled")
    refresh_interval_seconds: Optional[int] = Field(None, description="Refresh interval", ge=1)
    
    # Filters
    filters: Dict[str, Any] = Field(default_factory=dict, description="Dashboard filters")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created dashboard")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class ReportType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    PERFORMANCE = "performance"
    DRIFT = "drift"
    COMPLIANCE = "compliance"


class ScheduleType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class AutomatedReport(BaseModel):
    """Schema for automated reports"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique report ID")
    
    # Report metadata
    report_name: str = Field(..., description="Report name")
    report_type: ReportType = Field(..., description="Type of report")
    description: Optional[str] = Field(None, description="Report description")
    
    # Schedule
    schedule_type: ScheduleType = Field(..., description="Schedule type")
    schedule_config: Dict[str, Any] = Field(default_factory=dict, description="Schedule configuration")
    next_run_at: Optional[datetime] = Field(None, description="Next scheduled run time")
    last_run_at: Optional[datetime] = Field(None, description="Last run time")
    
    # Report content
    report_config: Dict[str, Any] = Field(..., description="Report configuration")
    included_metrics: List[str] = Field(default_factory=list, description="Metrics to include")
    included_models: List[str] = Field(default_factory=list, description="Models to include")
    time_window_days: Optional[int] = Field(None, description="Time window for report", ge=1)
    
    # Delivery
    delivery_method: List[str] = Field(..., description="Delivery methods: email, slack, webhook")
    recipients: List[str] = Field(..., description="Recipients")
    email_template: Optional[str] = Field(None, description="Email template")
    slack_webhook: Optional[str] = Field(None, description="Slack webhook URL")
    
    # Status
    is_active: bool = Field(True, description="Whether report is active")
    last_report_data: Dict[str, Any] = Field(default_factory=dict, description="Last generated report data")
    last_report_location: Optional[str] = Field(None, description="Location of last report")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created report")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

