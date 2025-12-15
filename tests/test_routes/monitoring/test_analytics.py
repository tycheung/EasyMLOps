"""
Comprehensive tests for monitoring routes
Tests all monitoring REST API endpoints including health checks, metrics, logs, and alerts
"""

import pytest
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status, HTTPException
from fastapi.testclient import TestClient

from app.schemas.monitoring import (
    ModelPerformanceMetrics, 
    PredictionLog, 
    SystemHealthStatus, 
    SystemComponent,
    SystemStatus,
    SystemHealthMetric,
    AlertSeverity,
    MetricType,
    ModelDriftDetection,
    DriftType,
    DriftSeverity,
    Alert,
    AlertRule,
    ModelPerformanceHistory,
    ModelBaseline,
    ModelVersionComparison,
    ABTest,
    ABTestMetrics,
    ABTestStatus,
    CanaryDeployment,
    CanaryDeploymentStatus,
    CanaryMetrics,
    ModelExplanation,
    ExplanationType,
    OutlierDetection,
    AnomalyDetection,
    DataQualityMetrics,
    BiasFairnessMetrics
)
from app.config import get_settings

settings = get_settings()


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics data"""
    return {
        "model_id": "test_model_123",
        "time_window_start": datetime.utcnow() - timedelta(hours=1),
        "time_window_end": datetime.utcnow(),
        "total_requests": 1500,
        "successful_requests": 1485,
        "failed_requests": 15,
        "requests_per_minute": 25.5,
        "avg_latency_ms": 45.2,
        "p50_latency_ms": 42.1,
        "p95_latency_ms": 78.5,
        "p99_latency_ms": 95.2,
        "p99_9_latency_ms": 110.0,
        "min_latency_ms": 20.0,
        "max_latency_ms": 120.5,
        "std_dev_latency_ms": 15.3,
        "latency_distribution": {
            "bins": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            "counts": [10, 20, 30, 40, 25, 15, 5, 3, 1, 1],
            "bin_width": 10.0,
            "total_samples": 1500
        },
        "success_rate": 99.0,
        "error_rate": 1.0
    }


@pytest.fixture
def sample_system_health():
    """Sample system health metrics, conforming to SystemHealthStatus schema"""
    return SystemHealthStatus(
        overall_status=SystemStatus.OPERATIONAL,
        components=[
            SystemHealthMetric(component=SystemComponent.API_SERVER, status=SystemStatus.OPERATIONAL, message="API server is responsive.", metric_type=MetricType.CPU_USAGE, value=45.2, unit="percent"),
            SystemHealthMetric(component=SystemComponent.DATABASE, status=SystemStatus.OPERATIONAL, message="Database connection healthy.")
        ],
        last_check=datetime.utcnow()
    )


@pytest.fixture
def sample_unhealthy_system_health():
    """Sample unhealthy system health metrics, conforming to SystemHealthStatus schema"""
    return SystemHealthStatus(
        overall_status=SystemStatus.UNHEALTHY,
        components=[
            SystemHealthMetric(component=SystemComponent.API_SERVER, status=SystemStatus.UNHEALTHY, message="API server CPU high.", metric_type=MetricType.CPU_USAGE, value=95.5, unit="percent"),
            SystemHealthMetric(component=SystemComponent.DATABASE, status=SystemStatus.UNHEALTHY, message="Database connection failed.")
        ],
        last_check=datetime.utcnow()
    )



# Tests for Analytics monitoring routes

class TestAnalyticsAdvanced:
    """Test advanced analytics endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_comparative_analytics', new_callable=AsyncMock)
    def test_create_comparative_analytics(self, mock_create, client):
        """Test creating comparative analytics"""
        mock_create.return_value = "analytics_123"
        
        analytics_data = {
            "comparison_type": "model_comparison",
            "comparison_name": "Model A vs Model B",
            "entity_ids": ["model_a", "model_b"],
            "entity_types": ["model", "model"],
            "comparison_metrics": {"accuracy": True, "latency": True},
            "comparison_results": {"accuracy": {"model_a": 0.9, "model_b": 0.85}},
            "time_window_start": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "time_window_end": datetime.utcnow().isoformat()
        }
        
        response = client.post(
            "/api/v1/monitoring/analytics/comparative",
            json=analytics_data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "analytics_123"
        mock_create.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.create_custom_dashboard', new_callable=AsyncMock)
    def test_create_custom_dashboard(self, mock_create, client):
        """Test creating custom dashboard"""
        mock_create.return_value = "dashboard_123"
        
        dashboard_data = {
            "dashboard_name": "My Custom Dashboard",
            "dashboard_config": {"layout": "grid"},
            "selected_metrics": ["accuracy", "latency"]
        }
        
        response = client.post(
            "/api/v1/monitoring/analytics/dashboards",
            json=dashboard_data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "dashboard_123"
        mock_create.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.create_automated_report', new_callable=AsyncMock)
    def test_create_automated_report(self, mock_create, client):
        """Test creating automated report"""
        mock_create.return_value = "report_123"
        
        report_data = {
            "report_name": "Weekly Performance Report",
            "report_type": "weekly",
            "schedule_type": "weekly",
            "report_config": {"format": "pdf"},
            "delivery_method": ["email"],
            "recipients": ["admin@example.com"]
        }
        
        response = client.post(
            "/api/v1/monitoring/analytics/reports",
            json=report_data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "report_123"
        mock_create.assert_called_once()




