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



# Tests for Canary monitoring routes

class TestCanaryDeployment:
    """Test canary deployment endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_canary_deployment', new_callable=AsyncMock)
    def test_create_canary_deployment(self, mock_create, client):
        """Test creating a canary deployment"""
        from app.schemas.monitoring import CanaryDeployment, CanaryDeploymentStatus
        
        mock_canary = CanaryDeployment(
            deployment_name="canary_v2",
            model_id="model_v1",
            production_deployment_id="model_v1",
            canary_deployment_id="model_v2",
            current_traffic_percentage=10.0,
            target_traffic_percentage=50.0,
            status=CanaryDeploymentStatus.PENDING
        )
        mock_create.return_value = mock_canary
        
        canary_data = {
            "deployment_name": "canary_v2",
            "model_id": "model_v1",
            "production_deployment_id": "model_v1",
            "canary_deployment_id": "model_v2",
            "current_traffic_percentage": 10.0,
            "target_traffic_percentage": 50.0
        }
        
        response = client.post("/api/v1/monitoring/canary", json=canary_data)
        
        assert response.status_code == 201
        result = response.json()
        assert result["deployment_name"] == "canary_v2"
        mock_create.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.start_canary_rollout', new_callable=AsyncMock)
    def test_start_canary_rollout(self, mock_start, client):
        """Test starting canary rollout"""
        mock_start.return_value = True
        
        response = client.post("/api/v1/monitoring/canary/canary_123/start")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_start.assert_called_once_with(canary_id="canary_123")
    
    @patch('app.services.monitoring_service.monitoring_service.advance_canary_rollout', new_callable=AsyncMock)
    def test_advance_canary_rollout(self, mock_advance, client):
        """Test advancing canary rollout"""
        mock_advance.return_value = True
        
        response = client.post("/api/v1/monitoring/canary/canary_123/advance")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_advance.assert_called_once_with(canary_id="canary_123")
    
    @patch('app.services.monitoring_service.monitoring_service.rollback_canary', new_callable=AsyncMock)
    def test_rollback_canary(self, mock_rollback, client):
        """Test rolling back canary"""
        mock_rollback.return_value = True
        
        response = client.post("/api/v1/monitoring/canary/canary_123/rollback")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_rollback.assert_called_once_with(canary_id="canary_123")




class TestCanaryAdvanced:
    """Test advanced canary endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.check_canary_health', new_callable=AsyncMock)
    def test_check_canary_health(self, mock_check, client):
        """Test checking canary health"""
        mock_check.return_value = (True, "Healthy", None)
        
        response = client.get("/api/v1/monitoring/canary/canary_123/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["is_healthy"] is True
        assert result["status_message"] == "Healthy"
        mock_check.assert_called_once_with(canary_id="canary_123")




