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



# Tests for Health monitoring routes

class TestSystemHealth:
    """Test system health monitoring endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health_status', new_callable=AsyncMock)
    def test_get_system_health_success(self, mock_health, client, sample_system_health: SystemHealthStatus):
        """Test successful system health retrieval"""
        mock_health.return_value = sample_system_health
        
        response = client.get(f"{settings.API_V1_PREFIX}/monitoring/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["overall_status"] == SystemStatus.OPERATIONAL.value
        assert len(result["components"]) > 0
        assert result["components"][0]["component"] == SystemComponent.API_SERVER.value
        assert result["components"][0]["status"] == SystemStatus.OPERATIONAL.value
        mock_health.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health_status', new_callable=AsyncMock)
    def test_get_system_health_unhealthy(self, mock_health, client, sample_unhealthy_system_health: SystemHealthStatus):
        """Test system health when system is unhealthy"""
        mock_health.return_value = sample_unhealthy_system_health
        
        response = client.get(f"{settings.API_V1_PREFIX}/monitoring/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["overall_status"] == SystemStatus.UNHEALTHY.value
        assert any(c["status"] == SystemStatus.UNHEALTHY.value for c in result["components"])
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health_status', new_callable=AsyncMock)
    def test_get_system_health_service_error(self, mock_health, client):
        """Test system health endpoint when service fails"""
        mock_health.side_effect = Exception("Health check failed")
        
        response = client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 500




class TestSystemHealthAdvanced:
    """Test advanced system health endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.collect_model_resource_usage', new_callable=AsyncMock)
    def test_get_model_resource_usage(self, mock_collect, client):
        """Test collecting model resource usage"""
        mock_usage = {
            "model_id": "test_model",
            "cpu_usage": 45.2,
            "memory_usage": 60.5,
            "gpu_usage": 30.0
        }
        mock_collect.return_value = mock_usage
        
        response = client.get("/api/v1/monitoring/models/test_model/resources")
        
        assert response.status_code == 200
        result = response.json()
        assert result["cpu_usage"] == 45.2
        mock_collect.assert_called_once()




