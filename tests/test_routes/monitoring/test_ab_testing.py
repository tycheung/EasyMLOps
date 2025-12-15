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



# Tests for Ab Testing monitoring routes

class TestABTesting:
    """Test A/B testing endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_ab_test', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_ab_test', new_callable=AsyncMock)
    def test_create_ab_test(self, mock_store, mock_create, client):
        """Test creating an A/B test"""
        from app.schemas.monitoring import ABTest, ABTestStatus
        
        mock_test = ABTest(
            test_name="model_v2_test",
            model_name="test_model",
            variant_a_model_id="model_v1",
            variant_b_model_id="model_v2",
            variant_a_percentage=50.0,
            variant_b_percentage=50.0,
            status=ABTestStatus.DRAFT
        )
        mock_create.return_value = mock_test
        mock_store.return_value = "test_123"
        
        test_data = {
            "test_name": "model_v2_test",
            "model_name": "test_model",
            "variant_a_model_id": "model_v1",
            "variant_b_model_id": "model_v2",
            "variant_a_percentage": 50.0,
            "variant_b_percentage": 50.0
        }
        
        response = client.post("/api/v1/monitoring/ab-tests", json=test_data)
        
        assert response.status_code == 201
        result = response.json()
        assert result["test_name"] == "model_v2_test"
        mock_create.assert_called_once()
        mock_store.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.start_ab_test', new_callable=AsyncMock)
    def test_start_ab_test(self, mock_start, client):
        """Test starting an A/B test"""
        mock_start.return_value = True
        
        response = client.post("/api/v1/monitoring/ab-tests/test_123/start")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_start.assert_called_once_with(test_id="test_123")
    
    @patch('app.services.monitoring_service.monitoring_service.stop_ab_test', new_callable=AsyncMock)
    def test_stop_ab_test(self, mock_stop, client):
        """Test stopping an A/B test"""
        mock_stop.return_value = True
        
        response = client.post("/api/v1/monitoring/ab-tests/test_123/stop")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_stop.assert_called_once_with(test_id="test_123")
    
    @patch('app.services.monitoring_service.monitoring_service.calculate_ab_test_metrics', new_callable=AsyncMock)
    def test_get_ab_test_metrics(self, mock_metrics, client):
        """Test getting A/B test metrics"""
        from app.schemas.monitoring import ABTestMetrics
        
        mock_result = ABTestMetrics(
            test_id="test_123",
            variant="variant_a",
            model_id="model_v1",
            time_window_start=datetime.utcnow() - timedelta(days=7),
            time_window_end=datetime.utcnow(),
            total_requests=1000,
            successful_requests=950,
            accuracy=0.90
        )
        mock_metrics.return_value = mock_result
        
        response = client.get("/api/v1/monitoring/ab-tests/test_123/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert result["test_id"] == "test_123"
        mock_metrics.assert_called_once()




class TestABTestingAdvanced:
    """Test advanced A/B testing endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.assign_variant', new_callable=AsyncMock)
    def test_assign_ab_test_variant(self, mock_assign, client):
        """Test assigning A/B test variant"""
        mock_assign.return_value = "variant_a"
        
        response = client.post(
            "/api/v1/monitoring/ab-tests/test_123/assign",
            params={"user_id": "user_123"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["variant_id"] == "variant_a"
        mock_assign.assert_called_once()




