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



# Tests for Performance monitoring routes

class TestModelPerformance:
    """Test model performance monitoring endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics', new_callable=AsyncMock)
    def test_get_model_performance_success(self, mock_metrics, client, sample_performance_metrics):
        """Test successful model performance metrics retrieval"""
        mock_metrics.return_value = sample_performance_metrics
        
        response = client.get(
            "/api/v1/monitoring/models/test_model_123/performance",
            params={
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T01:00:00Z"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == "test_model_123"
        assert result["total_requests"] == 1500
        assert result["success_rate"] == 99.0
        assert result["avg_latency_ms"] == 45.2
        assert result["avg_latency_ms"] > 0
        mock_metrics.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics', new_callable=AsyncMock)
    def test_get_model_performance_with_deployment_filter(self, mock_metrics, client, sample_performance_metrics):
        """Test model performance metrics with deployment filter"""
        mock_metrics.return_value = sample_performance_metrics
        
        response = client.get(
            "/api/v1/monitoring/models/test_model_123/performance",
            params={
                "deployment_id": "deploy_456",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T01:00:00Z"
            }
        )
        
        assert response.status_code == 200
        mock_metrics.assert_called_once_with(
            model_id="test_model_123",
            start_time=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
            end_time=datetime.fromisoformat("2024-01-01T01:00:00+00:00"),
            deployment_id="deploy_456"
        )
        call_args = mock_metrics.call_args[1]
        assert call_args["deployment_id"] == "deploy_456"
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics', new_callable=AsyncMock)
    def test_get_model_performance_default_time_window(self, mock_metrics, client, sample_performance_metrics):
        """Test model performance metrics with default time window"""
        mock_metrics.return_value = sample_performance_metrics
        
        response = client.get("/api/v1/monitoring/models/test_model_123/performance")
        
        assert response.status_code == 200
        # Should use default 1-hour window
        mock_metrics.assert_called_once()
        call_args = mock_metrics.call_args[1]
        assert "start_time" in call_args
        assert "end_time" in call_args
        # End time should be close to now, start time should be ~1 hour earlier
        time_diff = call_args["end_time"] - call_args["start_time"]
        assert abs(time_diff.total_seconds() - 3600) < 60  # Within 1 minute of 1 hour
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics', new_callable=AsyncMock)
    def test_get_model_performance_not_found(self, mock_metrics, client):
        """Test model performance metrics for non-existent model"""
        mock_metrics.side_effect = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
        
        response = client.get("/api/v1/monitoring/models/nonexistent/performance")
        
        assert response.status_code == 404
        result = response.json()
        assert "error" in result
        assert result["error"]["message"] == "Not found"
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics', new_callable=AsyncMock)
    def test_get_model_performance_service_error(self, mock_metrics, client):
        """Test model performance endpoint when service fails"""
        mock_metrics.side_effect = Exception("Database connection failed")
        
        response = client.get("/api/v1/monitoring/models/test_model_123/performance")
        
        assert response.status_code == 500




class TestPerformanceMonitoring:
    """Test performance monitoring endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_prediction_logs', new_callable=AsyncMock)
    def test_get_prediction_logs(self, mock_get, client):
        """Test getting prediction logs"""
        mock_logs = [
            {
                "id": "log_1",
                "model_id": "test_model",
                "timestamp": datetime.utcnow().isoformat(),
                "latency_ms": 45.2,
                "success": True
            }
        ]
        mock_get.return_value = mock_logs
        
        response = client.get("/api/v1/monitoring/models/test_model/predictions/logs")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["id"] == "log_1"
        # Check that it was called with the path parameter model_id
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs.get("model_id") == "test_model" or call_args.args[0] == "test_model"
    
    @patch('app.services.monitoring_service.monitoring_service.get_aggregated_metrics', new_callable=AsyncMock)
    def test_get_aggregated_metrics(self, mock_get, client):
        """Test getting aggregated metrics"""
        mock_metrics = {
            "total_requests": 1000,
            "avg_latency": 45.2,
            "success_rate": 0.99
        }
        mock_get.return_value = mock_metrics
        
        response = client.get(
            "/api/v1/monitoring/models/test_model/metrics/aggregated",
            params={"time_range": "24h"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["total_requests"] == 1000
        mock_get.assert_called_once_with(model_id="test_model", time_range="24h")
    
    @patch('app.services.monitoring_service.monitoring_service.get_deployment_summary', new_callable=AsyncMock)
    def test_get_deployment_summary(self, mock_get, client):
        """Test getting deployment summary"""
        mock_summary = {
            "deployment_id": "deploy_123",
            "total_requests": 5000,
            "avg_latency": 50.0
        }
        mock_get.return_value = mock_summary
        
        response = client.get("/api/v1/monitoring/deployments/deploy_123/summary")
        
        assert response.status_code == 200
        result = response.json()
        assert result["deployment_id"] == "deploy_123"
        mock_get.assert_called_once_with(deployment_id="deploy_123")
    
    @patch('app.services.monitoring_service.monitoring_service.calculate_confidence_metrics', new_callable=AsyncMock)
    def test_get_confidence_metrics(self, mock_calculate, client):
        """Test calculating confidence metrics"""
        from app.schemas.monitoring import ModelConfidenceMetrics
        
        mock_metrics = ModelConfidenceMetrics(
            model_id="test_model",
            time_window_start=datetime.utcnow() - timedelta(days=7),
            time_window_end=datetime.utcnow(),
            avg_confidence=0.85,
            low_confidence_rate=0.05,
            total_samples=1000,
            samples_with_confidence=950
        )
        mock_calculate.return_value = mock_metrics
        
        response = client.get(
            "/api/v1/monitoring/models/test_model/confidence",
            params={
                "start_time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["avg_confidence"] == 0.85
        mock_calculate.assert_called_once()




