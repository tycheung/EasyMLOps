"""
Comprehensive tests for monitoring routes
Tests all monitoring REST API endpoints including health checks, metrics, logs, and alerts
"""

import pytest
import json
from datetime import datetime, timedelta
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
    MetricType
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
        "max_latency_ms": 120.5,
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


class TestAlerts:
    """Test alerting endpoints"""
    
    def test_get_alerts_success(self, client, sample_alerts_list):
        """Test successful retrieval of alerts"""
        response = client.get(f"{settings.API_V1_PREFIX}/monitoring/alerts")
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2
        alert_ids_from_response = sorted([item['id'] for item in result])
        assert alert_ids_from_response == sorted(["alert_1", "alert_2"])
    
    def test_get_alerts_with_filters(self, client, sample_alerts_list):
        """Test filtering alerts"""
        response_critical = client.get(f"{settings.API_V1_PREFIX}/monitoring/alerts?severity=critical")
        assert response_critical.status_code == 200
        result_critical = response_critical.json()
        assert len(result_critical) == 1
        assert result_critical[0]["id"] == "alert_2"
        assert result_critical[0]["severity"] == str(AlertSeverity.CRITICAL.value)

        response_component = client.get(f"{settings.API_V1_PREFIX}/monitoring/alerts?component={SystemComponent.API_SERVER.value}")
        assert response_component.status_code == 200
        result_component = response_component.json()
        assert len(result_component) == 1
        assert result_component[0]["id"] == "alert_1"
        assert result_component[0]["component"] == str(SystemComponent.API_SERVER.value)

        response_all = client.get(f"{settings.API_V1_PREFIX}/monitoring/alerts?active_only=false")
        assert response_all.status_code == 200
        result_all = response_all.json()
        assert len(result_all) == 2
        alert_ids_from_all = sorted([item['id'] for item in result_all])
        assert alert_ids_from_all == sorted(["alert_1", "alert_2"])

        response_limit = client.get(f"{settings.API_V1_PREFIX}/monitoring/alerts?limit=1")
        assert response_limit.status_code == 200
        result_limit = response_limit.json()
        assert len(result_limit) == 1


class TestMonitoringErrorHandling:
    """Test error handling in monitoring routes"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health_status', new_callable=AsyncMock)
    def test_monitoring_service_unavailable(self, mock_health, client):
        """Test handling when monitoring service is unavailable"""
        mock_health.side_effect = Exception("Monitoring service unavailable")
        
        response = client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 500
        result = response.json()
        assert "error" in result
        assert result["error"]["message"] == "Monitoring service unavailable"
    
    def test_invalid_time_range(self, client):
        """Test handling invalid time range parameters"""
        response = client.get(
            "/api/v1/monitoring/models/test_model/performance",
            params={
                "start_time": "invalid-date",
                "end_time": "2024-01-01T01:00:00Z"
            }
        )
        
        assert response.status_code == 422


# The TestMonitoringIntegration class and its methods are being removed 
# because they test non-existent/removed POST routes like /logs and /alerts/resolve.
# The GET /alerts test is covered in TestAlerts.
# The GET /models/{model_id}/performance is covered in TestModelPerformance.

# Remove the entire TestMonitoringIntegration class:
# class TestMonitoringIntegration:
#     """Test integration of monitoring components"""
#     ... 