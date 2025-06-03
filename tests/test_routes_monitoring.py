"""
Comprehensive tests for monitoring routes
Tests all monitoring REST API endpoints including health checks, metrics, logs, and alerts
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.monitoring import ModelPerformanceMetrics, PredictionLog, SystemHealthStatus


@pytest.fixture
def monitoring_client():
    """Create test client for monitoring routes"""
    return TestClient(app)


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
    """Sample system health metrics"""
    return {
        "timestamp": datetime.utcnow(),
        "cpu_usage": 45.2,
        "memory_usage": 68.5,
        "disk_usage": 78.3,
        "active_deployments": 5,
        "pending_requests": 12,
        "error_rate": 1.2,
        "uptime_seconds": 86400,
        "status": "healthy"
    }


class TestSystemHealth:
    """Test system health monitoring endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health')
    def test_get_system_health_success(self, mock_health, monitoring_client, sample_system_health):
        """Test successful system health retrieval"""
        mock_health.return_value = sample_system_health
        
        response = monitoring_client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert result["cpu_usage"] == 45.2
        assert result["memory_usage"] == 68.5
        assert result["active_deployments"] == 5
        mock_health.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health')
    def test_get_system_health_unhealthy(self, mock_health, monitoring_client):
        """Test system health when system is unhealthy"""
        unhealthy_metrics = {
            "timestamp": datetime.utcnow(),
            "cpu_usage": 95.5,
            "memory_usage": 88.9,
            "disk_usage": 92.1,
            "active_deployments": 5,
            "pending_requests": 250,
            "error_rate": 15.5,
            "uptime_seconds": 86400,
            "status": "unhealthy"
        }
        mock_health.return_value = unhealthy_metrics
        
        response = monitoring_client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "unhealthy"
        assert result["cpu_usage"] > 90
        assert result["error_rate"] > 10
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health')
    def test_get_system_health_service_error(self, mock_health, monitoring_client):
        """Test system health endpoint when service fails"""
        mock_health.side_effect = Exception("Health check failed")
        
        response = monitoring_client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 500


class TestModelPerformance:
    """Test model performance monitoring endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics')
    def test_get_model_performance_success(self, mock_metrics, monitoring_client, sample_performance_metrics):
        """Test successful model performance metrics retrieval"""
        mock_metrics.return_value = sample_performance_metrics
        
        response = monitoring_client.get(
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
        mock_metrics.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics')
    def test_get_model_performance_with_deployment_filter(self, mock_metrics, monitoring_client, sample_performance_metrics):
        """Test model performance metrics with deployment filter"""
        mock_metrics.return_value = sample_performance_metrics
        
        response = monitoring_client.get(
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
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics')
    def test_get_model_performance_default_time_window(self, mock_metrics, monitoring_client, sample_performance_metrics):
        """Test model performance metrics with default time window"""
        mock_metrics.return_value = sample_performance_metrics
        
        response = monitoring_client.get("/api/v1/monitoring/models/test_model_123/performance")
        
        assert response.status_code == 200
        # Should use default 1-hour window
        mock_metrics.assert_called_once()
        call_args = mock_metrics.call_args[1]
        assert "start_time" in call_args
        assert "end_time" in call_args
        # End time should be close to now, start time should be ~1 hour earlier
        time_diff = call_args["end_time"] - call_args["start_time"]
        assert abs(time_diff.total_seconds() - 3600) < 60  # Within 1 minute of 1 hour
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics')
    def test_get_model_performance_not_found(self, mock_metrics, monitoring_client):
        """Test model performance metrics for non-existent model"""
        mock_metrics.return_value = None
        
        response = monitoring_client.get("/api/v1/monitoring/models/nonexistent/performance")
        
        assert response.status_code == 404
    
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics')
    def test_get_model_performance_service_error(self, mock_metrics, monitoring_client):
        """Test model performance endpoint when service fails"""
        mock_metrics.side_effect = Exception("Database connection failed")
        
        response = monitoring_client.get("/api/v1/monitoring/models/test_model_123/performance")
        
        assert response.status_code == 500


class TestPredictionLogs:
    """Test prediction logging endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_prediction_logs')
    def test_get_prediction_logs_success(self, mock_logs, monitoring_client):
        """Test successful prediction logs retrieval"""
        sample_logs = [
            {
                "id": "log_123",
                "model_id": "test_model_123",
                "deployment_id": "deploy_456", 
                "input_data": {"feature1": 0.5, "feature2": "test"},
                "output_data": {"prediction": 0.75, "confidence": 0.85},
                "latency_ms": 45.2,
                "timestamp": datetime.utcnow(),
                "success": True,
                "api_endpoint": "/predict/deploy_456"
            },
            {
                "id": "log_124",
                "model_id": "test_model_123", 
                "deployment_id": "deploy_456",
                "input_data": {"feature1": 0.8, "feature2": "test2"},
                "output_data": {"prediction": 0.65, "confidence": 0.92},
                "latency_ms": 52.1,
                "timestamp": datetime.utcnow() - timedelta(minutes=5),
                "success": True,
                "api_endpoint": "/predict/deploy_456"
            }
        ]
        mock_logs.return_value = sample_logs
        
        response = monitoring_client.get(
            "/api/v1/monitoring/models/test_model_123/logs",
            params={"limit": 100, "offset": 0}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2
        assert result[0]["id"] == "log_123"
        assert result[0]["model_id"] == "test_model_123"
        assert result[0]["latency_ms"] == 45.2
        mock_logs.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_prediction_logs')
    def test_get_prediction_logs_with_filters(self, mock_logs, monitoring_client):
        """Test prediction logs with time range and deployment filters"""
        mock_logs.return_value = []
        
        response = monitoring_client.get(
            "/api/v1/monitoring/models/test_model_123/logs",
            params={
                "deployment_id": "deploy_456",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T01:00:00Z",
                "success_only": True,
                "limit": 50,
                "offset": 10
            }
        )
        
        assert response.status_code == 200
        mock_logs.assert_called_once()
        call_args = mock_logs.call_args[1]
        assert call_args["deployment_id"] == "deploy_456"
        assert call_args["success_only"] is True
        assert call_args["limit"] == 50
        assert call_args["offset"] == 10
    
    @patch('app.services.monitoring_service.monitoring_service.log_prediction')
    def test_log_prediction_success(self, mock_log, monitoring_client):
        """Test successful prediction logging"""
        mock_log.return_value = True
        
        log_data = {
            "model_id": "test_model_123",
            "deployment_id": "deploy_456",
            "input_data": {"feature1": 0.5, "feature2": "test"},
            "output_data": {"prediction": 0.75, "confidence": 0.85},
            "latency_ms": 45.2,
            "api_endpoint": "/predict/deploy_456",
            "success": True
        }
        
        response = monitoring_client.post("/api/v1/monitoring/logs", json=log_data)
        
        assert response.status_code == 201
        result = response.json()
        assert "message" in result
        assert "log_id" in result
        mock_log.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.log_prediction')
    def test_log_prediction_failure(self, mock_log, monitoring_client):
        """Test prediction logging failure"""
        mock_log.side_effect = Exception("Database error")
        
        log_data = {
            "model_id": "test_model_123",
            "input_data": {"feature1": 0.5},
            "output_data": {"error": "Model failed"},
            "latency_ms": 100.0,
            "api_endpoint": "/predict/deploy_456",
            "success": False,
            "error_message": "Model failed to process input"
        }
        
        response = monitoring_client.post("/api/v1/monitoring/logs", json=log_data)
        
        assert response.status_code == 500


class TestAlerts:
    """Test alerting endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_alerts')
    def test_get_alerts_success(self, mock_alerts, monitoring_client):
        """Test successful alerts retrieval"""
        sample_alerts = [
            {
                "id": "alert_123",
                "alert_type": "high_error_rate",
                "severity": "warning",
                "message": "Model error rate above threshold",
                "model_id": "test_model_123",
                "deployment_id": "deploy_456",
                "threshold_value": 5.0,
                "actual_value": 7.5,
                "created_at": datetime.utcnow(),
                "resolved": False
            },
            {
                "id": "alert_124", 
                "alert_type": "high_latency",
                "severity": "critical",
                "message": "Model latency exceeds threshold",
                "model_id": "test_model_123",
                "deployment_id": "deploy_456",
                "threshold_value": 100.0,
                "actual_value": 150.2,
                "created_at": datetime.utcnow() - timedelta(minutes=10),
                "resolved": False
            }
        ]
        mock_alerts.return_value = sample_alerts
        
        response = monitoring_client.get("/api/v1/monitoring/alerts")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2
        assert result[0]["alert_type"] == "high_error_rate"
        assert result[0]["severity"] == "warning"
        assert result[1]["severity"] == "critical"
        mock_alerts.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_alerts')
    def test_get_alerts_with_filters(self, mock_alerts, monitoring_client):
        """Test alerts retrieval with filters"""
        mock_alerts.return_value = []
        
        response = monitoring_client.get(
            "/api/v1/monitoring/alerts",
            params={
                "model_id": "test_model_123",
                "severity": "critical",
                "resolved": False,
                "limit": 20
            }
        )
        
        assert response.status_code == 200
        mock_alerts.assert_called_once()
        call_args = mock_alerts.call_args[1]
        assert call_args["model_id"] == "test_model_123"
        assert call_args["severity"] == "critical"
        assert call_args["resolved"] is False
    
    @patch('app.services.monitoring_service.monitoring_service.create_alert')
    def test_create_alert_success(self, mock_create, monitoring_client):
        """Test successful alert creation"""
        mock_alert = {
            "id": "alert_123",
            "alert_type": "high_error_rate",
            "severity": "warning",
            "message": "Model error rate above threshold",
            "model_id": "test_model_123"
        }
        mock_create.return_value = mock_alert
        
        alert_data = {
            "alert_type": "high_error_rate",
            "severity": "warning",
            "message": "Model error rate above threshold",
            "model_id": "test_model_123",
            "deployment_id": "deploy_456",
            "threshold_value": 5.0,
            "actual_value": 7.5
        }
        
        response = monitoring_client.post("/api/v1/monitoring/alerts", json=alert_data)
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "alert_123"
        assert result["alert_type"] == "high_error_rate"
        mock_create.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.resolve_alert')
    def test_resolve_alert_success(self, mock_resolve, monitoring_client):
        """Test successful alert resolution"""
        mock_resolve.return_value = True
        
        response = monitoring_client.post("/api/v1/monitoring/alerts/alert_123/resolve")
        
        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        assert "resolved" in result["message"]
        mock_resolve.assert_called_once_with("alert_123")
    
    @patch('app.services.monitoring_service.monitoring_service.resolve_alert')
    def test_resolve_alert_not_found(self, mock_resolve, monitoring_client):
        """Test resolving non-existent alert"""
        mock_resolve.return_value = False
        
        response = monitoring_client.post("/api/v1/monitoring/alerts/nonexistent/resolve")
        
        assert response.status_code == 404


class TestMetricsAggregation:
    """Test metrics aggregation endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_aggregated_metrics')
    def test_get_aggregated_metrics_success(self, mock_metrics, monitoring_client):
        """Test successful aggregated metrics retrieval"""
        aggregated_data = {
            "time_window": "1h",
            "total_models": 10,
            "total_deployments": 15,
            "total_requests": 50000,
            "avg_latency_ms": 45.5,
            "overall_success_rate": 98.5,
            "overall_error_rate": 1.5,
            "top_performing_models": [
                {"model_id": "model_1", "success_rate": 99.8},
                {"model_id": "model_2", "success_rate": 99.5}
            ],
            "alerts_summary": {
                "critical": 1,
                "warning": 3,
                "info": 0
            }
        }
        mock_metrics.return_value = aggregated_data
        
        response = monitoring_client.get(
            "/api/v1/monitoring/metrics/aggregated",
            params={"time_window": "1h"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["total_models"] == 10
        assert result["overall_success_rate"] == 98.5
        assert len(result["top_performing_models"]) == 2
        assert result["alerts_summary"]["critical"] == 1
        mock_metrics.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_deployment_summary')
    def test_get_deployment_summary(self, mock_summary, monitoring_client):
        """Test deployment summary endpoint"""
        summary_data = {
            "total_deployments": 15,
            "active_deployments": 12,
            "inactive_deployments": 3,
            "deployments_by_status": {
                "running": 10,
                "pending": 2,
                "stopped": 2,
                "failed": 1
            },
            "deployments_by_framework": {
                "sklearn": 8,
                "tensorflow": 4,
                "pytorch": 3
            },
            "total_requests_last_hour": 15000,
            "avg_response_time_ms": 45.2
        }
        mock_summary.return_value = summary_data
        
        response = monitoring_client.get("/api/v1/monitoring/deployments/summary")
        
        assert response.status_code == 200
        result = response.json()
        assert result["total_deployments"] == 15
        assert result["active_deployments"] == 12
        assert result["deployments_by_status"]["running"] == 10
        assert result["deployments_by_framework"]["sklearn"] == 8


class TestMonitoringErrorHandling:
    """Test error handling in monitoring routes"""
    
    @patch('app.services.monitoring_service.monitoring_service.get_system_health')
    def test_monitoring_service_unavailable(self, mock_health, monitoring_client):
        """Test handling when monitoring service is unavailable"""
        mock_health.side_effect = Exception("Monitoring service unavailable")
        
        response = monitoring_client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 500
        result = response.json()
        assert "detail" in result
    
    @patch('app.database.get_session')
    def test_database_connection_error(self, mock_get_session, monitoring_client):
        """Test handling database connection errors"""
        mock_get_session.side_effect = Exception("Database connection failed")
        
        response = monitoring_client.get("/api/v1/monitoring/models/test_model/logs")
        
        assert response.status_code == 500
    
    def test_invalid_time_range(self, monitoring_client):
        """Test handling invalid time range parameters"""
        response = monitoring_client.get(
            "/api/v1/monitoring/models/test_model/performance",
            params={
                "start_time": "invalid-date",
                "end_time": "2024-01-01T01:00:00Z"
            }
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, monitoring_client):
        """Test handling missing required fields in request bodies"""
        incomplete_log_data = {
            "model_id": "test_model_123",
            # Missing required fields like input_data, output_data, etc.
        }
        
        response = monitoring_client.post("/api/v1/monitoring/logs", json=incomplete_log_data)
        
        assert response.status_code == 422


class TestMonitoringIntegration:
    """Integration tests for monitoring workflows"""
    
    @patch('app.services.monitoring_service.monitoring_service.log_prediction')
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics')
    @patch('app.services.monitoring_service.monitoring_service.create_alert')
    def test_complete_monitoring_workflow(self, mock_alert, mock_metrics, mock_log, 
                                        monitoring_client, sample_performance_metrics):
        """Test complete monitoring workflow: log -> metrics -> alert"""
        # Mock successful prediction logging
        mock_log.return_value = True
        
        # Mock performance metrics that trigger an alert
        high_error_metrics = sample_performance_metrics.copy()
        high_error_metrics["error_rate"] = 15.0  # High error rate
        mock_metrics.return_value = high_error_metrics
        
        # Mock alert creation
        mock_alert.return_value = {
            "id": "alert_123",
            "alert_type": "high_error_rate",
            "severity": "critical"
        }
        
        # Log prediction
        log_data = {
            "model_id": "test_model_123",
            "input_data": {"feature1": 0.5},
            "output_data": {"error": "Model failed"},
            "latency_ms": 100.0,
            "success": False,
            "api_endpoint": "/predict/deploy_456"
        }
        log_response = monitoring_client.post("/api/v1/monitoring/logs", json=log_data)
        assert log_response.status_code == 201
        
        # Get performance metrics
        metrics_response = monitoring_client.get("/api/v1/monitoring/models/test_model_123/performance")
        assert metrics_response.status_code == 200
        metrics_result = metrics_response.json()
        assert metrics_result["error_rate"] == 15.0
        
        # Create alert based on high error rate
        alert_data = {
            "alert_type": "high_error_rate",
            "severity": "critical",
            "message": "Model error rate exceeds threshold",
            "model_id": "test_model_123",
            "threshold_value": 5.0,
            "actual_value": 15.0
        }
        alert_response = monitoring_client.post("/api/v1/monitoring/alerts", json=alert_data)
        assert alert_response.status_code == 201
        
        # Verify all services were called
        mock_log.assert_called_once()
        mock_metrics.assert_called_once()
        mock_alert.assert_called_once() 