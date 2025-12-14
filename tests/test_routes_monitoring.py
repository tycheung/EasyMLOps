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


# ============================================================================
# NEW ENDPOINT TESTS - Added for comprehensive API coverage
# ============================================================================

class TestDriftDetection:
    """Test drift detection endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.detect_feature_drift', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_drift_detection', new_callable=AsyncMock)
    def test_detect_feature_drift(self, mock_store, mock_detect, client):
        """Test feature drift detection endpoint"""
        from app.schemas.monitoring import ModelDriftDetection, DriftType, DriftSeverity
        
        mock_result = ModelDriftDetection(
            model_id="test_model",
            drift_type=DriftType.FEATURE,
            detection_method="ks_test",
            baseline_window_start=datetime.utcnow() - timedelta(days=7),
            baseline_window_end=datetime.utcnow() - timedelta(days=1),
            current_window_start=datetime.utcnow() - timedelta(days=1),
            current_window_end=datetime.utcnow(),
            drift_detected=True,
            drift_score=0.75,
            drift_severity=DriftSeverity.HIGH
        )
        mock_detect.return_value = mock_result
        mock_store.return_value = "drift_id_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/drift/feature",
            params={
                "baseline_window_start": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "baseline_window_end": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "current_window_start": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "current_window_end": datetime.utcnow().isoformat(),
                "drift_threshold": 0.2
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["drift_detected"] is True
        assert result["drift_severity"] == DriftSeverity.HIGH.value
        mock_detect.assert_called_once()
        mock_store.assert_called_once()


class TestAlertManagement:
    """Test alert management endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_alert', new_callable=AsyncMock)
    def test_create_alert(self, mock_create, client):
        """Test creating an alert"""
        mock_create.return_value = "alert_123"
        
        response = client.post(
            "/api/v1/monitoring/alerts",
            params={
                "severity": "critical",
                "component": "api_server",
                "title": "High CPU Usage",
                "description": "CPU usage exceeded 90%",
                "metric_value": 95.5,
                "threshold_value": 90.0
            }
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "alert_123"
        mock_create.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.resolve_alert', new_callable=AsyncMock)
    def test_resolve_alert(self, mock_resolve, client):
        """Test resolving an alert"""
        mock_resolve.return_value = True
        
        response = client.post("/api/v1/monitoring/alerts/alert_123/resolve")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_resolve.assert_called_once_with(alert_id="alert_123")
    
    @patch('app.services.monitoring_service.monitoring_service.acknowledge_alert', new_callable=AsyncMock)
    def test_acknowledge_alert(self, mock_ack, client):
        """Test acknowledging an alert"""
        mock_ack.return_value = True
        
        response = client.post(
            "/api/v1/monitoring/alerts/alert_123/acknowledge",
            params={"acknowledged_by": "user@example.com"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_ack.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.check_and_create_alerts', new_callable=AsyncMock)
    def test_check_and_create_alerts(self, mock_check, client):
        """Test checking and creating alerts"""
        from app.schemas.monitoring import Alert, AlertSeverity
        
        mock_alerts = [
            Alert(
                id="alert_1",
                severity=AlertSeverity.WARNING,
                component="api_server",
                title="Test Alert",
                description="Test description"
            )
        ]
        mock_check.return_value = mock_alerts
        
        response = client.post("/api/v1/monitoring/alerts/check")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["id"] == "alert_1"


class TestAlertRules:
    """Test alert rules endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_alert_rule', new_callable=AsyncMock)
    def test_create_alert_rule(self, mock_create, client):
        """Test creating an alert rule"""
        mock_create.return_value = "rule_123"
        
        rule_data = {
            "rule_name": "High Latency Alert",
            "description": "Alert when latency exceeds threshold",
            "metric_name": "avg_latency_ms",
            "condition": "gt",
            "threshold_value": 1000.0,
            "severity": "warning",
            "component": "api_server"
        }
        
        response = client.post(
            "/api/v1/monitoring/alert-rules",
            json=rule_data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "rule_123"
        mock_create.assert_called_once()


class TestPerformanceDegradation:
    """Test performance degradation endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.log_prediction_with_ground_truth', new_callable=AsyncMock)
    def test_log_prediction_with_ground_truth(self, mock_log, client):
        """Test logging prediction with ground truth"""
        mock_log.return_value = "log_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/degradation/log",
            params={
                "prediction": 0.85,
                "ground_truth": 0.90
            }
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "log_123"
        mock_log.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.detect_performance_degradation', new_callable=AsyncMock)
    def test_detect_performance_degradation(self, mock_detect, client):
        """Test detecting performance degradation"""
        from app.schemas.monitoring import ModelPerformanceHistory
        
        mock_result = ModelPerformanceHistory(
            model_id="test_model",
            time_window_start=datetime.utcnow() - timedelta(days=7),
            time_window_end=datetime.utcnow(),
            model_type="classification",
            accuracy=0.75,
            baseline_accuracy=0.85,
            performance_degraded=True,
            total_samples=1000,
            samples_with_ground_truth=1000
        )
        mock_detect.return_value = mock_result
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/degradation/detect",
            params={
                "start_time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "degradation_threshold": 0.1
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["performance_degraded"] is True
        mock_detect.assert_called_once()


class TestModelBaseline:
    """Test model baseline endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_model_baseline', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_model_baseline', new_callable=AsyncMock)
    def test_create_model_baseline(self, mock_store, mock_create, client):
        """Test creating a model baseline"""
        from app.schemas.monitoring import ModelBaseline
        
        mock_baseline = ModelBaseline(
            model_id="test_model",
            model_name="test_model",
            model_version="1.0.0",
            baseline_type="performance",
            baseline_sample_count=1000,
            baseline_time_window_start=datetime.utcnow() - timedelta(days=30),
            baseline_time_window_end=datetime.utcnow() - timedelta(days=1),
            baseline_accuracy=0.90
        )
        mock_create.return_value = mock_baseline
        mock_store.return_value = "baseline_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/baseline",
            params={
                "baseline_name": "v1_baseline",
                "start_time": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "end_time": (datetime.utcnow() - timedelta(days=1)).isoformat()
            }
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["model_name"] == "test_model"
        mock_create.assert_called_once()
        mock_store.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.get_active_baseline', new_callable=AsyncMock)
    def test_get_active_baseline(self, mock_get, client):
        """Test getting active baseline"""
        from app.schemas.monitoring import ModelBaseline
        
        mock_baseline = ModelBaseline(
            model_id="test_model",
            model_name="test_model",
            model_version="1.0.0",
            baseline_type="performance",
            baseline_sample_count=1000,
            baseline_time_window_start=datetime.utcnow() - timedelta(days=30),
            baseline_time_window_end=datetime.utcnow() - timedelta(days=1),
            baseline_accuracy=0.90
        )
        mock_get.return_value = mock_baseline
        
        response = client.get("/api/v1/monitoring/models/test_model/baseline")
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_name"] == "test_model"
        mock_get.assert_called_once()


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


class TestExplainability:
    """Test explainability endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.generate_shap_explanation', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_explanation', new_callable=AsyncMock)
    def test_generate_shap_explanation(self, mock_store, mock_generate, client):
        """Test generating SHAP explanation"""
        from app.schemas.monitoring import ModelExplanation, ExplanationType
        
        mock_explanation = ModelExplanation(
            model_id="test_model",
            explanation_type=ExplanationType.SHAP,
            input_data={"feature1": 0.5, "feature2": 0.3},
            feature_importance={"feature1": 0.7, "feature2": 0.3}
        )
        mock_generate.return_value = mock_explanation
        mock_store.return_value = "explanation_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/explain/shap",
            json={"feature1": 0.5, "feature2": 0.3}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["explanation_type"] == ExplanationType.SHAP.value
        mock_generate.assert_called_once()
        mock_store.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.generate_lime_explanation', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_explanation', new_callable=AsyncMock)
    def test_generate_lime_explanation(self, mock_store, mock_generate, client):
        """Test generating LIME explanation"""
        from app.schemas.monitoring import ModelExplanation, ExplanationType
        
        mock_explanation = ModelExplanation(
            model_id="test_model",
            explanation_type=ExplanationType.LIME,
            input_data={"feature1": 0.5, "feature2": 0.3},
            feature_importance={"feature1": 0.7, "feature2": 0.3}
        )
        mock_generate.return_value = mock_explanation
        mock_store.return_value = "explanation_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/explain/lime",
            json={"feature1": 0.5, "feature2": 0.3}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["explanation_type"] == ExplanationType.LIME.value
        mock_generate.assert_called_once()
        mock_store.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.calculate_global_feature_importance', new_callable=AsyncMock)
    def test_get_global_feature_importance(self, mock_importance, client):
        """Test getting global feature importance"""
        mock_result = {
            "feature1": 0.7,
            "feature2": 0.3
        }
        mock_importance.return_value = mock_result
        
        response = client.get("/api/v1/monitoring/models/test_model/explain/importance")
        
        assert response.status_code == 200
        result = response.json()
        assert result["feature1"] == 0.7
        mock_importance.assert_called_once()


class TestDataQuality:
    """Test data quality endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.detect_outliers', new_callable=AsyncMock)
    def test_detect_outliers(self, mock_detect, client):
        """Test detecting outliers"""
        from app.schemas.monitoring import OutlierDetectionMethod, OutlierType
        
        mock_result = {
            "model_id": "test_model",
            "detection_method": OutlierDetectionMethod.ISOLATION_FOREST,
            "outlier_type": OutlierType.INPUT,
            "is_outlier": True,
            "outlier_score": 0.95
        }
        mock_detect.return_value = mock_result
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/data-quality/outliers",
            json=[{"feature1": 0.5}, {"feature1": 10.0}],
            params={"method": "isolation_forest"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["is_outlier"] is True
        mock_detect.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.calculate_data_quality_metrics', new_callable=AsyncMock)
    def test_calculate_data_quality_metrics(self, mock_metrics, client):
        """Test calculating data quality metrics"""
        mock_result = {
            "model_id": "test_model",
            "time_window_start": datetime.utcnow() - timedelta(days=1),
            "time_window_end": datetime.utcnow(),
            "total_samples": 100,
            "valid_samples": 100,
            "completeness_score": 1.0
        }
        mock_metrics.return_value = mock_result
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/data-quality/metrics",
            json=[{"feature1": 0.5, "feature2": 0.3}]
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["completeness_score"] == 1.0
        mock_metrics.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.detect_anomaly', new_callable=AsyncMock)
    def test_detect_anomaly(self, mock_detect, client):
        """Test detecting anomaly"""
        from app.schemas.monitoring import AnomalyType
        
        mock_result = {
            "model_id": "test_model",
            "anomaly_type": AnomalyType.INPUT,
            "detection_method": "statistical",
            "anomaly_score": 0.95,
            "is_anomaly": True
        }
        mock_detect.return_value = mock_result
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/data-quality/anomaly",
            json={"feature1": 10.0, "feature2": 20.0}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["is_anomaly"] is True
        mock_detect.assert_called_once()


class TestFairness:
    """Test bias and fairness endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.calculate_fairness_metrics', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_bias_fairness_metrics', new_callable=AsyncMock)
    def test_calculate_fairness_metrics(self, mock_store, mock_calculate, client):
        """Test calculating fairness metrics"""
        from app.schemas.monitoring import BiasFairnessMetrics
        
        mock_metrics = BiasFairnessMetrics(
            model_id="test_model",
            protected_attribute="gender",
            time_window_start=datetime.utcnow() - timedelta(days=7),
            time_window_end=datetime.utcnow(),
            demographic_parity_score=0.95,
            equalized_odds_score=0.92
        )
        mock_calculate.return_value = mock_metrics
        mock_store.return_value = "metrics_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/fairness/metrics",
            params={
                "protected_attribute": "gender",
                "start_time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["protected_attribute"] == "gender"
        mock_calculate.assert_called_once()
        mock_store.assert_called_once()


class TestModelVersioning:
    """Test model versioning endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.compare_model_versions', new_callable=AsyncMock)
    @patch('app.services.monitoring_service.monitoring_service.store_version_comparison', new_callable=AsyncMock)
    def test_compare_model_versions(self, mock_store, mock_compare, client):
        """Test comparing model versions"""
        from app.schemas.monitoring import ModelVersionComparison
        
        mock_comparison = ModelVersionComparison(
            model_name="test_model",
            baseline_version="v1",
            comparison_version="v2",
            baseline_model_id="model_v1",
            comparison_model_id="model_v2",
            comparison_window_start=datetime.utcnow() - timedelta(days=7),
            comparison_window_end=datetime.utcnow(),
            accuracy_delta=0.05,
            performance_improved=True
        )
        mock_compare.return_value = mock_comparison
        mock_store.return_value = "comparison_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/versions/compare",
            params={
                "version_a_id": "v1",
                "version_b_id": "v2",
                "start_time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["performance_improved"] is True
        mock_compare.assert_called_once()
        mock_store.assert_called_once()


class TestAuditLogs:
    """Test audit log endpoints"""
    
    def test_get_audit_logs(self, client, test_session):
        """Test getting audit logs"""
        from app.models.monitoring import AuditLogDB
        
        # Create test audit log
        audit_log = AuditLogDB(
            id="audit_123",
            resource_type="model",
            resource_id="test_model",
            action="create",
            user_id="user_123",
            timestamp=datetime.utcnow()
        )
        test_session.add(audit_log)
        test_session.commit()
        
        response = client.get("/api/v1/monitoring/audit")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert any(log["id"] == "audit_123" for log in result)
    
    def test_get_audit_logs_with_filters(self, client, test_session):
        """Test getting audit logs with filters"""
        from app.models.monitoring import AuditLogDB
        
        audit_log = AuditLogDB(
            id="audit_456",
            resource_type="model",
            resource_id="test_model",
            action="update",
            user_id="user_123",
            timestamp=datetime.utcnow()
        )
        test_session.add(audit_log)
        test_session.commit()
        
        response = client.get(
            "/api/v1/monitoring/audit",
            params={"entity_type": "model", "action": "update"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert all(log["resource_type"] == "model" for log in result)
        assert all(log["action"] == "update" for log in result)


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


class TestNotifications:
    """Test notification endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.send_alert_notification', new_callable=AsyncMock)
    def test_send_alert_notification(self, mock_send, client):
        """Test sending alert notification"""
        mock_send.return_value = True
        
        response = client.post(
            "/api/v1/monitoring/notifications/send",
            params={"alert_id": "alert_123"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_send.assert_called_once()


class TestAlertManagementAdvanced:
    """Test advanced alert management endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.group_alerts', new_callable=AsyncMock)
    def test_group_alerts(self, mock_group, client):
        """Test grouping alerts"""
        mock_group.return_value = "group_123"
        
        response = client.post(
            "/api/v1/monitoring/alerts/group",
            json={"alert_ids": ["alert_1", "alert_2"], "group_name": "Test Group"}
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "group_123"
        mock_group.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.create_alert_escalation', new_callable=AsyncMock)
    def test_create_alert_escalation(self, mock_create, client):
        """Test creating alert escalation"""
        from app.schemas.monitoring import AlertEscalation, EscalationTriggerCondition
        
        mock_escalation = AlertEscalation(
            escalation_name="High Priority Escalation",
            trigger_condition=EscalationTriggerCondition.SEVERITY,
            trigger_value={"severity": "critical"},
            escalation_actions=["notify", "page"]
        )
        mock_escalation.id = "escalation_123"
        mock_create.return_value = mock_escalation
        
        response = client.post(
            "/api/v1/monitoring/alerts/escalations",
            json={
                "escalation_name": "High Priority Escalation",
                "trigger_condition": "severity",
                "trigger_value": {"severity": "critical"},
                "escalation_actions": ["notify", "page"]
            }
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "escalation_123"
        mock_create.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.check_and_escalate_alerts', new_callable=AsyncMock)
    def test_check_and_escalate_alerts(self, mock_check, client):
        """Test checking and escalating alerts"""
        mock_check.return_value = ["alert_1", "alert_2"]
        
        response = client.post("/api/v1/monitoring/alerts/escalate")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["escalated_alert_ids"]) == 2
        mock_check.assert_called_once()


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


class TestFairnessAdvanced:
    """Test advanced fairness endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.configure_protected_attribute', new_callable=AsyncMock)
    def test_configure_protected_attribute(self, mock_configure, client):
        """Test configuring protected attribute"""
        mock_configure.return_value = "config_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/fairness/attributes",
            params={
                "attribute_name": "gender",
                "attribute_type": "categorical"
            },
            json={"values": ["male", "female", "other"]}
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "config_123"
        mock_configure.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.calculate_demographic_distribution', new_callable=AsyncMock)
    def test_get_demographic_distribution(self, mock_calculate, client):
        """Test calculating demographic distribution"""
        mock_distribution = {
            "male": 0.45,
            "female": 0.50,
            "other": 0.05
        }
        mock_calculate.return_value = mock_distribution
        
        response = client.get(
            "/api/v1/monitoring/models/test_model/fairness/demographics",
            params={
                "protected_attribute": "gender",
                "start_time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["male"] == 0.45
        mock_calculate.assert_called_once()


class TestLifecycleAdvanced:
    """Test advanced lifecycle endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.configure_retraining_trigger', new_callable=AsyncMock)
    def test_configure_retraining_trigger(self, mock_configure, client):
        """Test configuring retraining trigger"""
        mock_configure.return_value = "trigger_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/retraining/triggers",
            params={"trigger_type": "performance_degradation"},
            json={"threshold": 0.1}
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "trigger_123"
        mock_configure.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.generate_model_card', new_callable=AsyncMock)
    def test_generate_model_card(self, mock_generate, client):
        """Test generating model card"""
        from app.schemas.monitoring import ModelCard
        
        mock_card = {
            "model_id": "test_model",
            "model_name": "Test Model",
            "version": "1.0.0",
            "description": "Test model card",
            "card_content": {"sections": []}
        }
        mock_generate.return_value = mock_card
        
        response = client.post("/api/v1/monitoring/models/test_model/card/generate")
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == "test_model"
        mock_generate.assert_called_once_with(model_id="test_model")


class TestGovernanceAdvanced:
    """Test advanced governance endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_data_retention_policy', new_callable=AsyncMock)
    def test_create_data_retention_policy(self, mock_create, client):
        """Test creating data retention policy"""
        mock_create.return_value = "policy_123"
        
        policy_data = {
            "policy_name": "30 Day Retention",
            "policy_description": "Retain data for 30 days",
            "resource_type": "prediction_logs",
            "retention_period_days": 30,
            "action_on_expiry": "delete"
        }
        
        response = client.post(
            "/api/v1/monitoring/governance/retention-policies",
            json=policy_data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "policy_123"
        mock_create.assert_called_once()


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


class TestIntegrationAdvanced:
    """Test advanced integration endpoints"""
    
    @patch('app.services.monitoring_service.monitoring_service.create_sampling_config', new_callable=AsyncMock)
    def test_create_sampling_config(self, mock_create, client):
        """Test creating sampling configuration"""
        mock_create.return_value = "sampling_123"
        
        config_data = {
            "config_name": "10% Random Sampling",
            "resource_type": "prediction_logs",
            "sampling_strategy": "random",
            "sampling_rate": 0.1
        }
        
        response = client.post(
            "/api/v1/monitoring/integrations/sampling",
            json=config_data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "sampling_123"
        mock_create.assert_called_once()
    
    @patch('app.services.monitoring_service.monitoring_service.create_metric_aggregation_config', new_callable=AsyncMock)
    def test_create_metric_aggregation_config(self, mock_create, client):
        """Test creating metric aggregation configuration"""
        mock_create.return_value = "agg_config_123"
        
        config_data = {
            "config_name": "Hourly Aggregation",
            "metric_type": "latency",
            "aggregation_window_seconds": 3600,
            "aggregation_method": "avg"
        }
        
        response = client.post(
            "/api/v1/monitoring/integrations/aggregation",
            json=config_data
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "agg_config_123"
        mock_create.assert_called_once() 