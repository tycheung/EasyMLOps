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



# Tests for Integration monitoring routes

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


class TestIntegrationGET:
    """Test integration GET endpoints"""
    
    def test_list_integrations(self, client, test_session):
        """Test listing all external integrations"""
        from app.models.monitoring import ExternalIntegrationDB
        from datetime import datetime
        
        integration1 = ExternalIntegrationDB(
            id="integration_1",
            integration_type="datadog",
            integration_name="Datadog Integration",
            description="Datadog monitoring integration",
            config={"api_key": "test_key"},
            is_active=True,
            created_at=datetime.utcnow()
        )
        integration2 = ExternalIntegrationDB(
            id="integration_2",
            integration_type="prometheus",
            integration_name="Prometheus Integration",
            description="Prometheus monitoring integration",
            config={"endpoint": "http://prometheus:9090"},
            is_active=False,
            created_at=datetime.utcnow()
        )
        
        test_session.add(integration1)
        test_session.add(integration2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/integrations")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        integration_ids = [i["id"] for i in result]
        assert "integration_1" in integration_ids
        assert "integration_2" in integration_ids
        
        # Test filtering by integration_type
        response = client.get("/api/v1/monitoring/integrations?integration_type=datadog")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(i["integration_type"] == "datadog" for i in result)
        
        # Test filtering by is_active
        response = client.get("/api/v1/monitoring/integrations?is_active=true")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(i["is_active"] is True for i in result)
        
        # Test limit
        response = client.get("/api/v1/monitoring/integrations?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1
    
    def test_list_webhooks(self, client, test_session):
        """Test listing all webhook configurations"""
        from app.models.monitoring import WebhookConfigDB
        from datetime import datetime
        
        webhook1 = WebhookConfigDB(
            id="webhook_1",
            webhook_name="Alert Webhook",
            webhook_url="https://example.com/webhook",
            trigger_events=["alert.created", "alert.resolved"],
            headers={"Authorization": "Bearer token"},
            is_active=True,
            created_at=datetime.utcnow()
        )
        webhook2 = WebhookConfigDB(
            id="webhook_2",
            webhook_name="Deployment Webhook",
            webhook_url="https://example.com/deploy",
            trigger_events=["deployment.started", "deployment.completed"],
            headers={},
            is_active=False,
            created_at=datetime.utcnow()
        )
        
        test_session.add(webhook1)
        test_session.add(webhook2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/integrations/webhooks")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        webhook_ids = [w["id"] for w in result]
        assert "webhook_1" in webhook_ids
        assert "webhook_2" in webhook_ids
        
        # Test filtering by is_active
        response = client.get("/api/v1/monitoring/integrations/webhooks?is_active=true")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(w["is_active"] is True for w in result)
        
        # Test limit
        response = client.get("/api/v1/monitoring/integrations/webhooks?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1 