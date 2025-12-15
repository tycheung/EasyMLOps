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



# Tests for Alerts monitoring routes

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
    
    def test_list_alert_rules(self, client, test_session):
        """Test listing all alert rules"""
        from app.models.monitoring import AlertRuleDB
        from datetime import datetime
        
        # Create test data
        rule1 = AlertRuleDB(
            id="rule_1",
            rule_name="rule_1",
            metric_name="latency",
            condition="gt",
            threshold_value=1000.0,
            severity="warning",
            component="api_server",
            is_active=True,
            created_at=datetime.utcnow()
        )
        rule2 = AlertRuleDB(
            id="rule_2",
            rule_name="rule_2",
            metric_name="error_rate",
            condition="gt",
            threshold_value=5.0,
            severity="critical",
            component="model_service",
            model_id="model_123",
            is_active=False,
            created_at=datetime.utcnow()
        )
        
        test_session.add(rule1)
        test_session.add(rule2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/alert-rules")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        rule_ids = [r["id"] for r in result]
        assert "rule_1" in rule_ids
        assert "rule_2" in rule_ids
        
        # Test filtering by is_active
        response = client.get("/api/v1/monitoring/alert-rules?is_active=true")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(r["is_active"] is True for r in result)
        
        # Test filtering by model_id
        response = client.get("/api/v1/monitoring/alert-rules?model_id=model_123")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        
        # Test filtering by severity
        response = client.get("/api/v1/monitoring/alert-rules?severity=critical")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(r["severity"] == "critical" for r in result)
        
        # Test limit
        response = client.get("/api/v1/monitoring/alert-rules?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1




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




