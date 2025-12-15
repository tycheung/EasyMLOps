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


class TestAnalyticsGET:
    """Test analytics GET endpoints"""
    
    def test_list_comparative_analytics(self, client, test_session):
        """Test listing all comparative analytics"""
        from app.models.monitoring import ComparativeAnalyticsDB
        from datetime import datetime
        
        analytics1 = ComparativeAnalyticsDB(
            id="analytics_1",
            comparison_type="model_comparison",
            comparison_name="Model A vs B",
            entity_ids=["model_a", "model_b"],
            entity_types=["model", "model"],
            comparison_metrics={"accuracy": True},
            comparison_results={"model_a": {"accuracy": 0.9}, "model_b": {"accuracy": 0.85}},
            time_window_start=datetime.utcnow() - timedelta(days=7),
            time_window_end=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        analytics2 = ComparativeAnalyticsDB(
            id="analytics_2",
            comparison_type="deployment_comparison",
            comparison_name="Deployment A vs B",
            entity_ids=["deploy_a", "deploy_b"],
            entity_types=["deployment", "deployment"],
            comparison_metrics={"latency": True},
            comparison_results={"deploy_a": {"latency": 50}, "deploy_b": {"latency": 45}},
            time_window_start=datetime.utcnow() - timedelta(days=7),
            time_window_end=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        test_session.add(analytics1)
        test_session.add(analytics2)
        test_session.commit()
        
        response = client.get("/api/v1/monitoring/analytics/comparative")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        analytics_ids = [a["id"] for a in result]
        assert "analytics_1" in analytics_ids
        assert "analytics_2" in analytics_ids
        
        # Test limit
        response = client.get("/api/v1/monitoring/analytics/comparative?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1
    
    def test_list_dashboards(self, client, test_session):
        """Test listing all custom dashboards"""
        from app.models.monitoring import CustomDashboardDB
        from datetime import datetime
        
        dashboard1 = CustomDashboardDB(
            id="dashboard_1",
            dashboard_name="Dashboard 1",
            dashboard_config={"layout": "grid"},
            selected_metrics=["accuracy", "latency"],
            is_shared=True,
            created_at=datetime.utcnow()
        )
        dashboard2 = CustomDashboardDB(
            id="dashboard_2",
            dashboard_name="Dashboard 2",
            dashboard_config={"layout": "list"},
            selected_metrics=["error_rate"],
            is_shared=False,
            created_at=datetime.utcnow()
        )
        
        test_session.add(dashboard1)
        test_session.add(dashboard2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/analytics/dashboards")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        
        # Test filtering by is_shared
        response = client.get("/api/v1/monitoring/analytics/dashboards?is_shared=true")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(d["is_shared"] is True for d in result)
        
        # Test limit
        response = client.get("/api/v1/monitoring/analytics/dashboards?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1
    
    def test_list_reports(self, client, test_session):
        """Test listing all automated reports"""
        from app.models.monitoring import AutomatedReportDB
        from datetime import datetime
        
        report1 = AutomatedReportDB(
            id="report_1",
            report_name="Weekly Report",
            report_type="performance",
            schedule_type="weekly",
            report_config={"format": "pdf", "include_charts": True},
            delivery_method=["email"],
            recipients=["admin@example.com"],
            is_active=True,
            created_at=datetime.utcnow()
        )
        report2 = AutomatedReportDB(
            id="report_2",
            report_name="Monthly Report",
            report_type="usage",
            schedule_type="monthly",
            report_config={"format": "html", "include_charts": False},
            delivery_method=["slack"],
            recipients=["#monitoring"],
            is_active=False,
            created_at=datetime.utcnow()
        )
        
        test_session.add(report1)
        test_session.add(report2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/analytics/reports")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        
        # Test filtering by is_active
        response = client.get("/api/v1/monitoring/analytics/reports?is_active=true")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(r["is_active"] is True for r in result)
        
        # Test filtering by report_type
        response = client.get("/api/v1/monitoring/analytics/reports?report_type=performance")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(r["report_type"] == "performance" for r in result)
        
        # Test limit
        response = client.get("/api/v1/monitoring/analytics/reports?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1




