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



# Tests for Baseline monitoring routes

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




