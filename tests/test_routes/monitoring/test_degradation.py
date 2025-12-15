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



# Tests for Degradation monitoring routes

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




