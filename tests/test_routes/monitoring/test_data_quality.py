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



# Tests for Data Quality monitoring routes

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




