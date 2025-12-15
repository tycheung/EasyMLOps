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



# Tests for Governance monitoring routes

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


class TestGovernanceGET:
    """Test governance GET endpoints"""
    
    def test_list_data_lineage(self, client, test_session):
        """Test listing all data lineage records"""
        from app.models.monitoring import DataLineageDB
        from datetime import datetime
        
        lineage1 = DataLineageDB(
            id="lineage_1",
            lineage_type="data",
            source_id="source_1",
            source_type="dataset",
            target_id="target_1",
            target_type="model",
            relationship_type="derived_from",
            created_at=datetime.utcnow()
        )
        lineage2 = DataLineageDB(
            id="lineage_2",
            lineage_type="model",
            source_id="model_1",
            source_type="model",
            target_id="deployment_1",
            target_type="deployment",
            relationship_type="deployed_as",
            model_id="model_123",
            created_at=datetime.utcnow()
        )
        
        test_session.add(lineage1)
        test_session.add(lineage2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/governance/lineage")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        
        # Test filtering by lineage_type
        response = client.get("/api/v1/monitoring/governance/lineage?lineage_type=data")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(l["lineage_type"] == "data" for l in result)
        
        # Test filtering by model_id
        response = client.get("/api/v1/monitoring/governance/lineage?model_id=model_123")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        
        # Test limit
        response = client.get("/api/v1/monitoring/governance/lineage?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1
    
    def test_list_governance_workflows(self, client, test_session):
        """Test listing all governance workflows"""
        from app.models.monitoring import GovernanceWorkflowDB
        from datetime import datetime
        
        workflow1 = GovernanceWorkflowDB(
            id="workflow_1",
            workflow_type="model_approval",
            workflow_status="pending",
            resource_type="model",
            resource_id="model_1",
            requested_by="user_1",
            requested_at=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        workflow2 = GovernanceWorkflowDB(
            id="workflow_2",
            workflow_type="deployment_approval",
            workflow_status="approved",
            resource_type="deployment",
            resource_id="deploy_1",
            requested_by="user_2",
            requested_at=datetime.utcnow(),
            approved_by="admin_1",
            approved_at=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        test_session.add(workflow1)
        test_session.add(workflow2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/governance/workflows")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        
        # Test filtering by workflow_type
        response = client.get("/api/v1/monitoring/governance/workflows?workflow_type=model_approval")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(w["workflow_type"] == "model_approval" for w in result)
        
        # Test limit
        response = client.get("/api/v1/monitoring/governance/workflows?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1
    
    def test_list_compliance_records(self, client, test_session):
        """Test listing all compliance records"""
        from app.models.monitoring import ComplianceRecordDB
        from datetime import datetime
        
        record1 = ComplianceRecordDB(
            id="compliance_1",
            compliance_type="gdpr",
            record_type="data_access",
            subject_id="user_1",
            subject_type="user",
            request_id="req_1",
            requested_by="user_1",
            requested_at=datetime.utcnow(),
            status="pending",
            created_at=datetime.utcnow()
        )
        record2 = ComplianceRecordDB(
            id="compliance_2",
            compliance_type="hipaa",
            record_type="data_deletion",
            subject_id="patient_1",
            subject_type="patient",
            request_id="req_2",
            requested_by="admin_1",
            requested_at=datetime.utcnow(),
            status="completed",
            created_at=datetime.utcnow()
        )
        
        test_session.add(record1)
        test_session.add(record2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/governance/compliance")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        
        # Test filtering by compliance_type
        response = client.get("/api/v1/monitoring/governance/compliance?compliance_type=gdpr")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(r["compliance_type"] == "gdpr" for r in result)
        
        # Test filtering by record_type
        response = client.get("/api/v1/monitoring/governance/compliance?record_type=data_access")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(r["record_type"] == "data_access" for r in result)
        
        # Test limit
        response = client.get("/api/v1/monitoring/governance/compliance?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1
    
    def test_list_retention_policies(self, client, test_session):
        """Test listing all data retention policies"""
        from app.models.monitoring import DataRetentionPolicyDB
        from datetime import datetime
        
        policy1 = DataRetentionPolicyDB(
            id="policy_1",
            policy_name="30 Day Policy",
            policy_description="Retain for 30 days",
            resource_type="prediction_logs",
            retention_period_days=30,
            action_on_expiry="delete",
            created_at=datetime.utcnow()
        )
        policy2 = DataRetentionPolicyDB(
            id="policy_2",
            policy_name="90 Day Policy",
            policy_description="Retain for 90 days",
            resource_type="model",
            model_id="model_123",
            retention_period_days=90,
            action_on_expiry="archive",
            created_at=datetime.utcnow()
        )
        
        test_session.add(policy1)
        test_session.add(policy2)
        test_session.commit()
        
        # Test listing all
        response = client.get("/api/v1/monitoring/governance/retention-policies")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 2
        
        # Test filtering by resource_type
        response = client.get("/api/v1/monitoring/governance/retention-policies?resource_type=prediction_logs")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert all(p["resource_type"] == "prediction_logs" for p in result)
        
        # Test filtering by model_id
        response = client.get("/api/v1/monitoring/governance/retention-policies?model_id=model_123")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        
        # Test limit
        response = client.get("/api/v1/monitoring/governance/retention-policies?limit=1")
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 1




