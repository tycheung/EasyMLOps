"""
Comprehensive tests for all service layer components
Tests deployment, monitoring, schema, and BentoML services business logic
"""

import pytest
import asyncio
import tempfile
import os
import json
import uuid
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.deployment_service import deployment_service, DeploymentService
from app.services.monitoring_service import monitoring_service
from app.services.schema_service import schema_service, SchemaService
from app.services.bentoml_service import bentoml_service_manager
from app.models.model import Model, ModelDeployment
from app.schemas.model import ModelDeploymentCreate, ModelStatus, DeploymentStatus, ModelFramework, ModelType
from app.schemas.monitoring import AlertSeverity, SystemComponent, ModelPerformanceMetrics, MetricType, DriftType, DriftSeverity, ModelPerformanceHistory, ModelConfidenceMetrics, ModelBaseline, ModelVersionComparison, ABTest, ABTestStatus, ABTestMetrics, ABTestComparison, CanaryDeployment, CanaryDeploymentStatus, CanaryMetrics, ProtectedAttributeConfig, ProtectedAttributeType, BiasFairnessMetrics, DemographicDistribution, ModelExplanation, ExplanationType, FeatureImportance, ImportanceType, OutlierDetection, OutlierDetectionMethod, OutlierType, AnomalyDetection, AnomalyType, DataQualityMetrics, RetrainingJob, RetrainingTriggerType, RetrainingJobStatus, RetrainingTriggerConfig, ModelCard, DataLineage, LineageType, RelationshipType, GovernanceWorkflow, WorkflowType, WorkflowStatus, ComplianceRecord, ComplianceType, ComplianceRecordType, ComplianceRecordStatus, DataRetentionPolicy, TimeSeriesAnalysis, AnalysisType, TrendDirection, ComparativeAnalytics, ComparisonType, CustomDashboard, AutomatedReport, ReportType, ScheduleType, ExternalIntegration, IntegrationType, WebhookConfig, SamplingConfig, SamplingStrategy, MetricAggregationConfig, AggregationMethod, AlertRule, AlertCondition, NotificationChannel, NotificationChannelType, AlertGroup, AlertEscalation, EscalationTriggerCondition, Alert
from app.database import get_session



# Tests for Deployment service

class TestDeploymentService:
    """Test deployment service comprehensive functionality"""
    
    @pytest.fixture
    def deployment_create_data(self, test_model):
        """Sample deployment creation data"""
        return ModelDeploymentCreate(
            model_id=test_model.id,
            name="test_deployment",
            description="Test deployment",
            config={"replicas": 2, "memory": "1Gi"}
        )
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.bentoml_service.bentoml_service_manager.create_service_for_model', new_callable=AsyncMock)
    @patch('app.services.bentoml_service.bentoml_service_manager.deploy_service', new_callable=AsyncMock)
    async def test_create_deployment_success(self, mock_get_session, mock_deploy, mock_create_service, test_model_validated):
        """Test successful deployment creation"""
        mock_create_service.return_value = (
            True, 
            "Service created successfully", 
            {
                "service_name": "test_sklearn_service_abc123", 
                "bento_model_tag": f"{test_model_validated.name.lower()}:{test_model_validated.version}",
                "framework": test_model_validated.framework,
                "endpoints": [{"name": "predict", "input_type": "json", "output_type": "json"}]
            }
        )
        mock_deploy.return_value = (True, "Deployment successful", {"endpoint_url": "http://mock-service/test_sklearn_service_abc123"})
        
        # Mock database session
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=test_model_validated)
        mock_session.execute = AsyncMock()
        mock_session.execute.return_value.first.return_value = None  # No existing deployments
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message, deployment = await deployment_service.create_deployment(
            ModelDeploymentCreate(
                model_id=test_model_validated.id,
                name="Test Deployment",
                description="A comprehensive test deployment",
                config={"cpu": "500m", "memory": "1Gi"}
            )
        )
        
        assert success is True
        assert "successfully" in message.lower()
        assert deployment is not None
        mock_create_service.assert_called_once()
        mock_deploy.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.bentoml_service.bentoml_service_manager.create_service_for_model', new_callable=AsyncMock)
    @patch('app.services.bentoml_service.bentoml_service_manager.deploy_service', new_callable=AsyncMock)
    async def test_create_deployment_model_not_found(self, mock_get_session, mock_deploy, mock_create_service, test_model_validated):
        """Test deployment creation when model is not found"""
        # Mock database session
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)  # Simulate model not found
        mock_get_session.return_value.__aenter__.return_value = mock_session

        success, message, deployment = await deployment_service.create_deployment(
            ModelDeploymentCreate(
                model_id="nonexistent_model",
                name="Test Deployment",
                config={}
            )
        )
        assert success is False
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.bentoml_service.bentoml_service_manager.create_service_for_model', new_callable=AsyncMock)
    @patch('app.services.bentoml_service.bentoml_service_manager.deploy_service', new_callable=AsyncMock)
    async def test_create_deployment_model_not_validated(self, mock_get_session, mock_deploy, mock_create_service, test_model_uploaded):
        """Test deployment creation when model is not validated"""
        # Mock database session
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=test_model_uploaded)  # Model exists but not validated
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message, deployment = await deployment_service.create_deployment(
            ModelDeploymentCreate(
                model_id=test_model_uploaded.id,
                name="Test Deployment",
                config={}
            )
        )
        assert success is False
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.create_service_for_model')
    @patch('app.services.bentoml_service.bentoml_service_manager.deploy_service')
    async def test_create_deployment_bentoml_service_failure(self, mock_create_service, mock_deploy, deployment_service, test_model_validated):
        """Test deployment creation when BentoML service creation fails"""
        mock_create_service.return_value = (False, "BentoML service creation failed", None)
        # mock_deploy should not be called if mock_create_service fails

        success, message, deployment = await deployment_service.create_deployment(
            ModelDeploymentCreate(
                model_id=test_model_validated.id,
                name="Test Deployment",
                config={"cpu": "500m", "memory": "1Gi"}
            )
        )
        assert success is False
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_delete_deployment_success(self, mock_get_session, test_model):
        """Test successful deployment deletion"""
        # Create mock deployment
        deployment = ModelDeployment(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",
            service_name="model_service_123",
            status=DeploymentStatus.ACTIVE
        )
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[deployment, test_model])
        mock_session.execute = AsyncMock()
        mock_session.execute.return_value.first.return_value = None  # No other deployments
        mock_session.delete = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message = await deployment_service.delete_deployment("deploy_123")
        
        assert success is True
        assert "successfully" in message.lower()
        mock_session.delete.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_delete_deployment_not_found(self, mock_get_session):
        """Test deletion of non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message = await deployment_service.delete_deployment("nonexistent")
        
        assert success is False
        assert "not found" in message.lower()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.get_service_status')
    async def test_get_deployment_status(self, mock_get_service_status, mock_get_session, test_model):
        """Test getting deployment status"""
        deployment = ModelDeployment(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",
            service_name="model_service_123",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3000/service",
            framework="sklearn",
            endpoints=["predict"]
        )
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=deployment)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        mock_service_status = {
            'status': 'active',
            'service_name': 'model_service_123'
        }
        mock_get_service_status.return_value = mock_service_status
        
        status = await deployment_service.get_deployment_status("deploy_123")
        
        assert status is not None
        assert status['deployment_id'] == "deploy_123"
        assert status['deployment_status'] == DeploymentStatus.ACTIVE
        assert status['service_status'] == mock_service_status


# NOTE: TestMonitoringService has been split into 15 domain-specific test files:
# - tests/test_services/test_monitoring_performance.py
# - tests/test_services/test_monitoring_health.py
# - tests/test_services/test_monitoring_alerts.py
# - tests/test_services/test_monitoring_drift.py
# - tests/test_services/test_monitoring_degradation.py
# - tests/test_services/test_monitoring_baseline.py
# - tests/test_services/test_monitoring_ab_testing.py
# - tests/test_services/test_monitoring_canary.py
# - tests/test_services/test_monitoring_fairness.py
# - tests/test_services/test_monitoring_explainability.py
# - tests/test_services/test_monitoring_data_quality.py
# - tests/test_services/test_monitoring_lifecycle.py
# - tests/test_services/test_monitoring_governance.py
# - tests/test_services/test_monitoring_analytics.py
# - tests/test_services/test_monitoring_integration.py
# 
# All monitoring service tests have been moved to the test_services/ directory.
# The original TestMonitoringService class has been removed from this file.




