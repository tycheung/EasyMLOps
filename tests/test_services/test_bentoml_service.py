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



# Tests for Bentoml service

class TestBentoMLService:
    """Test BentoML service comprehensive functionality"""
    
    @pytest.fixture
    def sklearn_model(self):
        """Create a mock sklearn model"""
        model = MagicMock()
        model.predict.return_value = [0.75]
        model.predict_proba.return_value = [[0.25, 0.75]]
        return model
    
    @pytest.fixture
    def sample_model_record(self, test_model):
        """Sample model record for testing"""
        test_model.framework = ModelFramework.SKLEARN
        test_model.model_type = ModelType.CLASSIFICATION
        test_model.file_path = "/tmp/test_model.joblib"
        return test_model
    
    @pytest.mark.asyncio
    @patch('app.services.bentoml_service.get_session')
    @patch('app.services.bentoml_service.aios.path.exists')
    @patch('app.services.bentoml_service.aios.makedirs')
    @patch('app.services.bentoml_service.aiofiles.open')
    @patch('app.services.bentoml_service.asyncio.to_thread')
    @patch('app.services.bentoml_service.bentoml.sklearn.save_model')
    async def test_create_sklearn_service_success(self, mock_save_model, mock_to_thread,
                                                mock_aiofiles_open, mock_makedirs, mock_path_exists,
                                                mock_get_session, sklearn_model, sample_model_record):
        """Test successful sklearn service creation"""
        # Mock database session
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_model_record)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock async file system
        mock_path_exists.return_value = True
        mock_makedirs.return_value = None
        
        # Mock async file writing
        mock_file = AsyncMock()
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file
        
        # Mock asyncio.to_thread for joblib.load and bentoml.sklearn.save_model
        def side_effect_to_thread(func, *args, **kwargs):
            # Handle both function objects and string representations  
            func_name = getattr(func, '__name__', str(func))
            if func_name == 'load' or 'load' in func_name:  # joblib.load
                return sklearn_model
            elif 'save_model' in func_name:  # bentoml.sklearn.save_model
                mock_bento_model = MagicMock()
                mock_bento_model.__str__ = MagicMock(return_value="sklearn_model_123:abc")
                return mock_bento_model
            return None
        
        mock_to_thread.side_effect = side_effect_to_thread
        
        success, message, service_info = await bentoml_service_manager.create_service_for_model(
            sample_model_record.id
        )
        
        assert success is True
        assert "successfully" in message.lower()
        assert service_info is not None
        assert service_info['framework'] == 'sklearn'
        assert 'predict' in service_info['endpoints']
        assert 'predict_proba' in service_info['endpoints']
    
    @pytest.mark.asyncio
    @patch('app.services.bentoml_service.get_session')
    async def test_create_service_model_not_found(self, mock_get_session):
        """Test service creation with non-existent model"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message, service_info = await bentoml_service_manager.create_service_for_model("nonexistent")
        
        assert success is False
        assert "not found" in message.lower()
        assert service_info == {}
    
    @pytest.mark.asyncio
    @patch('app.services.bentoml_service.get_session')
    @patch('app.services.bentoml_service.os.path.exists')
    async def test_create_service_model_file_not_found(self, mock_path_exists, mock_get_session, sample_model_record):
        """Test service creation when model file doesn't exist"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_model_record)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        mock_path_exists.return_value = False
        
        success, message, service_info = await bentoml_service_manager.create_service_for_model(sample_model_record.id)
        
        assert success is False
        assert "file not found" in message.lower()
        assert service_info == {}
    
    @pytest.mark.asyncio
    async def test_deploy_service_success(self):
        """Test successful service deployment"""
        success, message, deployment_info = await bentoml_service_manager.deploy_service(
            "model_service_123", {"replicas": 1}
        )
        
        assert success is True
        assert "successfully" in message.lower()
        assert deployment_info is not None
        assert 'endpoint_url' in deployment_info
        assert 'deployment_id' in deployment_info
        assert deployment_info['status'] == 'deployed'
    
    @pytest.mark.asyncio
    async def test_undeploy_service_success(self):
        """Test successful service undeployment"""
        # First register a service
        bentoml_service_manager.active_services["test_service"] = {}
        
        success, message = await bentoml_service_manager.undeploy_service("test_service")
        
        assert success is True
        assert "successfully" in message.lower()
        assert "test_service" not in bentoml_service_manager.active_services
    
    @pytest.mark.asyncio
    async def test_get_service_status_active(self):
        """Test getting status of active service"""
        bentoml_service_manager.active_services["test_service"] = {
            'service_name': 'test_service'
        }
        
        status = await bentoml_service_manager.get_service_status("test_service")
        
        assert status['status'] == 'active'
        assert status['service_name'] == "test_service"
    
    @pytest.mark.asyncio
    async def test_get_service_status_inactive(self):
        """Test getting status of inactive service"""
        status = await bentoml_service_manager.get_service_status("nonexistent_service")
        
        assert status['status'] == 'inactive'
        assert status['service_name'] == "nonexistent_service"
    
    def test_generate_sklearn_service_code(self, sample_model_record):
        """Test sklearn service code generation"""
        bento_model_tag = "sklearn_model_123:abc"
        config = {"timeout": 30}
        
        service_code = bentoml_service_manager._generate_sklearn_service_code(
            sample_model_record, bento_model_tag, config
        )
        
        assert isinstance(service_code, str)
        assert "sklearn" in service_code.lower()
        assert bento_model_tag in service_code
        assert "predict" in service_code
        assert "predict_proba" in service_code  # For classification models
        assert "@bentoml.api" in service_code
    
    def test_generate_tensorflow_service_code(self, sample_model_record):
        """Test TensorFlow service code generation"""
        sample_model_record.framework = ModelFramework.TENSORFLOW
        bento_model_tag = "tensorflow_model_123:abc"
        config = {}
        
        service_code = bentoml_service_manager._generate_tensorflow_service_code(
            sample_model_record, bento_model_tag, config
        )
        
        assert isinstance(service_code, str)
        assert "tensorflow" in service_code.lower()
        assert bento_model_tag in service_code
        assert "predict" in service_code
        assert "@bentoml.api" in service_code
    
    def test_generate_pytorch_service_code(self, sample_model_record):
        """Test PyTorch service code generation"""
        sample_model_record.framework = ModelFramework.PYTORCH
        bento_model_tag = "pytorch_model_123:abc"
        config = {}
        
        service_code = bentoml_service_manager._generate_pytorch_service_code(
            sample_model_record, bento_model_tag, config
        )
        
        assert isinstance(service_code, str)
        assert "pytorch" in service_code.lower()
        assert bento_model_tag in service_code
        assert "predict" in service_code
    
    @pytest.mark.asyncio
    @patch('app.services.bentoml_service.SKLEARN_AVAILABLE', False)
    async def test_create_sklearn_service_unavailable(self, sample_model_record):
        """Test sklearn service creation when sklearn is not available"""
        success, message, service_info = await bentoml_service_manager._create_sklearn_service(
            sample_model_record, "test_service", {}
        )
        
        assert success is False
        assert "not available" in message.lower()
        assert service_info == {}




