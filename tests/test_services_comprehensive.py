"""
Comprehensive tests for all service layer components
Tests deployment, monitoring, schema, and BentoML services business logic
"""

import pytest
import asyncio
import tempfile
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from datetime import datetime, timedelta
from pathlib import Path

from app.services.deployment_service import deployment_service
from app.services.monitoring_service import monitoring_service
from app.services.schema_service import schema_service
from app.services.bentoml_service import bentoml_service_manager
from app.models.model import Model, ModelDeployment
from app.schemas.model import ModelDeploymentCreate, ModelStatus, DeploymentStatus, ModelFramework, ModelType


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
    @patch('app.services.deployment_service.bentoml_service_manager.create_service_for_model')
    @patch('app.services.deployment_service.bentoml_service_manager.deploy_service')
    async def test_create_deployment_success(self, mock_deploy_service, mock_create_service,
                                           mock_get_session, deployment_create_data, test_model):
        """Test successful deployment creation"""
        # Mock database session
        mock_session = MagicMock()
        mock_session.get.return_value = test_model
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock BentoML service creation
        service_info = {
            'service_name': 'model_service_123',
            'bento_model_tag': 'sklearn_model:abc123',
            'framework': 'sklearn',
            'endpoints': ['predict', 'predict_proba']
        }
        mock_create_service.return_value = (True, "Service created", service_info)
        
        # Mock service deployment
        deploy_info = {'endpoint_url': 'http://localhost:3000/model_service_123'}
        mock_deploy_service.return_value = (True, "Deployed", deploy_info)
        
        # Create deployment
        success, message, deployment = await deployment_service.create_deployment(deployment_create_data)
        
        assert success is True
        assert "successfully" in message.lower()
        assert deployment is not None
        mock_create_service.assert_called_once()
        mock_deploy_service.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_create_deployment_model_not_found(self, mock_get_session, deployment_create_data):
        """Test deployment creation with non-existent model"""
        mock_session = MagicMock()
        mock_session.get.return_value = None
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message, deployment = await deployment_service.create_deployment(deployment_create_data)
        
        assert success is False
        assert "not found" in message.lower()
        assert deployment is None
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_create_deployment_model_not_validated(self, mock_get_session, deployment_create_data, test_model):
        """Test deployment creation with unvalidated model"""
        test_model.status = ModelStatus.UPLOADED
        
        mock_session = MagicMock()
        mock_session.get.return_value = test_model
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message, deployment = await deployment_service.create_deployment(deployment_create_data)
        
        assert success is False
        assert "validated" in message.lower()
        assert deployment is None
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.create_service_for_model')
    async def test_create_deployment_bentoml_service_failure(self, mock_create_service, mock_get_session,
                                                           deployment_create_data, test_model):
        """Test deployment creation when BentoML service creation fails"""
        mock_session = MagicMock()
        mock_session.get.return_value = test_model
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock BentoML service creation failure
        mock_create_service.return_value = (False, "Service creation failed", {})
        
        success, message, deployment = await deployment_service.create_deployment(deployment_create_data)
        
        assert success is False
        assert "Failed to create BentoML service" in message
        assert deployment is None
    
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
        mock_session.get.side_effect = [deployment, test_model]
        mock_session.exec.return_value.first.return_value = None  # No other deployments
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
        mock_session.get.return_value = None
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
        mock_session.get.return_value = deployment
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


class TestMonitoringService:
    """Test monitoring service comprehensive functionality"""
    
    @pytest.mark.asyncio
    async def test_log_prediction_success(self, test_model):
        """Test successful prediction logging"""
        input_data = {"feature1": 0.5, "feature2": "test"}
        output_data = {"prediction": 0.75, "confidence": 0.85}
        latency_ms = 45.2
        
        # Mock the monitoring service method
        with patch.object(monitoring_service, '_save_prediction_log') as mock_save:
            mock_save.return_value = True
            
            result = await monitoring_service.log_prediction(
                model_id=test_model.id,
                deployment_id="deploy_123",
                input_data=input_data,
                output_data=output_data,
                latency_ms=latency_ms,
                api_endpoint="/predict/deploy_123",
                success=True
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_get_system_health(self):
        """Test system health monitoring"""
        # Mock system metrics
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 45.2
            mock_memory.return_value.percent = 68.5
            mock_disk.return_value.percent = 78.3
            
            health = await monitoring_service.get_system_health()
            
            assert health is not None
            assert 'cpu_usage' in health
            assert 'memory_usage' in health
            assert 'disk_usage' in health
            assert 'status' in health
            assert health['cpu_usage'] == 45.2
    
    @pytest.mark.asyncio
    async def test_get_model_performance_metrics(self, test_model):
        """Test getting model performance metrics"""
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        # Mock database query
        with patch.object(monitoring_service, '_query_prediction_logs') as mock_query:
            mock_logs = [
                {
                    'latency_ms': 45.0,
                    'success': True,
                    'created_at': datetime.utcnow()
                },
                {
                    'latency_ms': 52.0,
                    'success': True,
                    'created_at': datetime.utcnow()
                },
                {
                    'latency_ms': 100.0,
                    'success': False,
                    'created_at': datetime.utcnow()
                }
            ]
            mock_query.return_value = mock_logs
            
            metrics = await monitoring_service.get_model_performance_metrics(
                model_id=test_model.id,
                start_time=start_time,
                end_time=end_time
            )
            
            assert metrics is not None
            assert 'total_requests' in metrics
            assert 'successful_requests' in metrics
            assert 'failed_requests' in metrics
            assert 'avg_latency_ms' in metrics
            assert 'success_rate' in metrics
            assert metrics['total_requests'] == 3
            assert metrics['successful_requests'] == 2
            assert metrics['failed_requests'] == 1
    
    @pytest.mark.asyncio
    async def test_create_alert(self, test_model):
        """Test alert creation"""
        alert_data = {
            'alert_type': 'high_error_rate',
            'severity': 'warning',
            'message': 'Model error rate above threshold',
            'model_id': test_model.id,
            'threshold_value': 5.0,
            'actual_value': 7.5
        }
        
        with patch.object(monitoring_service, '_save_alert') as mock_save:
            mock_alert = {
                'id': 'alert_123',
                **alert_data,
                'created_at': datetime.utcnow(),
                'resolved': False
            }
            mock_save.return_value = mock_alert
            
            alert = await monitoring_service.create_alert(**alert_data)
            
            assert alert is not None
            assert alert['id'] == 'alert_123'
            assert alert['alert_type'] == 'high_error_rate'
            assert alert['severity'] == 'warning'
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self):
        """Test alert resolution"""
        with patch.object(monitoring_service, '_update_alert_status') as mock_update:
            mock_update.return_value = True
            
            result = await monitoring_service.resolve_alert("alert_123")
            
            assert result is True
            mock_update.assert_called_once_with("alert_123", resolved=True)


class TestSchemaService:
    """Test schema service comprehensive functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for schema generation"""
        return [
            {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft": 2000,
                "location": "suburban",
                "price": 350000
            },
            {
                "bedrooms": 4,
                "bathrooms": 3.0,
                "sqft": 2500,
                "location": "urban",
                "price": 450000
            },
            {
                "bedrooms": 2,
                "bathrooms": 1.5,
                "sqft": 1200,
                "location": "rural",
                "price": 250000
            }
        ]
    
    @pytest.mark.asyncio
    async def test_generate_schema_from_data(self, sample_data):
        """Test schema generation from sample data"""
        schema = await schema_service.generate_schema_from_data(
            sample_data=sample_data,
            schema_type="input",
            include_target=False
        )
        
        assert schema is not None
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "bedrooms" in schema["properties"]
        assert "bathrooms" in schema["properties"]
        assert "sqft" in schema["properties"]
        assert "location" in schema["properties"]
        assert "price" not in schema["properties"]  # Target excluded
    
    @pytest.mark.asyncio
    async def test_generate_schema_with_target(self, sample_data):
        """Test schema generation including target variable"""
        schema = await schema_service.generate_schema_from_data(
            sample_data=sample_data,
            schema_type="input",
            include_target=True
        )
        
        assert "price" in schema["properties"]
    
    @pytest.mark.asyncio
    async def test_validate_input_schema_success(self):
        """Test successful input validation"""
        schema = {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "integer", "minimum": 1},
                "bathrooms": {"type": "number", "minimum": 0.5},
                "sqft": {"type": "number", "minimum": 100}
            },
            "required": ["bedrooms", "bathrooms", "sqft"]
        }
        
        data = {
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft": 2000
        }
        
        is_valid, errors = await schema_service.validate_input_schema(data, schema)
        
        assert is_valid is True
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_input_schema_failure(self):
        """Test input validation failure"""
        schema = {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "integer", "minimum": 1},
                "bathrooms": {"type": "number", "minimum": 0.5}
            },
            "required": ["bedrooms", "bathrooms"]
        }
        
        data = {
            "bedrooms": "not_a_number",  # Wrong type
            # Missing required bathrooms
        }
        
        is_valid, errors = await schema_service.validate_input_schema(data, schema)
        
        assert is_valid is False
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_compare_schemas_compatible(self):
        """Test schema compatibility comparison"""
        schema1 = {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "integer"},
                "bathrooms": {"type": "number"}
            },
            "required": ["bedrooms", "bathrooms"]
        }
        
        schema2 = {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "integer"},
                "bathrooms": {"type": "number"},
                "garage": {"type": "boolean"}  # Optional field added
            },
            "required": ["bedrooms", "bathrooms"]
        }
        
        comparison = await schema_service.compare_schemas(schema1, schema2)
        
        assert comparison["compatible"] is True
        assert comparison["compatibility_score"] > 0.8
        assert len(comparison["differences"]) > 0
        assert len(comparison["breaking_changes"]) == 0
    
    @pytest.mark.asyncio
    async def test_compare_schemas_incompatible(self):
        """Test schema incompatibility detection"""
        schema1 = {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "integer"}
            },
            "required": ["bedrooms"]
        }
        
        schema2 = {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "string"}  # Type changed
            },
            "required": ["bedrooms"]
        }
        
        comparison = await schema_service.compare_schemas(schema1, schema2)
        
        assert comparison["compatible"] is False
        assert comparison["compatibility_score"] < 0.5
        assert len(comparison["breaking_changes"]) > 0
    
    @pytest.mark.asyncio
    async def test_convert_to_openapi_schema(self):
        """Test schema conversion to OpenAPI format"""
        json_schema = {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "integer", "minimum": 1},
                "bathrooms": {"type": "number"}
            },
            "required": ["bedrooms"]
        }
        
        openapi_schema = await schema_service.convert_to_openapi_schema(
            json_schema, include_examples=True
        )
        
        assert openapi_schema is not None
        assert "properties" in openapi_schema
        assert openapi_schema["type"] == "object"
        # OpenAPI specific features should be added
    
    @pytest.mark.asyncio
    @patch('app.database.get_session')
    async def test_save_model_schema(self, mock_get_session, test_model):
        """Test saving model schema"""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        schema_data = {
            "type": "object",
            "properties": {"feature1": {"type": "number"}}
        }
        
        result = await schema_service.save_model_schema(
            model_id=test_model.id,
            schema_type="input",
            schema_data=schema_data,
            version="1.0"
        )
        
        assert result is not None
        mock_session.add.assert_called_once()


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
    @patch('app.services.bentoml_service.os.path.exists')
    @patch('app.services.bentoml_service.joblib.load')
    @patch('app.services.bentoml_service.bentoml.sklearn.save_model')
    async def test_create_sklearn_service_success(self, mock_save_model, mock_joblib_load,
                                                mock_path_exists, mock_get_session,
                                                sklearn_model, sample_model_record):
        """Test successful sklearn service creation"""
        # Mock database session
        mock_session = MagicMock()
        mock_session.get.return_value = sample_model_record
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock file system
        mock_path_exists.return_value = True
        mock_joblib_load.return_value = sklearn_model
        
        # Mock BentoML
        mock_bento_model = MagicMock()
        mock_bento_model.tag = "sklearn_model_123:abc"
        mock_save_model.return_value = mock_bento_model
        
        # Mock file operations
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('os.makedirs'):
            
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
        mock_session.get.return_value = None
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
        mock_session.get.return_value = sample_model_record
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


class TestServiceIntegration:
    """Integration tests for service layer interactions"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.create_service_for_model')
    @patch('app.services.deployment_service.bentoml_service_manager.deploy_service')
    @patch('app.services.monitoring_service.log_prediction')
    async def test_deployment_monitoring_integration(self, mock_log_prediction, mock_deploy_service,
                                                   mock_create_service, mock_get_session, test_model):
        """Test integration between deployment and monitoring services"""
        # Setup deployment
        deployment_data = ModelDeploymentCreate(
            model_id=test_model.id,
            name="integration_test",
            description="Integration test deployment"
        )
        
        # Mock database session
        mock_session = MagicMock()
        mock_session.get.return_value = test_model
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock service creation and deployment
        service_info = {'service_name': 'test_service', 'framework': 'sklearn', 'endpoints': ['predict']}
        mock_create_service.return_value = (True, "Created", service_info)
        mock_deploy_service.return_value = (True, "Deployed", {'endpoint_url': 'http://localhost:3000'})
        
        # Create deployment
        success, message, deployment = await deployment_service.create_deployment(deployment_data)
        assert success is True
        
        # Mock monitoring
        mock_log_prediction.return_value = True
        
        # Log a prediction for the deployment
        prediction_result = await monitoring_service.log_prediction(
            model_id=test_model.id,
            deployment_id=deployment.id,
            input_data={"feature1": 0.5},
            output_data={"prediction": 0.75},
            latency_ms=45.0,
            api_endpoint=f"/predict/{deployment.id}",
            success=True
        )
        
        assert prediction_result is True
        mock_log_prediction.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.schema_service.generate_schema_from_data')
    @patch('app.services.schema_service.validate_input_schema')
    async def test_schema_validation_workflow(self, mock_validate, mock_generate):
        """Test complete schema generation and validation workflow"""
        # Sample training data
        training_data = [
            {"feature1": 1.0, "feature2": "A", "target": 0},
            {"feature1": 2.0, "feature2": "B", "target": 1}
        ]
        
        # Mock schema generation
        generated_schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string", "enum": ["A", "B"]}
            },
            "required": ["feature1", "feature2"]
        }
        mock_generate.return_value = generated_schema
        
        # Generate schema
        schema = await schema_service.generate_schema_from_data(training_data, "input")
        assert schema == generated_schema
        
        # Mock validation
        mock_validate.return_value = (True, [])
        
        # Validate new data against schema
        new_data = {"feature1": 1.5, "feature2": "A"}
        is_valid, errors = await schema_service.validate_input_schema(new_data, schema)
        
        assert is_valid is True
        assert len(errors) == 0
        mock_generate.assert_called_once()
        mock_validate.assert_called_once() 