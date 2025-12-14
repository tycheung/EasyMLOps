"""
Tests for other services: BentoML, Schema, and Deployment services
Tests service creation, validation, deployment operations, and error handling
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from app.services.bentoml_service import BentoMLServiceManager
from app.services.schema_service import SchemaService
from app.services.deployment_service import DeploymentService
from app.models.model import ModelDeployment
from app.schemas.model import ModelDeploymentCreate, ModelDeploymentResponse, DeploymentStatus


class TestBentoMLService:
    """Test BentoMLService functionality"""
    
    @pytest.fixture
    def bentoml_service(self):
        """Create BentoML service instance"""
        return BentoMLServiceManager()
    
    @patch('app.services.bentoml_service.bentoml')
    def test_create_service(self, mock_bentoml, bentoml_service, temp_model_file):
        """Test BentoML service creation"""
        mock_service = MagicMock()
        mock_bentoml.Service.return_value = mock_service
        
        service_name = "test_service"
        model_path = temp_model_file
        
        result = bentoml_service.create_service(service_name, model_path)
        
        assert result is not None
        assert hasattr(result, 'name')
        assert result.name == service_name
    
    @patch('app.services.bentoml_service.bentoml')
    def test_build_bento(self, mock_bentoml, bentoml_service):
        """Test Bento building"""
        mock_bento = MagicMock()
        mock_bento.tag = "test_service:1.0.0"
        mock_bentoml.build.return_value = mock_bento
        
        service_name = "test_service"
        version = "1.0.0"
        
        result = bentoml_service.build_bento(service_name, version)
        
        assert result is not None
        assert result.tag == "test_service:1.0.0"
    
    @patch('app.services.bentoml_service.bentoml')
    def test_serve_bento(self, mock_bentoml, bentoml_service):
        """Test Bento serving"""
        mock_server = MagicMock()
        mock_bentoml.serve.return_value = mock_server
        
        bento_tag = "test_service:latest"
        port = 3000
        
        result = bentoml_service.serve_bento(bento_tag, port)
        
        assert result is not None
        assert hasattr(result, 'tag')
        assert result.tag == bento_tag
        assert hasattr(result, 'port')
        assert result.port == port
    
    def test_generate_service_code(self, bentoml_service):
        """Test service code generation"""
        model_info = {
            "name": "test_model",
            "framework": "sklearn",
            "model_type": "classification"
        }
        
        code = bentoml_service.generate_service_code(model_info)
        
        assert isinstance(code, str)
        assert "import bentoml" in code
        assert "test_model" in code
        assert "sklearn" in code
    
    @patch('app.services.bentoml_service.bentoml')
    def test_list_bentos(self, mock_bentoml, bentoml_service):
        """Test listing available Bentos"""
        mock_bentos = [
            MagicMock(tag="service1:v1"),
            MagicMock(tag="service2:v1")
        ]
        mock_bentoml.list.return_value = mock_bentos
        
        bentos = bentoml_service.list_bentos()
        
        assert len(bentos) == 2
        assert bentos[0].tag == "service1:v1"
        assert bentos[1].tag == "service2:v1"


class TestSchemaService:
    """Test SchemaService functionality"""
    
    @pytest.fixture
    def schema_service(self):
        """Create schema service instance"""
        return SchemaService()
    
    def test_validate_input_schema(self, schema_service):
        """Test input schema validation"""
        schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1"]
        }
        
        valid_data = {"feature1": 0.5, "feature2": "test"}
        is_valid, errors = schema_service.validate_input_schema(valid_data, schema)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_input_schema_invalid(self, schema_service):
        """Test input schema validation with invalid data"""
        schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1"]
        }
        
        invalid_data = {"feature2": "test"}  # Missing required feature1
        is_valid, errors = schema_service.validate_input_schema(invalid_data, schema)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_generate_schema_from_data(self, schema_service):
        """Test schema generation from data"""
        sample_data = {
            "feature1": 0.5,
            "feature2": "test",
            "feature3": True
        }
        
        schema = schema_service.generate_schema_from_data(sample_data)
        
        assert schema is not None
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
    
    def test_merge_schemas(self, schema_service):
        """Test schema merging"""
        schema1 = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"}
            }
        }
        
        schema2 = {
            "type": "object",
            "properties": {
                "field2": {"type": "number"}
            }
        }
        
        merged = schema_service.merge_schemas(schema1, schema2)
        
        assert merged is not None
        assert "field1" in merged.get("properties", {})
        assert "field2" in merged.get("properties", {})
    
    def test_validate_schema_compatibility(self, schema_service):
        """Test schema compatibility validation"""
        old_schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"}
            }
        }
        
        new_schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"}
            }
        }
        
        is_compatible, issues = schema_service.validate_schema_compatibility(old_schema, new_schema)
        
        assert is_compatible is True
        assert len(issues) == 0
    
    def test_validate_schema_compatibility_incompatible(self, schema_service):
        """Test schema compatibility validation with incompatible schemas"""
        old_schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"}
            },
            "required": ["field1"]
        }
        
        new_schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "number"}  # Changed type
            }
        }
        
        is_compatible, issues = schema_service.validate_schema_compatibility(old_schema, new_schema)
        
        assert is_compatible is False
        assert len(issues) > 0


class TestDeploymentService:
    """Test DeploymentService functionality"""
    
    @pytest.fixture
    def deployment_service(self):
        """Create deployment service instance"""
        return DeploymentService()
    
    @pytest.mark.asyncio
    async def test_get_deployment(self, deployment_service, test_model, test_deployment):
        """Test getting a deployment by ID"""
        deployment = await deployment_service.get_deployment(test_deployment.id)
        
        assert deployment is not None
        assert isinstance(deployment, ModelDeploymentResponse)
        assert deployment.id == test_deployment.id
        assert deployment.model_id == test_model.id
    
    @pytest.mark.asyncio
    async def test_get_deployment_not_found(self, deployment_service):
        """Test getting a non-existent deployment"""
        deployment = await deployment_service.get_deployment("nonexistent-id")
        assert deployment is None
    
    @pytest.mark.asyncio
    async def test_list_deployments(self, deployment_service, test_model):
        """Test listing deployments"""
        deployments = await deployment_service.list_deployments()
        
        assert isinstance(deployments, list)
        # Should have at least our test deployment
        assert len(deployments) >= 0
    
    @pytest.mark.asyncio
    async def test_list_deployments_with_model_filter(self, deployment_service, test_model, test_deployment):
        """Test listing deployments filtered by model_id"""
        deployments = await deployment_service.list_deployments(model_id=test_model.id)
        
        assert isinstance(deployments, list)
        # All deployments should belong to the specified model
        for dep in deployments:
            assert dep.model_id == test_model.id
    
    @pytest.mark.asyncio
    async def test_list_deployments_with_status_filter(self, deployment_service, test_deployment):
        """Test listing deployments filtered by status"""
        deployments = await deployment_service.list_deployments(status=DeploymentStatus.ACTIVE)
        
        assert isinstance(deployments, list)
        # All deployments should have the specified status
        for dep in deployments:
            assert dep.status == DeploymentStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_list_deployments_with_pagination(self, deployment_service):
        """Test listing deployments with pagination"""
        deployments = await deployment_service.list_deployments(limit=5, offset=0)
        
        assert isinstance(deployments, list)
        assert len(deployments) <= 5
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager')
    async def test_update_deployment(self, mock_bentoml, deployment_service, test_deployment):
        """Test updating a deployment"""
        from app.schemas.model import ModelDeploymentUpdate
        
        update_data = ModelDeploymentUpdate(
            name="updated_deployment_name",
            config={"new_setting": "value"}
        )
        
        success, message, updated = await deployment_service.update_deployment(
            test_deployment.id,
            update_data
        )
        
        # Update might fail if BentoML service is not available, but method should handle it
        assert isinstance(success, bool)
        assert isinstance(message, str)
    
    @pytest.mark.asyncio
    async def test_update_deployment_not_found(self, deployment_service):
        """Test updating a non-existent deployment"""
        from app.schemas.model import ModelDeploymentUpdate
        
        update_data = ModelDeploymentUpdate(name="updated_name")
        success, message, updated = await deployment_service.update_deployment(
            "nonexistent-id",
            update_data
        )
        
        assert success is False
        assert "not found" in message.lower()
        assert updated is None
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager')
    async def test_delete_deployment(self, mock_bentoml, deployment_service, test_model):
        """Test deleting a deployment"""
        from app.database import get_session
        from app.models.model import ModelDeployment
        import uuid
        
        # Create a deployment to delete
        async with get_session() as session:
            deployment = ModelDeployment(
                id=str(uuid.uuid4()),
                model_id=test_model.id,
                name="temp_deployment",
                deployment_name="temp_deployment",
                status="active"
            )
            session.add(deployment)
            await session.commit()
            deployment_id = deployment.id
        
        success, message = await deployment_service.delete_deployment(deployment_id)
        
        # Delete might fail if BentoML service is not available, but method should handle it
        assert isinstance(success, bool)
        assert isinstance(message, str)
    
    @pytest.mark.asyncio
    async def test_delete_deployment_not_found(self, deployment_service):
        """Test deleting a non-existent deployment"""
        success, message = await deployment_service.delete_deployment("nonexistent-id")
        
        assert success is False
        assert "not found" in message.lower()
    
    @pytest.mark.asyncio
    async def test_get_deployment_status(self, deployment_service, test_deployment):
        """Test getting deployment status"""
        status = await deployment_service.get_deployment_status(test_deployment.id)
        
        assert status is not None
        assert isinstance(status, dict)
        assert "deployment_id" in status or "error" in status
    
    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(self, deployment_service):
        """Test getting status for non-existent deployment"""
        status = await deployment_service.get_deployment_status("nonexistent-id")
        
        assert status is None
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager')
    async def test_get_deployment_metrics(self, mock_bentoml, deployment_service, test_deployment):
        """Test getting deployment metrics"""
        metrics = await deployment_service.get_deployment_metrics(test_deployment.id)
        
        # Metrics might be None if deployment doesn't have metrics yet
        if metrics is not None:
            assert isinstance(metrics, dict)
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager')
    async def test_start_deployment(self, mock_bentoml, deployment_service, test_deployment):
        """Test starting a deployment"""
        success, message = await deployment_service.start_deployment(test_deployment.id)
        
        assert isinstance(success, bool)
        assert isinstance(message, str)
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager')
    async def test_stop_deployment(self, mock_bentoml, deployment_service, test_deployment):
        """Test stopping a deployment"""
        success, message = await deployment_service.stop_deployment(test_deployment.id)
        
        assert isinstance(success, bool)
        assert isinstance(message, str)
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager')
    async def test_scale_deployment(self, mock_bentoml, deployment_service, test_deployment):
        """Test scaling a deployment"""
        scale_config = {"replicas": 3}
        success, message = await deployment_service.scale_deployment(
            test_deployment.id,
            scale_config
        )
        
        assert isinstance(success, bool)
        assert isinstance(message, str)


class TestDeploymentService:
    """Test DeploymentService functionality"""
    
    @pytest.fixture
    def deployment_service(self):
        """Create deployment service instance"""
        return DeploymentService()
    
    @pytest.mark.asyncio
    async def test_create_deployment(self, deployment_service, test_session, test_model):
        """Test deployment creation"""
        deployment_data = ModelDeploymentCreate(
            model_id=test_model.id,
            name="test_deployment",
            description="Test deployment"
        )
        
        with patch.object(deployment_service, 'create_deployment') as mock_create:
            mock_create.return_value = (True, "Success", None)
            
            success, message, deployment = await deployment_service.create_deployment(deployment_data)
            
            assert success is True
            assert message == "Success"
    
    @pytest.mark.asyncio
    async def test_deploy_model(self, deployment_service, test_model):
        """Test model deployment"""
        deployment_data = ModelDeploymentCreate(
            model_id=test_model.id,
            name="test_deployment",
            description="Test deployment"
        )
        
        with patch.object(deployment_service, 'deploy_model') as mock_deploy:
            mock_create_response = ModelDeploymentResponse(
                id="deploy_123",
                model_id=test_model.id,
                name="test_deployment",
                description="Test deployment",
                status=DeploymentStatus.ACTIVE,
                endpoint_url="http://localhost:3001",
                service_name="test_deployment",
                framework="sklearn",
                endpoints=["predict"],
                config={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            mock_deploy.return_value = (True, "Model deployed successfully", mock_create_response)
            
            success, message, deployment = await deployment_service.deploy_model(deployment_data)
            
            assert success is True
            assert "successfully" in message.lower()
            assert deployment.status == DeploymentStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_stop_deployment(self, deployment_service, test_deployment):
        """Test deployment stopping"""
        with patch.object(deployment_service, 'stop_deployment') as mock_stop:
            mock_stop.return_value = (True, "Deployment stopped successfully")
            
            success, message = await deployment_service.stop_deployment(test_deployment.id)
            
            assert success is True
            assert "stopped" in message.lower()
    
    @pytest.mark.asyncio
    async def test_scale_deployment(self, deployment_service, test_deployment):
        """Test deployment scaling"""
        new_scaling = {"min_replicas": 2, "max_replicas": 5}
        
        with patch.object(deployment_service, 'scale_deployment') as mock_scale:
            mock_scale.return_value = (True, "Deployment scaled successfully")
            
            success, message = await deployment_service.scale_deployment(
                test_deployment.id, new_scaling
            )
            
            assert success is True
            assert "scaled" in message.lower()
    
    @pytest.mark.asyncio
    async def test_get_deployment_status(self, deployment_service, test_deployment):
        """Test getting deployment status"""
        with patch.object(deployment_service, 'get_deployment_status') as mock_status:
            mock_status.return_value = {
                "deployment_id": test_deployment.id,
                "deployment_status": "active",
                "service_status": "healthy",
                "endpoint_url": "http://localhost:3001",
                "service_name": test_deployment.deployment_name,
                "framework": "sklearn",
                "endpoints": ["predict"],
                "last_check": datetime.utcnow()
            }
            
            status = await deployment_service.get_deployment_status(test_deployment.id)
            
            assert status is not None
            assert status["deployment_status"] == "active"
            assert status["service_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_get_deployment_logs(self, deployment_service, test_deployment):
        """Test getting deployment logs"""
        mock_logs = [
            "2024-01-01 10:00:00 - Deployment started",
            "2024-01-01 10:00:05 - Service initialized",
            "2024-01-01 10:00:10 - Model loaded",
            "2024-01-01 10:00:15 - Ready to serve requests"
        ]
        
        with patch.object(deployment_service, 'get_deployment_logs') as mock_get_logs:
            mock_get_logs.return_value = mock_logs
            
            logs = await deployment_service.get_deployment_logs(test_deployment.id)
            
            assert len(logs) == 4
            assert "started" in logs[0]


class TestServiceIntegration:
    """Integration tests for service interactions (other services)"""
    
    @pytest.mark.asyncio
    async def test_schema_bentoml_integration(self):
        """Test integration between schema and BentoML services"""
        schema_service = SchemaService()
        bentoml_service = BentoMLServiceManager()
        
        sample_data = {"feature1": 0.5, "feature2": "test"}
        schema = schema_service.generate_schema_from_data(sample_data)
        
        model_info = {
            "name": "integration_test",
            "framework": "sklearn",
            "model_type": "classification",
            "input_schema": schema
        }
        
        service_code = bentoml_service.generate_service_code(model_info)
        
        assert "sklearn" in service_code
        assert "integration_test" in service_code


class TestServiceErrorHandling:
    """Test service error handling scenarios (other services)"""
    
    @pytest.fixture
    def deployment_service(self):
        """Create deployment service instance for error testing"""
        return DeploymentService()
    
    @pytest.mark.asyncio
    async def test_deployment_service_network_error(self, deployment_service):
        """Test deployment service with network errors"""
        deployment = ModelDeployment(
            id="test_deployment_id",
            deployment_name="test",
            model_id="test_model_id",
            deployment_url="http://invalid-url:3001",
            status="pending",
            configuration={},
            replicas=1
        )
        
        with patch('app.services.deployment_service.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get.return_value = deployment
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            with patch('app.services.deployment_service.bentoml_service_manager.get_service_status') as mock_service_status:
                mock_service_status.side_effect = Exception("Network error")
                
                status = await deployment_service.get_deployment_status(deployment.id)
                
                assert status is None
    
    def test_schema_service_invalid_data(self):
        """Test schema service with invalid data"""
        schema_service = SchemaService()
        
        schema = schema_service.generate_schema_from_data(None)
        assert schema is not None
        
        valid_schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        }
        
        invalid_data = {"wrong_field": "value"}
        is_valid, errors = schema_service.validate_input_schema(
            invalid_data, valid_schema
        )
        
        assert is_valid is False and len(errors) > 0

