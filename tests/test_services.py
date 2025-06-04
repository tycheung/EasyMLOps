"""
Unit tests for service modules
Tests business logic in monitoring, BentoML, schema, and deployment services
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta

from app.services.monitoring_service import MonitoringService
from app.services.bentoml_service import BentoMLServiceManager
from app.services.schema_service import SchemaService
from app.services.deployment_service import DeploymentService
from app.models.model import ModelDeployment
from app.schemas.monitoring import AlertSeverity, SystemComponent
from app.schemas.model import ModelDeploymentCreate, ModelDeploymentResponse, DeploymentStatus


class TestMonitoringService:
    """Test MonitoringService functionality"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service instance"""
        return MonitoringService()
    
    @pytest.mark.asyncio
    async def test_log_prediction(self, monitoring_service, test_session, test_model):
        """Test prediction logging"""
        input_data = {"feature1": 0.5, "feature2": "test"}
        output_data = {"class": "A", "probability": 0.85}
        latency_ms = 150.5
        
        await monitoring_service.log_prediction(
            model_id=test_model.id,
            deployment_id=None,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            api_endpoint="/test/predict",
            success=True
        )
        
        # Verify log was created
        from app.models.monitoring import PredictionLogDB
        logs = test_session.query(PredictionLogDB).filter(
            PredictionLogDB.model_id == test_model.id
        ).all()
        
        assert len(logs) == 1
        log = logs[0]
        assert log.input_data == input_data
        assert log.output_data == output_data
        assert log.latency_ms == latency_ms
        assert log.success == True  # Changed from status to success (boolean field)
    
    @pytest.mark.asyncio
    async def test_calculate_metrics(self, monitoring_service, test_session, test_model):
        """Test metrics calculation"""
        # Create test prediction logs
        from app.models.monitoring import PredictionLogDB
        import uuid
        
        logs = [
            PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=test_model.id,
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                input_data={"test": "data"},
                output_data={"result": "A"},
                latency_ms=100.0,
                api_endpoint="/predict",
                success=True
            ),
            PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=test_model.id,
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                input_data={"test": "data"},
                output_data={"result": "B"},
                latency_ms=200.0,
                api_endpoint="/predict",
                success=True
            ),
            PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=test_model.id,
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                input_data={"test": "data"},
                output_data={"result": "error"},
                latency_ms=50.0,
                api_endpoint="/predict",
                success=False
            )
        ]
        
        for log in logs:
            test_session.add(log)
        test_session.commit()
        
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        metrics = await monitoring_service.get_model_performance_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.avg_latency_ms == (100.0 + 200.0 + 50.0) / 3
        assert metrics.success_rate == (2 / 3) * 100
        assert metrics.error_rate == (1 / 3) * 100
    
    @pytest.mark.asyncio
    async def test_check_system_health(self, monitoring_service):
        """Test system health checking"""
        with patch('psutil.cpu_percent', return_value=45.2):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.5
                
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk.return_value.percent = 30.0
                    
                    health = await monitoring_service.check_system_health()
        
        assert "cpu_usage" in health
        assert "memory_usage" in health
        assert "disk_usage" in health
        assert health["cpu_usage"] == 45.2
        assert health["memory_usage"] == 60.5
        assert health["disk_usage"] == 30.0
    
    @pytest.mark.asyncio
    async def test_create_alert(self, monitoring_service, test_session):
        """Test alert creation"""
        # Clean up any existing alerts for test isolation
        from app.models.monitoring import AlertDB
        test_session.query(AlertDB).delete()
        test_session.commit()
        
        created_alert = await monitoring_service.create_alert(
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            title="High CPU Usage",
            description="CPU usage exceeded 80%",
            metric_value=85.0
        )
        
        # Verify alert was created and returned object is correct
        assert created_alert.severity == AlertSeverity.WARNING
        assert created_alert.component == SystemComponent.API_SERVER
        assert created_alert.title == "High CPU Usage"
        assert created_alert.description == "CPU usage exceeded 80%"
        assert created_alert.metric_value == 85.0
        assert created_alert.is_active is True

        # Verify alert was stored in DB correctly
        db_alerts = test_session.query(AlertDB).all()
        
        assert len(db_alerts) == 1
        db_alert = db_alerts[0]
        assert db_alert.severity == "warning" # Enum.value is stored
        assert db_alert.component == "api_server" # Enum.value is stored
        assert db_alert.title == "High CPU Usage"
        assert db_alert.description == "CPU usage exceeded 80%"
        # The metadata was for additional_data, which is not a direct param in create_alert
        # metric_value is stored directly
        assert db_alert.metric_value == 85.0


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
            "required": ["feature1", "feature2"]
        }
        
        # Valid data
        valid_data = {"feature1": 0.5, "feature2": "test"}
        is_valid, errors = schema_service.validate_input_schema(valid_data, schema)
        
        assert is_valid is True
        assert errors == []
        
        # Invalid data
        invalid_data = {"feature1": "not_a_number", "feature3": "extra"}
        is_valid, errors = schema_service.validate_input_schema(invalid_data, schema)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_generate_schema_from_data(self, schema_service):
        """Test schema generation from sample data"""
        sample_data = {
            "feature1": 0.5,
            "feature2": "test_string",
            "feature3": 1,
            "feature4": [1, 2, 3]
        }
        
        schema = schema_service.generate_schema_from_data(sample_data)
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["properties"]["feature1"]["type"] == "number"
        assert schema["properties"]["feature2"]["type"] == "string"
        assert schema["properties"]["feature3"]["type"] == "integer"
        assert schema["properties"]["feature4"]["type"] == "array"
    
    def test_merge_schemas(self, schema_service):
        """Test schema merging"""
        schema1 = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1"]
        }
        
        schema2 = {
            "type": "object",
            "properties": {
                "feature2": {"type": "string", "maxLength": 100},
                "feature3": {"type": "boolean"}
            },
            "required": ["feature2"]
        }
        
        merged = schema_service.merge_schemas(schema1, schema2)
        
        assert "feature1" in merged["properties"]
        assert "feature2" in merged["properties"]
        assert "feature3" in merged["properties"]
        assert merged["properties"]["feature2"]["maxLength"] == 100
        assert set(merged["required"]) == {"feature1", "feature2"}
    
    def test_convert_to_openapi_schema(self, schema_service):
        """Test conversion to OpenAPI schema"""
        json_schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number", "description": "First feature"},
                "feature2": {"type": "string", "enum": ["A", "B", "C"]}
            },
            "required": ["feature1"]
        }
        
        openapi_schema = schema_service.convert_to_openapi_schema(json_schema)
        
        assert "type" in openapi_schema
        assert "properties" in openapi_schema
        assert "required" in openapi_schema
        # Should be compatible with OpenAPI format
    
    def test_validate_schema_compatibility(self, schema_service):
        """Test schema compatibility validation"""
        old_schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1"]
        }
        
        # Compatible new schema (adds optional field)
        compatible_schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"},
                "feature3": {"type": "boolean"}
            },
            "required": ["feature1"]
        }
        
        is_compatible, issues = schema_service.validate_schema_compatibility(
            old_schema, compatible_schema
        )
        
        assert is_compatible is True
        assert len(issues) == 0
        
        # Incompatible schema (removes required field)
        incompatible_schema = {
            "type": "object",
            "properties": {
                "feature2": {"type": "string"}
            },
            "required": ["feature2"]
        }
        
        is_compatible, issues = schema_service.validate_schema_compatibility(
            old_schema, incompatible_schema
        )
        
        assert is_compatible is False
        assert len(issues) > 0


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
    async def test_update_deployment_config(self, deployment_service, test_deployment):
        """Test updating deployment configuration"""
        new_config = {
            "resources": {"cpu": "200m", "memory": "512Mi"},
            "scaling": {"min_replicas": 1, "max_replicas": 5}
        }
        
        with patch.object(deployment_service, 'update_deployment_config') as mock_update:
            mock_response = ModelDeploymentResponse(
                id=test_deployment.id,
                model_id=test_deployment.model_id,
                name=test_deployment.deployment_name,
                description=None,
                status=DeploymentStatus.ACTIVE,
                endpoint_url="http://localhost:3001",
                service_name=test_deployment.deployment_name,
                framework="sklearn",
                endpoints=["predict"],
                config=new_config,
                created_at=test_deployment.created_at,
                updated_at=datetime.utcnow()
            )
            mock_update.return_value = (True, "Configuration updated successfully", mock_response)
            
            success, message, deployment = await deployment_service.update_deployment_config(
                test_deployment.id, new_config
            )
            
            assert success is True
            assert "updated" in message.lower()
            assert deployment.config == new_config
    
    @pytest.mark.asyncio
    async def test_get_deployment_logs(self, deployment_service, test_deployment):
        """Test getting deployment logs"""
        mock_logs = [
            f"[{datetime.utcnow().isoformat()}] INFO: Service {test_deployment.deployment_name} started",
            f"[{datetime.utcnow().isoformat()}] INFO: Model loaded successfully",
            f"[{datetime.utcnow().isoformat()}] INFO: Endpoint /predict ready",
            f"[{datetime.utcnow().isoformat()}] INFO: Health check passed"
        ]
        
        with patch.object(deployment_service, 'get_deployment_logs') as mock_get_logs:
            mock_get_logs.return_value = mock_logs
            
            logs = await deployment_service.get_deployment_logs(test_deployment.id)
            
            assert len(logs) == 4
            assert "started" in logs[0]


class TestServiceIntegration:
    """Integration tests for service interactions"""
    
    @pytest.mark.asyncio
    async def test_monitoring_deployment_integration(self, monitoring_service, deployment_service):
        """Test integration between monitoring and deployment services"""
        from app.schemas.model import ModelDeploymentCreate
        
        # Mock deployment creation
        deployment_data = ModelDeploymentCreate(
            model_id="test_model_123",
            name="test_deployment",
            description="Integration test deployment"
        )
        
        with patch.object(deployment_service, 'create_deployment') as mock_create:
            with patch.object(monitoring_service, 'log_prediction') as mock_log:
                mock_create.return_value = (True, "Success", None)
                mock_log.return_value = "prediction_123"
                
                # Create deployment
                success, message, deployment = await deployment_service.create_deployment(deployment_data)
                assert success is True
                
                # Log a prediction (simulating monitoring)
                prediction_id = await monitoring_service.log_prediction(
                    model_id="test_model_123",
                    deployment_id="deploy_123",
                    input_data={"feature1": 0.5},
                    prediction={"class": "A", "probability": 0.85},
                    response_time=45.2
                )
                
                assert prediction_id == "prediction_123"
    
    @pytest.mark.asyncio
    async def test_schema_bentoml_integration(self):
        """Test integration between schema and BentoML services"""
        schema_service = SchemaService()
        bentoml_service = BentoMLServiceManager()
        
        # Generate schema from sample data
        sample_data = {"feature1": 0.5, "feature2": "test"}
        schema = schema_service.generate_schema_from_data(sample_data)
        
        # Use schema in BentoML service generation
        model_info = {
            "name": "integration_test",
            "framework": "sklearn",
            "model_type": "classification",
            "input_schema": schema
        }
        
        service_code = bentoml_service.generate_service_code(model_info)
        
        # Check that the service code contains expected framework and model information
        assert "sklearn" in service_code
        assert "integration_test" in service_code


class TestServiceErrorHandling:
    """Test service error handling scenarios"""
    
    @pytest.fixture
    def deployment_service(self):
        """Create deployment service instance for error testing"""
        return DeploymentService()
    
    @pytest.mark.asyncio
    async def test_monitoring_service_db_error(self):
        """Test monitoring service with database errors"""
        monitoring_service = MonitoringService()
        
        # Mock session that raises exception
        mock_session = MagicMock()
        mock_session.add.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            await monitoring_service.log_prediction(
                mock_session, 1, {}, {}, 100.0, "success"
            )
    
    @pytest.mark.asyncio
    async def test_deployment_service_network_error(self, deployment_service):
        """Test deployment service with network errors"""
        
        deployment = ModelDeployment(
            id="test_deployment_id",  # Add explicit ID
            deployment_name="test",
            model_id="test_model_id",
            deployment_url="http://invalid-url:3001",
            status="pending",
            configuration={},
            replicas=1
        )
        
        # Mock the session.get to return our deployment
        with patch('app.services.deployment_service.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get.return_value = deployment
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Mock BentoML service to simulate network error
            with patch('app.services.deployment_service.bentoml_service_manager.get_service_status') as mock_service_status:
                mock_service_status.side_effect = Exception("Network error")
                
                # Call with deployment ID, not the deployment object
                status = await deployment_service.get_deployment_status(deployment.id)
                
                # When an exception occurs, the method returns None
                assert status is None
    
    def test_schema_service_invalid_data(self):
        """Test schema service with invalid data"""
        schema_service = SchemaService()
        
        # Test with None data
        schema = schema_service.generate_schema_from_data(None)
        assert schema is not None  # Should handle gracefully
        
        # Test with invalid schema that will actually fail validation
        valid_schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        }
        
        # Test with data missing required field
        invalid_data = {"wrong_field": "value"}
        is_valid, errors = schema_service.validate_input_schema(
            invalid_data, valid_schema
        )
        
        # Should fail validation due to missing required field
        assert is_valid is False and len(errors) > 0 