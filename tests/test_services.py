"""
Unit tests for service modules
Tests business logic in monitoring, BentoML, schema, and deployment services
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from app.services.monitoring_service import MonitoringService
from app.services.bentoml_service import BentoMLService
from app.services.schema_service import SchemaService
from app.services.deployment_service import DeploymentService


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
        prediction = {"class": "A", "probability": 0.85}
        response_time = 150.5
        
        await monitoring_service.log_prediction(
            test_session, test_model.id, input_data, prediction, response_time, "success"
        )
        
        # Verify log was created
        from app.models.monitoring import PredictionLog
        logs = test_session.query(PredictionLog).filter(
            PredictionLog.model_id == test_model.id
        ).all()
        
        assert len(logs) == 1
        log = logs[0]
        assert log.input_data == input_data
        assert log.prediction == prediction
        assert log.response_time_ms == response_time
        assert log.status == "success"
    
    @pytest.mark.asyncio
    async def test_calculate_metrics(self, monitoring_service, test_session, test_model):
        """Test metrics calculation"""
        # Create test prediction logs
        from app.models.monitoring import PredictionLog
        logs = [
            PredictionLog(
                model_id=test_model.id,
                input_data={"test": "data"},
                prediction={"result": "A"},
                response_time_ms=100.0,
                status="success"
            ),
            PredictionLog(
                model_id=test_model.id,
                input_data={"test": "data"},
                prediction={"result": "B"},
                response_time_ms=200.0,
                status="success"
            ),
            PredictionLog(
                model_id=test_model.id,
                input_data={"test": "data"},
                prediction={"result": "error"},
                response_time_ms=50.0,
                status="error"
            )
        ]
        
        for log in logs:
            test_session.add(log)
        test_session.commit()
        
        metrics = await monitoring_service.calculate_metrics(test_session, test_model.id)
        
        assert "total_predictions" in metrics
        assert "success_rate" in metrics
        assert "average_response_time" in metrics
        assert metrics["total_predictions"] == 3
        assert metrics["success_rate"] == 2/3  # 2 out of 3 successful
        assert metrics["average_response_time"] == 150.0  # (100+200+50)/3
    
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
        await monitoring_service.create_alert(
            test_session,
            alert_type="performance",
            severity="warning",
            title="High CPU Usage",
            message="CPU usage exceeded threshold",
            source_component="api_server",
            metadata={"cpu_usage": 85.0}
        )
        
        # Verify alert was created
        from app.models.monitoring import Alert
        alerts = test_session.query(Alert).all()
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.alert_type == "performance"
        assert alert.severity == "warning"
        assert alert.title == "High CPU Usage"
        assert alert.metadata["cpu_usage"] == 85.0


class TestBentoMLService:
    """Test BentoMLService functionality"""
    
    @pytest.fixture
    def bentoml_service(self):
        """Create BentoML service instance"""
        return BentoMLService()
    
    @patch('app.services.bentoml_service.bentoml')
    def test_create_service(self, mock_bentoml, bentoml_service, temp_model_file):
        """Test BentoML service creation"""
        mock_service = MagicMock()
        mock_bentoml.Service.return_value = mock_service
        
        service_name = "test_service"
        model_path = temp_model_file
        
        result = bentoml_service.create_service(service_name, model_path)
        
        assert result is not None
        mock_bentoml.Service.assert_called_once_with(service_name)
    
    @patch('app.services.bentoml_service.bentoml')
    def test_build_bento(self, mock_bentoml, bentoml_service):
        """Test Bento building"""
        mock_bento = MagicMock()
        mock_bento.tag = "test_service:latest"
        mock_bentoml.build.return_value = mock_bento
        
        service_name = "test_service"
        version = "1.0.0"
        
        result = bentoml_service.build_bento(service_name, version)
        
        assert result is not None
        assert result.tag == "test_service:latest"
    
    @patch('app.services.bentoml_service.bentoml')
    def test_serve_bento(self, mock_bentoml, bentoml_service):
        """Test Bento serving"""
        mock_server = MagicMock()
        mock_bentoml.serve.return_value = mock_server
        
        bento_tag = "test_service:latest"
        port = 3000
        
        result = bentoml_service.serve_bento(bento_tag, port)
        
        assert result is not None
        mock_bentoml.serve.assert_called_once()
    
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
        is_valid, errors = schema_service.validate_input_schema(schema, valid_data)
        
        assert is_valid is True
        assert errors == []
        
        # Invalid data
        invalid_data = {"feature1": "not_a_number", "feature3": "extra"}
        is_valid, errors = schema_service.validate_input_schema(schema, invalid_data)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_generate_schema_from_data(self, schema_service):
        """Test schema generation from sample data"""
        sample_data = {
            "feature1": 0.5,
            "feature2": "test_string",
            "feature3": True,
            "feature4": [1, 2, 3]
        }
        
        schema = schema_service.generate_schema_from_data(sample_data)
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["properties"]["feature1"]["type"] == "number"
        assert schema["properties"]["feature2"]["type"] == "string"
        assert schema["properties"]["feature3"]["type"] == "boolean"
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
        deployment_config = {
            "name": "test_deployment",
            "model_id": test_model.id,
            "environment": "test",
            "resources": {"cpu": "100m", "memory": "256Mi"},
            "scaling": {"min_replicas": 1, "max_replicas": 3}
        }
        
        deployment = await deployment_service.create_deployment(
            test_session, deployment_config
        )
        
        assert deployment is not None
        assert deployment.name == "test_deployment"
        assert deployment.model_id == test_model.id
        assert deployment.status == "pending"
    
    @pytest.mark.asyncio
    async def test_deploy_model(self, deployment_service, test_deployment):
        """Test model deployment"""
        with patch.object(deployment_service, '_start_bento_server') as mock_start:
            mock_start.return_value = "http://localhost:3001"
            
            result = await deployment_service.deploy_model(test_deployment)
            
            assert result is True
            mock_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_deployment(self, deployment_service, test_deployment):
        """Test deployment stopping"""
        test_deployment.status = "running"
        test_deployment.endpoint_url = "http://localhost:3001"
        
        with patch.object(deployment_service, '_stop_bento_server') as mock_stop:
            mock_stop.return_value = True
            
            result = await deployment_service.stop_deployment(test_deployment)
            
            assert result is True
            assert test_deployment.status == "stopped"
    
    @pytest.mark.asyncio
    async def test_scale_deployment(self, deployment_service, test_deployment):
        """Test deployment scaling"""
        test_deployment.status = "running"
        
        new_scaling = {"min_replicas": 2, "max_replicas": 5}
        
        with patch.object(deployment_service, '_update_scaling') as mock_scale:
            mock_scale.return_value = True
            
            result = await deployment_service.scale_deployment(
                test_deployment, new_scaling
            )
            
            assert result is True
            assert test_deployment.scaling == new_scaling
    
    @pytest.mark.asyncio
    async def test_get_deployment_status(self, deployment_service, test_deployment):
        """Test getting deployment status"""
        test_deployment.endpoint_url = "http://localhost:3001"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            status = await deployment_service.get_deployment_status(test_deployment)
            
            assert status["status"] == "healthy"
            assert status["endpoint_accessible"] is True
    
    @pytest.mark.asyncio
    async def test_update_deployment_config(self, deployment_service, test_deployment):
        """Test updating deployment configuration"""
        new_config = {
            "resources": {"cpu": "200m", "memory": "512Mi"},
            "scaling": {"min_replicas": 1, "max_replicas": 5}
        }
        
        result = await deployment_service.update_deployment_config(
            test_deployment, new_config
        )
        
        assert result is True
        assert test_deployment.resources == new_config["resources"]
        assert test_deployment.scaling == new_config["scaling"]
    
    @pytest.mark.asyncio
    async def test_get_deployment_logs(self, deployment_service, test_deployment):
        """Test getting deployment logs"""
        test_deployment.endpoint_url = "http://localhost:3001"
        
        with patch.object(deployment_service, '_fetch_container_logs') as mock_logs:
            mock_logs.return_value = ["Log line 1", "Log line 2", "Log line 3"]
            
            logs = await deployment_service.get_deployment_logs(test_deployment)
            
            assert len(logs) == 3
            assert "Log line 1" in logs


class TestServiceIntegration:
    """Integration tests for service interactions"""
    
    @pytest.mark.asyncio
    async def test_monitoring_deployment_integration(self, test_session, test_model):
        """Test integration between monitoring and deployment services"""
        monitoring_service = MonitoringService()
        deployment_service = DeploymentService()
        
        # Create deployment
        deployment_config = {
            "name": "integration_test",
            "model_id": test_model.id,
            "environment": "test",
            "resources": {"cpu": "100m", "memory": "256Mi"},
            "scaling": {"min_replicas": 1, "max_replicas": 1}
        }
        
        deployment = await deployment_service.create_deployment(
            test_session, deployment_config
        )
        
        # Log prediction for monitoring
        await monitoring_service.log_prediction(
            test_session,
            test_model.id,
            {"feature1": 0.5},
            {"prediction": "A"},
            150.0,
            "success"
        )
        
        # Get metrics
        metrics = await monitoring_service.calculate_metrics(
            test_session, test_model.id
        )
        
        assert metrics["total_predictions"] == 1
        assert deployment.model_id == test_model.id
    
    @pytest.mark.asyncio
    async def test_schema_bentoml_integration(self):
        """Test integration between schema and BentoML services"""
        schema_service = SchemaService()
        bentoml_service = BentoMLService()
        
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
        
        assert "feature1" in service_code or "feature2" in service_code
        assert "sklearn" in service_code


class TestServiceErrorHandling:
    """Test error handling in services"""
    
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
        from app.models.model import Deployment
        
        deployment = Deployment(
            name="test",
            model_id=1,
            endpoint_url="http://invalid-url:3001",
            environment="test",
            resources={},
            scaling={}
        )
        
        # Should handle network errors gracefully
        status = await deployment_service.get_deployment_status(deployment)
        
        assert "error" in status or status["endpoint_accessible"] is False
    
    def test_schema_service_invalid_data(self):
        """Test schema service with invalid data"""
        schema_service = SchemaService()
        
        # Test with None data
        schema = schema_service.generate_schema_from_data(None)
        assert schema is not None  # Should handle gracefully
        
        # Test with invalid schema
        invalid_schema = {"invalid": "schema"}
        is_valid, errors = schema_service.validate_input_schema(
            invalid_schema, {"test": "data"}
        )
        
        # Should handle invalid schema gracefully
        assert is_valid is False or len(errors) > 0 