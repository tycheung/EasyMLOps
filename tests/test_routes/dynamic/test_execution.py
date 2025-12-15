"""
Tests for dynamic route execution, prediction, and inference
Tests prediction endpoints, batch processing, schema validation, and logging
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import asyncio

from app.core.app_factory import create_app
from app.models.model import ModelDeployment, ModelPrediction
from app.schemas.model import DeploymentStatus
from app.routes.dynamic import route_manager


@pytest.fixture
def sample_deployment(test_model):
    """Sample deployment for testing"""
    return ModelDeployment(
        id="deploy_123",
        model_id=test_model.id,
        deployment_name="test_deployment",
        deployment_url="http://localhost:3000/model_service_123",
        status="active",
        configuration={},
        framework="sklearn",
        endpoints=["predict", "predict_proba"],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_prediction_data():
    """Sample prediction input data"""
    return {
        "data": [1.0, 2.5, -0.5, 3.2]
    }


@pytest.fixture
def sample_schema_data():
    """Sample schema-formatted input data"""
    return {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft": 2000,
        "location": "suburban"
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch prediction data"""
    return {
        "data": [
            {"bedrooms": 3, "bathrooms": 2.5, "sqft": 2000},
            {"bedrooms": 4, "bathrooms": 3.0, "sqft": 2500},
            {"bedrooms": 2, "bathrooms": 1.5, "sqft": 1200}
        ]
    }


class TestPredictEndpoint:
    """Test main prediction endpoint"""
    
    def test_predict_success_no_schema(self, client, test_deployment, sample_prediction_data):
        """Test successful prediction without schema validation"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas:
            mock_get_schemas.return_value = (None, None)
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}",
                json=sample_prediction_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
            assert "validation" in result
            assert result["validation"]["validation_performed"] is False
            assert result["model_id"] == test_deployment.model_id
    
    def test_predict_success_with_schema(self, client, test_deployment, 
                                       sample_schema_data, test_model):
        """Test successful prediction with schema validation"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.validate_prediction_data') as mock_validate_data:
            
            mock_schema = MagicMock()
            mock_get_schemas.return_value = (mock_schema, None)
            mock_validate_data.return_value = (True, "Validation successful", sample_schema_data)
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}",
                json=sample_schema_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
            assert "validation" in result
            assert result["validation"]["validation_performed"] is True
            assert result["validation"]["schema_applied"] is True
            mock_validate_data.assert_called_once()
    
    def test_predict_schema_validation_failure(self, client, test_deployment,
                                             sample_schema_data):
        """Test prediction with schema validation failure"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.validate_prediction_data') as mock_validate_data:
            
            mock_schema = MagicMock()
            mock_get_schemas.return_value = (mock_schema, None)
            mock_validate_data.return_value = (False, "Required field 'bedrooms' missing", {})
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}",
                json={"invalid": "data"}
            )
            
            assert response.status_code == 422
            result = response.json()
            assert "Input validation failed" in result["error"]["message"]
    
    def test_predict_deployment_not_found(self, client):
        """Test prediction with non-existent deployment"""
        response = client.post(
            "/api/v1/predict/nonexistent",
            json={"data": [1, 2, 3]}
        )
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    def test_predict_deployment_inactive(self, client, test_deployment):
        """Test prediction with inactive deployment"""
        test_deployment.status = "stopped"
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.get') as mock_get:
            mock_get.return_value = test_deployment
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}",
                json={"data": [1, 2, 3]}
            )
            
            assert response.status_code == 400
            result = response.json()
            assert "not active" in result["error"]["message"].lower()
    
    def test_predict_invalid_json(self, client, test_deployment):
        """Test prediction with invalid JSON data"""
        response = client.post(
            f"/api/v1/predict/{test_deployment.id}",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "Invalid JSON" in result["error"]["message"]


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint"""
    
    def test_batch_predict_success_no_schema(self, client, test_deployment, sample_batch_data):
        """Test successful batch prediction without schema validation"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas:
            mock_get_schemas.return_value = (None, None)
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}/batch",
                json=sample_batch_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
            assert "validation" in result
            assert result["validation"]["validation_performed"] is False
            assert result["batch_size"] == 3
    
    def test_batch_predict_success_with_schema(self, client, test_deployment, sample_batch_data):
        """Test successful batch prediction with schema validation"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.validate_prediction_data') as mock_validate_data:
            
            mock_schema = MagicMock()
            mock_get_schemas.return_value = (mock_schema, None)
            mock_validate_data.side_effect = [
                (True, "Valid", sample_batch_data["data"][0]),
                (True, "Valid", sample_batch_data["data"][1]),
                (True, "Valid", sample_batch_data["data"][2])
            ]
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}/batch",
                json=sample_batch_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
            assert "validation" in result
            assert result["validation"]["validation_performed"] is True
            assert len(result["validation"]["item_validations"]) == 3
            assert mock_validate_data.call_count == 3
    
    def test_batch_predict_validation_failure(self, client, test_deployment, sample_batch_data):
        """Test batch prediction with validation failure on one item"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.validate_prediction_data') as mock_validate_data:
            
            mock_schema = MagicMock()
            mock_get_schemas.return_value = (mock_schema, None)
            mock_validate_data.side_effect = [
                (True, "Valid", sample_batch_data["data"][0]),
                (False, "Invalid field", {}),
                (True, "Valid", sample_batch_data["data"][2])
            ]
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}/batch",
                json=sample_batch_data
            )
            
            assert response.status_code == 422
            result = response.json()
            assert "Validation failed for batch item 1" in result["error"]["message"]
    
    def test_batch_predict_invalid_format(self, client, test_deployment):
        """Test batch prediction with invalid data format"""
        invalid_data = {"not_data_key": [1, 2, 3]}
        
        response = client.post(
            f"/api/v1/predict/{test_deployment.id}/batch",
            json=invalid_data
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "must be provided as a list" in result["error"]["message"]


class TestPredictProbaEndpoint:
    """Test probability prediction endpoint"""
    
    def test_predict_proba_success(self, client, test_deployment, sample_prediction_data):
        """Test successful probability prediction"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas:
            mock_get_schemas.return_value = (None, None)
            
            response = client.post(
                f"/api/v1/predict/{test_deployment.id}/proba",
                json=sample_prediction_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "probabilities" in result
            assert "validation" in result
            assert result["model_id"] == test_deployment.model_id
    
    def test_predict_proba_not_supported(self, client, test_deployment, test_session):
        """Test probability prediction when not supported by model"""
        test_deployment.endpoints = ["predict"]
        test_session.add(test_deployment)
        test_session.commit()
        
        response = client.post(
            f"/api/v1/predict/{test_deployment.id}/proba",
            json={"data": [1, 2, 3]}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "not supported" in result["error"]["message"].lower()


class TestPredictionSchemaEndpoint:
    """Test prediction schema endpoint"""
    
    def test_get_prediction_schema_success(self, client, test_deployment):
        """Test successful schema retrieval"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.get_model_example_data') as mock_get_example:
            
            mock_input_schema = MagicMock()
            mock_input_schema.dict.return_value = {"type": "object", "properties": {}}
            mock_output_schema = MagicMock()
            mock_output_schema.dict.return_value = {"type": "array", "items": {"type": "number"}}
            mock_get_schemas.return_value = (mock_input_schema, mock_output_schema)
            mock_get_example.return_value = {"bedrooms": 3, "bathrooms": 2}
            
            response = client.get(f"/api/v1/predict/{test_deployment.id}/schema")
            
            assert response.status_code == 200
            result = response.json()
            assert result["deployment_id"] == test_deployment.id
            assert result["model_id"] == test_deployment.model_id
            assert result["framework"] == test_deployment.framework
            assert "input_schema" in result
            assert "output_schema" in result
            assert "example_input" in result
            assert result["validation_enabled"] is True
    
    def test_get_prediction_schema_deployment_not_found(self, client):
        """Test schema retrieval for non-existent deployment"""
        response = client.get("/api/v1/predict/nonexistent/schema")
        
        assert response.status_code == 404


class TestPredictionHelpers:
    """Test prediction helper functions"""
    
    def test_simulate_sklearn_prediction(self, sample_deployment):
        """Test sklearn prediction simulation"""
        from app.routes.dynamic.simulation_helpers import simulate_sklearn_prediction
        
        request_data = {"data": [1, 2, 3, 4]}
        result = simulate_sklearn_prediction(sample_deployment, request_data)
        
        assert isinstance(result, dict)
        assert "predictions" in result
        assert "model_id" in result
        assert "framework" in result
        assert result["framework"] == "sklearn"
    
    def test_simulate_tensorflow_prediction(self, sample_deployment):
        """Test TensorFlow prediction simulation"""
        from app.routes.dynamic.simulation_helpers import simulate_tensorflow_prediction
        
        sample_deployment.framework = "tensorflow"
        request_data = {"data": [[1, 2, 3, 4]]}
        result = simulate_tensorflow_prediction(sample_deployment, request_data)
        
        assert isinstance(result, dict)
        assert "predictions" in result
        assert result["framework"] == "tensorflow"
        assert isinstance(result["predictions"], list)
    
    def test_simulate_pytorch_prediction(self, sample_deployment):
        """Test PyTorch prediction simulation"""
        from app.routes.dynamic.simulation_helpers import simulate_pytorch_prediction
        
        sample_deployment.framework = "pytorch"
        request_data = {"data": [1, 2, 3, 4]}
        result = simulate_pytorch_prediction(sample_deployment, request_data)
        
        assert isinstance(result, dict)
        assert "predictions" in result
        assert result["framework"] == "pytorch"
    
    def test_simulate_boosting_prediction(self, sample_deployment):
        """Test XGBoost/LightGBM prediction simulation"""
        from app.routes.dynamic.simulation_helpers import simulate_boosting_prediction
        
        sample_deployment.framework = "xgboost"
        request_data = {"data": [1, 2, 3, 4]}
        result = simulate_boosting_prediction(sample_deployment, request_data)
        
        assert isinstance(result, dict)
        assert "predictions" in result
        assert result["framework"] == "xgboost"
    
    def test_simulate_generic_prediction(self, sample_deployment):
        """Test generic prediction simulation"""
        from app.routes.dynamic.simulation_helpers import simulate_generic_prediction
        
        sample_deployment.framework = "unknown"
        request_data = {"data": [1, 2, 3, 4]}
        result = simulate_generic_prediction(sample_deployment, request_data)
        
        assert isinstance(result, dict)
        assert "predictions" in result
        assert result["framework"] == "unknown"


class TestPredictionLogging:
    """Test prediction logging functionality"""
    
    @patch('app.database.get_async_session')
    def test_log_prediction_success(self, mock_get_session):
        """Test successful prediction logging"""
        from app.routes.dynamic.logging_helpers import log_prediction
        from app.models.model import ModelDeployment
        
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        mock_deployment = MagicMock(spec=ModelDeployment)
        mock_deployment.model_id = "test_model_123"
        mock_session.get = AsyncMock(return_value=mock_deployment)
        
        request_data = {"data": [1, 2, 3]}
        response_data = {"predictions": [0.75]}
        
        asyncio.run(log_prediction(
            mock_session, "deploy_123", request_data, response_data
        ))
        
        mock_session.add.assert_called_once()
    
    @patch('app.database.get_async_session')
    def test_log_prediction_batch(self, mock_get_session):
        """Test batch prediction logging"""
        from app.routes.dynamic.logging_helpers import log_prediction
        from app.models.model import ModelDeployment
        
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        mock_deployment = MagicMock(spec=ModelDeployment)
        mock_deployment.model_id = "test_model_123"
        mock_session.get = AsyncMock(return_value=mock_deployment)
        
        request_data = {"data": [[1, 2], [3, 4]]}
        response_data = {"predictions": [0.75, 0.85]}
        
        asyncio.run(log_prediction(
            mock_session, "deploy_123", request_data, response_data, is_batch=True
        ))
        
        mock_session.add.assert_called_once()
    
    @patch('app.database.get_async_session')
    def test_log_prediction_error_handling(self, mock_get_session):
        """Test prediction logging error handling"""
        from app.routes.dynamic.logging_helpers import log_prediction
        
        mock_session = MagicMock()
        mock_session.add.side_effect = Exception("Database error")
        
        asyncio.run(log_prediction(
            mock_session, "deploy_123", {}, {}
        ))


class TestDynamicRoutesIntegration:
    """Integration tests for dynamic routes"""
    
    def test_complete_prediction_workflow(self, client, test_deployment, sample_schema_data):
        """Test complete prediction workflow with schema validation"""
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.validate_prediction_data') as mock_validate_data, \
             patch('app.services.schema_service.schema_service.get_model_example_data') as mock_get_example:
            
            mock_schema = MagicMock()
            mock_get_schemas.return_value = (mock_schema, None)
            mock_validate_data.return_value = (True, "Validation successful", sample_schema_data)
            mock_get_example.return_value = {"bedrooms": 3, "bathrooms": 2}
            
            predict_response = client.post(
                f"/api/v1/predict/{test_deployment.id}",
                json=sample_schema_data
            )
            assert predict_response.status_code == 200
            
            batch_data = {"data": [sample_schema_data, sample_schema_data]}
            batch_response = client.post(
                f"/api/v1/predict/{test_deployment.id}/batch",
                json=batch_data
            )
            assert batch_response.status_code == 200
            
            proba_response = client.post(
                f"/api/v1/predict/{test_deployment.id}/proba",
                json=sample_schema_data
            )
            assert proba_response.status_code == 200
            
            schema_response = client.get(f"/api/v1/predict/{test_deployment.id}/schema")
            assert schema_response.status_code == 200
            
            assert mock_validate_data.call_count >= 3


class TestErrorHandling:
    """Test error handling in model routes (execution-related)"""
    
    def test_invalid_json_request(self, client, test_model):
        """Test handling of invalid JSON in requests"""
        response = client.post(
            f"/api/v1/models/{test_model.id}/validate",
            data="invalid json data",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestIntegrationScenarios:
    """Integration test scenarios (execution-related)"""
    
    def test_concurrent_model_operations(self, client, test_model):
        """Test concurrent operations on the same model"""
        model_id = test_model.id
        
        responses = []
        for _ in range(5):
            response = client.get(f"/api/v1/models/{model_id}")
            responses.append(response)
        
        for response in responses:
            assert response.status_code == 200
            result = response.json()
            assert result["model"]["id"] == model_id

