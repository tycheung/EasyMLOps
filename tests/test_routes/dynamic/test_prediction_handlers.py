"""
Tests for prediction handlers
Tests predict, predict_batch, and predict_proba endpoints with comprehensive coverage
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException, status
from datetime import datetime

from app.models.model import ModelDeployment
from app.schemas.model import DeploymentStatus


@pytest.fixture
def active_deployment(test_model, test_session):
    """Create an active deployment for testing"""
    deployment = ModelDeployment(
        id="deploy_pred_123",
        model_id=test_model.id,
        name="test_deployment",
        deployment_name="test_deployment",
        deployment_url="http://localhost:3000/service",
        status=DeploymentStatus.ACTIVE.value,
        framework="sklearn",
        endpoints=["predict", "predict_proba"],
        configuration={},
        created_at=datetime.utcnow()
    )
    test_session.add(deployment)
    test_session.commit()
    test_session.refresh(deployment)
    return deployment


@pytest.fixture
def inactive_deployment(test_model, test_session):
    """Create an inactive deployment for testing"""
    deployment = ModelDeployment(
        id="deploy_inactive_123",
        model_id=test_model.id,
        name="inactive_deployment",
        deployment_name="inactive_deployment",
        deployment_url="http://localhost:3000/service",
        status=DeploymentStatus.STOPPED.value,
        framework="sklearn",
        endpoints=["predict"],
        configuration={},
        created_at=datetime.utcnow()
    )
    test_session.add(deployment)
    test_session.commit()
    test_session.refresh(deployment)
    return deployment


class TestPredictEndpoint:
    """Test predict endpoint comprehensively"""
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_prediction_call', new_callable=AsyncMock)
    @patch('app.routes.dynamic.prediction_handlers.log_prediction', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    async def test_predict_no_schema(self, mock_get_schemas, mock_log, mock_predict_call, 
                                     client, active_deployment, test_session):
        """Test predict endpoint without schema"""
        mock_get_schemas.return_value = (None, None)
        mock_predict_call.return_value = {
            "predictions": [0.75],
            "model_id": active_deployment.model_id,
            "framework": "sklearn"
        }
        mock_log.return_value = None
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}",
            json={"data": [1.0, 2.0, 3.0]}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert result["validation"]["validation_performed"] is False
        mock_predict_call.assert_called_once()
        mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_prediction_call', new_callable=AsyncMock)
    @patch('app.routes.dynamic.prediction_handlers.log_prediction', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.validate_prediction_data', new_callable=AsyncMock)
    async def test_predict_with_schema_direct_format(self, mock_validate, mock_get_schemas, 
                                                     mock_log, mock_predict_call,
                                                     client, active_deployment, test_session):
        """Test predict endpoint with schema validation - direct format"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_validate.return_value = (True, "Valid", {"bedrooms": 3, "bathrooms": 2})
        mock_predict_call.return_value = {
            "predictions": [250000.0],
            "model_id": active_deployment.model_id
        }
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}",
            json={"bedrooms": 3, "bathrooms": 2}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["validation"]["validation_performed"] is True
        assert result["validation"]["schema_applied"] is True
        mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_prediction_call', new_callable=AsyncMock)
    @patch('app.routes.dynamic.prediction_handlers.log_prediction', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.validate_prediction_data', new_callable=AsyncMock)
    async def test_predict_with_schema_traditional_format(self, mock_validate, mock_get_schemas,
                                                         mock_log, mock_predict_call,
                                                         client, active_deployment, test_session):
        """Test predict endpoint with schema validation - traditional format"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_validate.return_value = (True, "Valid", {"bedrooms": 3, "bathrooms": 2})
        mock_predict_call.return_value = {"predictions": [250000.0]}
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}",
            json={"data": {"bedrooms": 3, "bathrooms": 2}}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["validation"]["validation_performed"] is True
        mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.validate_prediction_data', new_callable=AsyncMock)
    async def test_predict_schema_validation_failure(self, mock_validate, mock_get_schemas,
                                                     client, active_deployment, test_session):
        """Test predict endpoint with schema validation failure"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_validate.return_value = (False, "Missing required field: bedrooms", {})
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}",
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422
        result = response.json()
        assert "Input validation failed" in result["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_predict_deployment_not_found(self, client, test_session):
        """Test predict endpoint with non-existent deployment"""
        response = client.post(
            "/api/v1/predict/nonexistent",
            json={"data": [1, 2, 3]}
        )
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_predict_deployment_inactive(self, client, inactive_deployment, test_session):
        """Test predict endpoint with inactive deployment"""
        response = client.post(
            f"/api/v1/predict/{inactive_deployment.id}",
            json={"data": [1, 2, 3]}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "not active" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_predict_invalid_json(self, client, active_deployment, test_session):
        """Test predict endpoint with invalid JSON"""
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "Invalid JSON" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_prediction_call', new_callable=AsyncMock)
    @patch('app.routes.dynamic.prediction_handlers.log_prediction', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    async def test_predict_array_input_no_validation(self, mock_get_schemas, mock_log, mock_predict_call,
                                                     client, active_deployment, test_session):
        """Test predict endpoint with array input (no schema validation)"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_predict_call.return_value = {"predictions": [0.75]}
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}",
            json={"data": [1.0, 2.0, 3.0]}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["validation"]["validation_performed"] is False
        assert "Array input cannot be validated" in result["validation"]["validation_message"]


class TestPredictBatchEndpoint:
    """Test predict_batch endpoint comprehensively"""
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_batch_prediction_call', new_callable=AsyncMock)
    @patch('app.routes.dynamic.prediction_handlers.log_prediction', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    async def test_predict_batch_no_schema(self, mock_get_schemas, mock_log, mock_batch_call,
                                          client, active_deployment, test_session):
        """Test batch predict endpoint without schema"""
        mock_get_schemas.return_value = (None, None)
        mock_batch_call.return_value = {
            "predictions": [[0.75], [0.85]],
            "batch_size": 2
        }
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/batch",
            json={"data": [[1, 2], [3, 4]]}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert result["batch_size"] == 2
        assert result["validation"]["validation_performed"] is False
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_batch_prediction_call', new_callable=AsyncMock)
    @patch('app.routes.dynamic.prediction_handlers.log_prediction', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.validate_prediction_data', new_callable=AsyncMock)
    async def test_predict_batch_with_schema(self, mock_validate, mock_get_schemas, mock_log, mock_batch_call,
                                            client, active_deployment, test_session):
        """Test batch predict endpoint with schema validation"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_validate.side_effect = [
            (True, "Valid", {"bedrooms": 3, "bathrooms": 2}),
            (True, "Valid", {"bedrooms": 4, "bathrooms": 3})
        ]
        mock_batch_call.return_value = {
            "predictions": [[250000], [350000]],
            "batch_size": 2
        }
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/batch",
            json={"data": [
                {"bedrooms": 3, "bathrooms": 2},
                {"bedrooms": 4, "bathrooms": 3}
            ]}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["validation"]["validation_performed"] is True
        assert len(result["validation"]["item_validations"]) == 2
        assert mock_validate.call_count == 2
    
    @pytest.mark.asyncio
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.validate_prediction_data', new_callable=AsyncMock)
    async def test_predict_batch_validation_failure(self, mock_validate, mock_get_schemas,
                                                    client, active_deployment, test_session):
        """Test batch predict endpoint with validation failure"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_validate.side_effect = [
            (True, "Valid", {"bedrooms": 3}),
            (False, "Missing bedrooms", {})
        ]
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/batch",
            json={"data": [
                {"bedrooms": 3, "bathrooms": 2},
                {"bathrooms": 3}  # Missing bedrooms
            ]}
        )
        
        assert response.status_code == 422
        result = response.json()
        assert "Validation failed for batch item 1" in result["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_predict_batch_invalid_format(self, client, active_deployment, test_session):
        """Test batch predict endpoint with invalid format"""
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/batch",
            json={"not_data": [[1, 2]]}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "must be provided as a list" in result["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_predict_batch_empty_list(self, client, active_deployment, test_session):
        """Test batch predict endpoint with empty list"""
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/batch",
            json={"data": []}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "cannot be empty" in result["error"]["message"].lower()


class TestPredictProbaEndpoint:
    """Test predict_proba endpoint comprehensively"""
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_proba_prediction_call', new_callable=AsyncMock)
    @patch('app.routes.dynamic.prediction_handlers.log_prediction', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    async def test_predict_proba_success(self, mock_get_schemas, mock_log, mock_proba_call,
                                        client, active_deployment, test_session):
        """Test successful probability prediction"""
        mock_get_schemas.return_value = (None, None)
        mock_proba_call.return_value = {
            "probabilities": [[0.7, 0.3], [0.6, 0.4]],
            "classes": ["class_0", "class_1"]
        }
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/proba",
            json={"data": [1.0, 2.0]}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "probabilities" in result
        assert "validation" in result
    
    @pytest.mark.asyncio
    async def test_predict_proba_not_supported(self, client, test_model, test_session):
        """Test probability prediction when not supported"""
        deployment = ModelDeployment(
            id="deploy_no_proba",
            model_id=test_model.id,
            name="no_proba_deployment",
            deployment_name="no_proba_deployment",
            deployment_url="http://localhost:3000/service",
            status=DeploymentStatus.ACTIVE.value,
            framework="sklearn",
            endpoints=["predict"],  # No predict_proba
            configuration={},
            created_at=datetime.utcnow()
        )
        test_session.add(deployment)
        test_session.commit()
        
        response = client.post(
            f"/api/v1/predict/{deployment.id}/proba",
            json={"data": [1, 2, 3]}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "not supported" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_proba_prediction_call', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.validate_prediction_data', new_callable=AsyncMock)
    async def test_predict_proba_with_schema(self, mock_validate, mock_get_schemas, mock_proba_call,
                                            client, active_deployment, test_session):
        """Test probability prediction with schema validation"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_validate.return_value = (True, "Valid", {"bedrooms": 3, "bathrooms": 2})
        mock_proba_call.return_value = {
            "probabilities": [[0.7, 0.3]],
            "classes": ["class_0", "class_1"]
        }
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/proba",
            json={"bedrooms": 3, "bathrooms": 2}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["validation"]["validation_performed"] is True
    
    @pytest.mark.asyncio
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.validate_prediction_data', new_callable=AsyncMock)
    async def test_predict_proba_schema_validation_failure(self, mock_validate, mock_get_schemas,
                                                            client, active_deployment, test_session):
        """Test probability prediction with schema validation failure"""
        mock_schema = MagicMock()
        mock_get_schemas.return_value = (mock_schema, None)
        mock_validate.return_value = (False, "Invalid data", {})
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/proba",
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422
        result = response.json()
        assert "Input validation failed" in result["error"]["message"]


class TestPredictionHandlersErrorHandling:
    """Test error handling in prediction handlers"""
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_prediction_call', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    async def test_predict_service_error(self, mock_get_schemas, mock_predict_call,
                                        client, active_deployment, test_session):
        """Test prediction endpoint with service error"""
        mock_get_schemas.return_value = (None, None)
        mock_predict_call.side_effect = Exception("Service unavailable")
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}",
            json={"data": [1, 2, 3]}
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Prediction failed" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_batch_prediction_call', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    async def test_predict_batch_service_error(self, mock_get_schemas, mock_batch_call,
                                              client, active_deployment, test_session):
        """Test batch prediction endpoint with service error"""
        mock_get_schemas.return_value = (None, None)
        mock_batch_call.side_effect = Exception("Service unavailable")
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/batch",
            json={"data": [[1, 2], [3, 4]]}
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Batch prediction failed" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.routes.dynamic.prediction_handlers.make_proba_prediction_call', new_callable=AsyncMock)
    @patch('app.services.schema_service.schema_service.get_model_schemas', new_callable=AsyncMock)
    async def test_predict_proba_service_error(self, mock_get_schemas, mock_proba_call,
                                              client, active_deployment, test_session):
        """Test probability prediction endpoint with service error"""
        mock_get_schemas.return_value = (None, None)
        mock_proba_call.side_effect = Exception("Service unavailable")
        
        response = client.post(
            f"/api/v1/predict/{active_deployment.id}/proba",
            json={"data": [1, 2, 3]}
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Probability prediction failed" in result["error"]["message"]

