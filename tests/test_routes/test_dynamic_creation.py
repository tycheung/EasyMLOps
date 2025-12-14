"""
Tests for dynamic route creation, model upload, and deployment
Tests route registration, model management, and lifecycle operations
"""

import pytest
import json
import io
from datetime import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import uuid

from app.core.app_factory import create_app
from app.models.model import ModelDeployment
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


class TestDynamicRouteManager:
    """Test dynamic route manager functionality"""
    
    def test_register_deployment_route(self, sample_deployment):
        """Test registering a deployment route"""
        route_manager.active_routes.clear()
        
        # Register route
        import asyncio
        asyncio.run(route_manager.register_deployment_route(sample_deployment))
        
        # Verify route was registered
        assert sample_deployment.id in route_manager.active_routes
        route_info = route_manager.active_routes[sample_deployment.id]
        assert route_info['deployment_id'] == sample_deployment.id
        assert route_info['model_id'] == sample_deployment.model_id
        assert route_info['framework'] == 'sklearn'
    
    def test_unregister_deployment_route(self, sample_deployment):
        """Test unregistering a deployment route"""
        # First register the route
        route_manager.active_routes[sample_deployment.id] = {
            'deployment_id': sample_deployment.id,
            'model_id': sample_deployment.model_id
        }
        
        # Unregister route
        import asyncio
        asyncio.run(route_manager.unregister_deployment_route(sample_deployment.id))
        
        # Verify route was unregistered
        assert sample_deployment.id not in route_manager.active_routes


class TestModelUpload:
    """Test model upload endpoints"""
    
    def test_upload_model_success(self, client, temp_model_file):
        """Test successful model upload"""
        with open(temp_model_file, "rb") as f:
            files = {"file": ("test_model.joblib", f, "application/octet-stream")}
            data = {
                "name": "test_upload_model",
                "description": "A test model upload",
                "model_type": "classification",
                "framework": "sklearn",
                "version": "1.0.0"
            }
            
            response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 201
        result = response.json()
        assert "model" in result
        assert "message" in result
        assert result["model"]["name"] == "test_upload_model"
        assert result["model"]["model_type"] == "classification"
        assert result["model"]["framework"] == "sklearn"
        assert "id" in result["model"]
    
    def test_upload_model_missing_file(self, client):
        """Test model upload without file"""
        data = {
            "name": "test_model",
            "description": "Test",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", data=data)
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_model_invalid_extension(self, client):
        """Test model upload with invalid file extension"""
        invalid_file = io.BytesIO(b"fake content")
        files = {"file": ("test_model.txt", invalid_file, "text/plain")}
        data = {
            "name": "test_model",
            "description": "Test",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert "Unsupported file format" in result["error"]["message"]
    
    def test_upload_model_duplicate_name(self, client, temp_model_file, test_model):
        """Test model upload with duplicate name"""
        with open(temp_model_file, "rb") as f:
            files = {"file": ("test_model.joblib", f, "application/octet-stream")}
            data = {
                "name": test_model.name,
                "description": "Duplicate name test",
                "model_type": "classification",
                "framework": "sklearn",
                "version": "1.0.0"
            }
            
            response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert "already exists" in result["error"]["message"]
    
    @patch('app.utils.model_utils.loaders.aios.makedirs')
    def test_upload_model_storage_error(self, mock_makedirs, client, temp_model_file):
        """Test model upload with storage error"""
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        with open(temp_model_file, "rb") as f:
            files = {"file": ("test_model.joblib", f, "application/octet-stream")}
            data = {
                "name": "storage_error_test",
                "description": "Test storage error",
                "model_type": "classification",
                "framework": "sklearn",
                "version": "1.0.0"
            }
            
            response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 500


class TestModelListing:
    """Test model listing endpoints"""
    
    def test_list_models_empty(self, client):
        """Test listing models when none exist"""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_list_models_with_data(self, client, test_model):
        """Test listing models with existing data"""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        assert len(result) >= 1
        
        model = result[0]
        assert "id" in model
        assert "name" in model
        assert "model_type" in model
        assert "framework" in model
        assert "created_at" in model
    
    def test_list_models_pagination(self, client, test_session, sample_model_data):
        """Test model listing with pagination"""
        for i in range(15):
            model_data = sample_model_data.copy()
            model_data["name"] = f"test_model_{i}"
            model_data["file_hash"] = f"pagination_hash_{uuid.uuid4().hex[:8]}"
            from app.models.model import Model
            model = Model(**model_data)
            test_session.add(model)
        test_session.commit()
        
        response = client.get("/api/v1/models?skip=0&limit=10")
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 10
        
        response = client.get("/api/v1/models?skip=10&limit=10")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 5
    
    def test_list_models_filter_by_type(self, client, test_session, sample_model_data):
        """Test model listing with type filter"""
        from app.models.model import Model
        
        classification_model = sample_model_data.copy()
        classification_model["name"] = "classification_model"
        classification_model["model_type"] = "classification"
        classification_model["file_hash"] = f"classification_hash_{uuid.uuid4().hex[:8]}"
        
        regression_model = sample_model_data.copy()
        regression_model["name"] = "regression_model"
        regression_model["model_type"] = "regression"
        regression_model["file_hash"] = f"regression_hash_{uuid.uuid4().hex[:8]}"
        
        test_session.add_all([
            Model(**classification_model),
            Model(**regression_model)
        ])
        test_session.commit()
        
        response = client.get("/api/v1/models?model_type=classification")
        assert response.status_code == 200
        result = response.json()
        
        for model in result:
            assert model["model_type"] == "classification"


class TestModelDetails:
    """Test model detail endpoints"""
    
    def test_get_model_by_id(self, client, test_model):
        """Test getting model by ID"""
        response = client.get(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 200
        result = response.json()
        assert "model" in result
        assert result["model"]["id"] == test_model.id
        assert result["model"]["name"] == test_model.name
        assert result["model"]["model_type"] == test_model.model_type
        assert result["model"]["framework"] == test_model.framework
    
    def test_get_model_not_found(self, client):
        """Test getting non-existent model"""
        response = client.get("/api/v1/models/99999")
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"]
    
    def test_get_model_invalid_id(self, client):
        """Test getting model with invalid ID"""
        response = client.get("/api/v1/models/invalid_id")
        
        assert response.status_code == 404


class TestModelManagement:
    """Test model management endpoints"""
    
    def test_update_model(self, client, test_model):
        """Test model update"""
        update_data = {
            "description": "Updated description",
            "version": "2.0.0"
        }
        
        response = client.put(f"/api/v1/models/{test_model.id}", json=update_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "model" in result
        assert "message" in result
        assert result["model"]["id"] == test_model.id
        assert result["model"]["description"] == "Updated description"
    
    def test_update_model_not_found(self, client):
        """Test updating non-existent model"""
        update_data = {"description": "Updated"}
        
        response = client.put("/api/v1/models/99999", json=update_data)
        
        assert response.status_code == 404
    
    def test_delete_model(self, client, test_model):
        """Test model deletion"""
        response = client.delete(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 204
        
        get_response = client.get(f"/api/v1/models/{test_model.id}")
        assert get_response.status_code == 404
    
    def test_delete_model_not_found(self, client):
        """Test deleting non-existent model"""
        response = client.delete("/api/v1/models/99999")
        
        assert response.status_code == 404


class TestModelValidation:
    """Test model validation endpoints"""
    
    def test_validate_model_input(self, client, test_model):
        """Test model input validation"""
        input_data = {
            "feature1": 0.5,
            "feature2": "valid_value"
        }
        
        response = client.post(
            f"/api/v1/models/{test_model.id}/validate",
            json=input_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "valid" in result
        assert result["valid"] is True
    
    def test_validate_model_input_invalid(self, client, test_model):
        """Test model input validation with invalid data"""
        invalid_data = {
            "feature1": "should_be_number",
            "missing_required_field": "value"
        }
        
        response = client.post(
            f"/api/v1/models/{test_model.id}/validate",
            json=invalid_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "valid" in result


class TestModelMetrics:
    """Test model metrics endpoints"""
    
    def test_get_model_metrics(self, client, test_model):
        """Test getting model performance metrics"""
        response = client.get(f"/api/v1/models/{test_model.id}/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert "model_id" in result
        assert "total_predictions" in result
        assert "avg_response_time" in result
        assert result["model_id"] == test_model.id
    
    def test_update_model_metrics(self, client, test_model):
        """Test updating model performance metrics"""
        new_metrics = {
            "prediction_count": 100,
            "avg_response_time": 150.5
        }
        
        response = client.post(
            f"/api/v1/models/{test_model.id}/metrics",
            json=new_metrics
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        assert "updated_metrics" in result


class TestErrorHandling:
    """Test error handling in model routes (creation-related)"""
    
    def test_database_error_handling(self, client):
        """Test handling of database errors"""
        response = client.get("/api/v1/models/99999")
        
        assert response.status_code == 404
    
    def test_file_system_error_handling(self, client):
        """Test handling of file upload errors"""
        invalid_file = io.BytesIO(b"not a valid model file")
        files = {"file": ("invalid_model.joblib", invalid_file, "application/octet-stream")}
        data = {
            "name": "fs_error_test",
            "description": "Test FS error",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code in [201, 400, 422, 500]


class TestIntegrationScenarios:
    """Integration test scenarios (creation-related)"""
    
    def test_complete_model_lifecycle(self, client, temp_model_file):
        """Test complete model lifecycle: upload -> get -> update -> delete"""
        with open(temp_model_file, "rb") as f:
            files = {"file": ("lifecycle_model.joblib", f, "application/octet-stream")}
            data = {
                "name": "lifecycle_test_model",
                "description": "Testing complete lifecycle",
                "model_type": "classification",
                "framework": "sklearn",
                "version": "1.0.0"
            }
            
            upload_response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert upload_response.status_code == 201
        model_data = upload_response.json()
        model_id = model_data["model"]["id"]
        
        get_response = client.get(f"/api/v1/models/{model_id}")
        assert get_response.status_code == 200
        
        update_response = client.put(
            f"/api/v1/models/{model_id}",
            json={"description": "Updated lifecycle model"}
        )
        assert update_response.status_code == 200
        
        delete_response = client.delete(f"/api/v1/models/{model_id}")
        assert delete_response.status_code == 204
        
        final_get_response = client.get(f"/api/v1/models/{model_id}")
        assert final_get_response.status_code == 404

