"""
Unit tests for dynamic routes API endpoints
Tests model upload, prediction, listing, and management operations
"""

import pytest
import json
import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile


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
        
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "test_upload_model"
        assert result["model_type"] == "classification"
        assert result["framework"] == "sklearn"
        assert "id" in result
    
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
        # Create a file with invalid extension
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
        assert "Invalid file extension" in response.json()["detail"]
    
    def test_upload_model_duplicate_name(self, client, temp_model_file, test_model):
        """Test model upload with duplicate name"""
        with open(temp_model_file, "rb") as f:
            files = {"file": ("test_model.joblib", f, "application/octet-stream")}
            data = {
                "name": test_model.name,  # Use existing model name
                "description": "Duplicate name test",
                "model_type": "classification",
                "framework": "sklearn",
                "version": "1.0.0"
            }
            
            response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    @patch('app.routes.dynamic.os.makedirs')
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
        
        # Check first model
        model = result[0]
        assert "id" in model
        assert "name" in model
        assert "model_type" in model
        assert "framework" in model
        assert "created_at" in model
    
    def test_list_models_pagination(self, client, test_session, sample_model_data):
        """Test model listing with pagination"""
        # Create multiple models
        for i in range(15):
            model_data = sample_model_data.copy()
            model_data["name"] = f"test_model_{i}"
            from app.models.model import Model
            model = Model(**model_data)
            test_session.add(model)
        test_session.commit()
        
        # Test first page
        response = client.get("/api/v1/models?skip=0&limit=10")
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 10
        
        # Test second page
        response = client.get("/api/v1/models?skip=10&limit=10")
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 5  # At least the remaining models
    
    def test_list_models_filter_by_type(self, client, test_session, sample_model_data):
        """Test model listing with type filter"""
        # Create models of different types
        from app.models.model import Model
        
        classification_model = sample_model_data.copy()
        classification_model["name"] = "classification_model"
        classification_model["model_type"] = "classification"
        
        regression_model = sample_model_data.copy()
        regression_model["name"] = "regression_model"
        regression_model["model_type"] = "regression"
        
        test_session.add_all([
            Model(**classification_model),
            Model(**regression_model)
        ])
        test_session.commit()
        
        # Filter by classification
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
        assert result["id"] == test_model.id
        assert result["name"] == test_model.name
        assert result["model_type"] == test_model.model_type
        assert result["framework"] == test_model.framework
    
    def test_get_model_not_found(self, client):
        """Test getting non-existent model"""
        response = client.get("/api/v1/models/99999")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_model_invalid_id(self, client):
        """Test getting model with invalid ID"""
        response = client.get("/api/v1/models/invalid_id")
        
        assert response.status_code == 422  # Validation error


class TestModelPrediction:
    """Test model prediction endpoints"""
    
    @patch('app.routes.dynamic.load_model_for_prediction')
    def test_predict_success(self, mock_load_model, client, test_model, sample_prediction_data):
        """Test successful prediction"""
        # Mock the model loading and prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = ["class_a"]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        mock_load_model.return_value = mock_model
        
        response = client.post(
            f"/api/v1/models/{test_model.id}/predict",
            json=sample_prediction_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert "timestamp" in result
        assert "model_id" in result
        assert result["model_id"] == test_model.id
    
    def test_predict_model_not_found(self, client, sample_prediction_data):
        """Test prediction with non-existent model"""
        response = client.post(
            "/api/v1/models/99999/predict",
            json=sample_prediction_data
        )
        
        assert response.status_code == 404
    
    @patch('app.routes.dynamic.load_model_for_prediction')
    def test_predict_model_loading_error(self, mock_load_model, client, test_model, sample_prediction_data):
        """Test prediction with model loading error"""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        response = client.post(
            f"/api/v1/models/{test_model.id}/predict",
            json=sample_prediction_data
        )
        
        assert response.status_code == 500
        assert "prediction failed" in response.json()["detail"].lower()
    
    @patch('app.routes.dynamic.load_model_for_prediction')
    def test_predict_invalid_input(self, mock_load_model, client, test_model):
        """Test prediction with invalid input data"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Invalid input data (missing required fields)
        invalid_data = {"invalid_field": "value"}
        
        response = client.post(
            f"/api/v1/models/{test_model.id}/predict",
            json=invalid_data
        )
        
        # Should handle validation error
        assert response.status_code in [400, 422, 500]


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
        assert result["description"] == "Updated description"
        assert result["version"] == "2.0.0"
    
    def test_update_model_not_found(self, client):
        """Test updating non-existent model"""
        update_data = {"description": "Updated"}
        
        response = client.put("/api/v1/models/99999", json=update_data)
        
        assert response.status_code == 404
    
    def test_delete_model(self, client, test_model):
        """Test model deletion (soft delete)"""
        response = client.delete(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 200
        result = response.json()
        assert "deleted" in result["message"].lower()
        
        # Verify model is soft deleted
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
        assert result["valid"] is False
        assert "errors" in result


class TestModelMetrics:
    """Test model metrics endpoints"""
    
    def test_get_model_metrics(self, client, test_model):
        """Test getting model performance metrics"""
        response = client.get(f"/api/v1/models/{test_model.id}/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert "performance_metrics" in result
        # Should return the performance metrics from the model
        if test_model.performance_metrics:
            assert result["performance_metrics"] == test_model.performance_metrics
    
    def test_update_model_metrics(self, client, test_model):
        """Test updating model performance metrics"""
        new_metrics = {
            "accuracy": 0.97,
            "precision": 0.95,
            "recall": 0.98,
            "f1_score": 0.965
        }
        
        response = client.post(
            f"/api/v1/models/{test_model.id}/metrics",
            json=new_metrics
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["performance_metrics"]["accuracy"] == 0.97


class TestErrorHandling:
    """Test error handling in dynamic routes"""
    
    def test_database_error_handling(self, client):
        """Test handling of database errors"""
        with patch('app.routes.dynamic.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_session.query.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_session
            
            response = client.get("/api/v1/models")
            
            # Should handle database errors gracefully
            assert response.status_code == 500
    
    def test_file_system_error_handling(self, client, temp_model_file):
        """Test handling of file system errors"""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with open(temp_model_file, "rb") as f:
                files = {"file": ("test_model.joblib", f, "application/octet-stream")}
                data = {
                    "name": "fs_error_test",
                    "description": "Test FS error",
                    "model_type": "classification",
                    "framework": "sklearn",
                    "version": "1.0.0"
                }
                
                response = client.post("/api/v1/models/upload", files=files, data=data)
            
            assert response.status_code == 500
    
    def test_invalid_json_request(self, client, test_model):
        """Test handling of invalid JSON in requests"""
        response = client.post(
            f"/api/v1/models/{test_model.id}/predict",
            data="invalid json data",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    def test_complete_model_lifecycle(self, client, temp_model_file):
        """Test complete model lifecycle: upload -> predict -> update -> delete"""
        # 1. Upload model
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
        
        assert upload_response.status_code == 200
        model_data = upload_response.json()
        model_id = model_data["id"]
        
        # 2. Get model details
        get_response = client.get(f"/api/v1/models/{model_id}")
        assert get_response.status_code == 200
        
        # 3. Make prediction (mock the actual prediction)
        with patch('app.routes.dynamic.load_model_for_prediction') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = ["class_a"]
            mock_load.return_value = mock_model
            
            predict_response = client.post(
                f"/api/v1/models/{model_id}/predict",
                json={"feature1": 0.5, "feature2": "test"}
            )
            assert predict_response.status_code == 200
        
        # 4. Update model
        update_response = client.put(
            f"/api/v1/models/{model_id}",
            json={"description": "Updated lifecycle model"}
        )
        assert update_response.status_code == 200
        
        # 5. Delete model
        delete_response = client.delete(f"/api/v1/models/{model_id}")
        assert delete_response.status_code == 200
        
        # 6. Verify deletion
        final_get_response = client.get(f"/api/v1/models/{model_id}")
        assert final_get_response.status_code == 404
    
    def test_concurrent_model_operations(self, client, test_model):
        """Test concurrent operations on the same model"""
        model_id = test_model.id
        
        # Simulate concurrent reads
        responses = []
        for _ in range(5):
            response = client.get(f"/api/v1/models/{model_id}")
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == model_id 