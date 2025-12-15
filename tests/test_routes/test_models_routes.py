"""
Comprehensive tests for models routes
Tests all model management endpoints including upload, CRUD, validation, and metrics
"""

import pytest
import io
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from datetime import datetime
from pathlib import Path

from app.models.model import Model
from app.schemas.model import ModelStatus


@pytest.fixture
def sample_model_data():
    """Sample model data for testing"""
    return {
        "name": "test_model_route",
        "description": "Test model for routes",
        "model_type": "classification",
        "framework": "sklearn",
        "version": "1.0.0"
    }


@pytest.fixture
def temp_model_file_content():
    """Create temporary model file content"""
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    
    file_content = io.BytesIO()
    pickle.dump(model, file_content)
    file_content.seek(0)
    return file_content.getvalue()


class TestUploadModel:
    """Test model upload endpoint"""
    
    @pytest.mark.asyncio
    @patch('app.routes.models.ModelFileManager.save_uploaded_file_async', new_callable=AsyncMock)
    @patch('app.routes.models.ModelValidator.validate_model_file_async', new_callable=AsyncMock)
    @patch('app.routes.models.ModelValidator.calculate_file_hash_async', new_callable=AsyncMock)
    @patch('app.routes.models.ModelValidator.detect_framework_from_file_async', new_callable=AsyncMock)
    async def test_upload_model_success(self, mock_detect, mock_hash, mock_validate, mock_save,
                                       client, temp_model_file_content, test_session):
        """Test successful model upload"""
        mock_save.return_value = "/tmp/test_model.joblib"
        mock_validate.return_value = True
        mock_hash.return_value = "abc123hash"
        mock_detect.return_value = "sklearn"
        
        files = {"file": ("test_model.joblib", io.BytesIO(temp_model_file_content), "application/octet-stream")}
        data = {
            "name": "upload_test_model",
            "description": "Test upload",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 201
        result = response.json()
        assert "model" in result
        assert result["model"]["name"] == "upload_test_model"
        assert result["model"]["framework"] == "sklearn"
        assert "message" in result
    
    @pytest.mark.asyncio
    async def test_upload_model_duplicate_name(self, client, test_model, test_session):
        """Test model upload with duplicate name"""
        files = {"file": ("test.joblib", io.BytesIO(b"fake content"), "application/octet-stream")}
        data = {
            "name": test_model.name,  # Use existing name
            "description": "Duplicate",
            "model_type": "classification",
            "framework": "sklearn"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert "already exists" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_upload_model_invalid_extension(self, client):
        """Test model upload with invalid file extension"""
        files = {"file": ("test.txt", io.BytesIO(b"content"), "text/plain")}
        data = {
            "name": "invalid_model",
            "description": "Test",
            "model_type": "classification",
            "framework": "sklearn"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert "Unsupported file format" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.routes.models.ModelFileManager.save_uploaded_file_async', new_callable=AsyncMock)
    @patch('app.routes.models.ModelValidator.validate_model_file_async', new_callable=AsyncMock)
    @patch('app.routes.models.aios.remove', new_callable=AsyncMock)
    async def test_upload_model_invalid_file(self, mock_remove, mock_validate, mock_save,
                                            client, temp_model_file_content):
        """Test model upload with invalid model file"""
        mock_save.return_value = "/tmp/invalid.joblib"
        mock_validate.return_value = False  # Invalid file
        
        files = {"file": ("invalid.joblib", io.BytesIO(temp_model_file_content), "application/octet-stream")}
        data = {
            "name": "invalid_model",
            "description": "Test",
            "model_type": "classification",
            "framework": "sklearn"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == 400
        result = response.json()
        assert "Invalid model file" in result["error"]["message"]
        mock_remove.assert_called_once()  # File should be cleaned up


class TestGetModels:
    """Test get models endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_models_empty(self, client, test_session):
        """Test getting models when none exist"""
        response = client.get("/api/v1/models/")
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_models_with_data(self, client, test_model, test_session):
        """Test getting models with data"""
        response = client.get("/api/v1/models/")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 1
        assert any(m["id"] == test_model.id for m in result)
    
    @pytest.mark.asyncio
    async def test_get_models_with_filter(self, client, test_model, test_session):
        """Test getting models with type filter"""
        response = client.get(f"/api/v1/models/?model_type={test_model.model_type}")
        
        assert response.status_code == 200
        result = response.json()
        assert all(m["model_type"] == test_model.model_type for m in result)
    
    @pytest.mark.asyncio
    async def test_get_models_with_pagination(self, client, test_model, test_session):
        """Test getting models with pagination"""
        response = client.get("/api/v1/models/?skip=0&limit=10")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) <= 10
    
    @pytest.mark.asyncio
    async def test_get_model_by_id_success(self, client, test_model, test_session):
        """Test getting a specific model by ID"""
        response = client.get(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 200
        result = response.json()
        assert "model" in result
        assert result["model"]["id"] == test_model.id
        assert result["model"]["name"] == test_model.name
    
    @pytest.mark.asyncio
    async def test_get_model_by_id_not_found(self, client, test_session):
        """Test getting non-existent model"""
        response = client.get("/api/v1/models/nonexistent_id")
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()


class TestCreateModel:
    """Test create model endpoint"""
    
    @pytest.mark.asyncio
    async def test_create_model_success(self, client, sample_model_data, test_session):
        """Test successful model creation"""
        response = client.post("/api/v1/models/", json=sample_model_data)
        
        assert response.status_code == 201
        result = response.json()
        assert "model" in result
        assert result["model"]["name"] == sample_model_data["name"]
        assert "message" in result
    
    @pytest.mark.asyncio
    async def test_create_model_invalid_data(self, client):
        """Test model creation with invalid data"""
        invalid_data = {"invalid": "data"}
        
        response = client.post("/api/v1/models/", json=invalid_data)
        
        assert response.status_code == 422


class TestUpdateModel:
    """Test update model endpoint"""
    
    @pytest.mark.asyncio
    async def test_update_model_success(self, client, test_model, test_session):
        """Test successful model update"""
        update_data = {
            "description": "Updated description",
            "version": "2.0.0"
        }
        
        response = client.put(f"/api/v1/models/{test_model.id}", json=update_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["model"]["description"] == "Updated description"
        assert result["model"]["version"] == "2.0.0"
    
    @pytest.mark.asyncio
    async def test_update_model_not_found(self, client, test_session):
        """Test updating non-existent model"""
        update_data = {"description": "Updated"}
        
        response = client.put("/api/v1/models/nonexistent", json=update_data)
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()


class TestDeleteModel:
    """Test delete model endpoint"""
    
    @pytest.mark.asyncio
    @patch('app.routes.models.aios.path.exists', new_callable=AsyncMock)
    @patch('app.routes.models.aios.remove', new_callable=AsyncMock)
    async def test_delete_model_success(self, mock_remove, mock_exists, client, test_model, test_session):
        """Test successful model deletion"""
        mock_exists.return_value = True
        
        response = client.delete(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 204
        mock_remove.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, client, test_session):
        """Test deleting non-existent model"""
        response = client.delete("/api/v1/models/nonexistent")
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.routes.models.aios.path.exists', new_callable=AsyncMock)
    @patch('app.routes.models.aios.remove', new_callable=AsyncMock)
    async def test_delete_model_file_not_exists(self, mock_remove, mock_exists, client, test_model, test_session):
        """Test model deletion when file doesn't exist"""
        mock_exists.return_value = False
        
        response = client.delete(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 204
        # Should still succeed even if file doesn't exist


class TestValidateModelInput:
    """Test validate model input endpoint"""
    
    @pytest.mark.asyncio
    async def test_validate_model_input_success(self, client, test_model, test_session):
        """Test successful input validation"""
        input_data = {"feature1": 1.0, "feature2": 2.0}
        
        response = client.post(f"/api/v1/models/{test_model.id}/validate", json=input_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["valid"] is True
        assert result["model_id"] == test_model.id
    
    @pytest.mark.asyncio
    async def test_validate_model_input_not_found(self, client, test_session):
        """Test input validation for non-existent model"""
        input_data = {"feature1": 1.0}
        
        response = client.post("/api/v1/models/nonexistent/validate", json=input_data)
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()


class TestModelMetrics:
    """Test model metrics endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_model_metrics_success(self, client, test_model, test_session):
        """Test getting model metrics"""
        response = client.get(f"/api/v1/models/{test_model.id}/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == test_model.id
        assert "total_predictions" in result
        assert "avg_response_time" in result
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_get_model_metrics_not_found(self, client, test_session):
        """Test getting metrics for non-existent model"""
        response = client.get("/api/v1/models/nonexistent/metrics")
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_update_model_metrics_success(self, client, test_model, test_session):
        """Test updating model metrics"""
        metrics_data = {
            "total_predictions": 100,
            "avg_response_time": 45.5
        }
        
        response = client.post(f"/api/v1/models/{test_model.id}/metrics", json=metrics_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == test_model.id
        assert "updated_metrics" in result
        assert result["updated_metrics"]["total_predictions"] == 100
    
    @pytest.mark.asyncio
    async def test_update_model_metrics_not_found(self, client, test_session):
        """Test updating metrics for non-existent model"""
        metrics_data = {"total_predictions": 100}
        
        response = client.post("/api/v1/models/nonexistent/metrics", json=metrics_data)
        
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["error"]["message"].lower()

