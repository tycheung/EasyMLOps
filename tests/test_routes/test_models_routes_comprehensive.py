"""
Comprehensive tests for models routes
Tests all model management endpoints
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import io

from app.schemas.model import ModelResponse, ModelMetadata
from fastapi import status


class TestModelsRoutes:
    """Test models route endpoints"""
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.ModelFileManager.save_uploaded_file_async', new_callable=AsyncMock)
    @patch('app.utils.model_utils.ModelValidator.validate_model_file_async', new_callable=AsyncMock)
    @patch('app.utils.model_utils.ModelValidator.calculate_file_hash_async', new_callable=AsyncMock)
    @patch('app.utils.model_utils.frameworks.FrameworkDetector.detect_framework_from_file_async', new_callable=AsyncMock)
    async def test_upload_model_success(self, mock_detect, mock_hash, mock_validate, mock_save, client, test_session):
        """Test successful model upload"""
        from app.models.model import Model
        from app.schemas.model import ModelValidationResult, ModelFramework
        
        mock_save.return_value = "/path/to/model.pkl"
        validation_result = ModelValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            framework_detected=ModelFramework.SKLEARN,
            model_type_detected=None,
            metadata={"file_hash": "abc123hash"}
        )
        mock_validate.return_value = validation_result
        mock_hash.return_value = "abc123hash"
        mock_detect.return_value = ModelFramework.SKLEARN
        
        file_content = b"fake model content"
        files = {"file": ("model.pkl", io.BytesIO(file_content), "application/octet-stream")}
        data = {
            "name": "test_model",
            "description": "Test model",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == status.HTTP_201_CREATED
        result = response.json()
        assert "model" in result
        assert result["model"]["name"] == "test_model"
        mock_save.assert_called_once()
        mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_model_duplicate_name(self, client, test_model, test_session):
        """Test uploading model with duplicate name"""
        file_content = b"fake model content"
        files = {"file": ("model.pkl", io.BytesIO(file_content), "application/octet-stream")}
        data = {
            "name": test_model.name,
            "description": "Test model",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "already exists" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_upload_model_invalid_format(self, client):
        """Test uploading model with invalid file format"""
        file_content = b"fake content"
        files = {"file": ("model.txt", io.BytesIO(file_content), "text/plain")}
        data = {
            "name": "test_model",
            "description": "Test model",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "unsupported file format" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.ModelFileManager.save_uploaded_file_async', new_callable=AsyncMock)
    @patch('app.utils.model_utils.ModelValidator.validate_model_file_async', new_callable=AsyncMock)
    async def test_upload_model_invalid_file(self, mock_validate, mock_save, client):
        """Test uploading invalid model file"""
        from app.schemas.model import ModelValidationResult
        
        mock_save.return_value = "/path/to/model.pkl"
        validation_result = ModelValidationResult(
            is_valid=False,
            errors=["Invalid model file"],
            warnings=[],
            framework_detected=None,
            model_type_detected=None,
            metadata={}
        )
        mock_validate.return_value = validation_result
        
        file_content = b"invalid model content"
        files = {"file": ("model.pkl", io.BytesIO(file_content), "application/octet-stream")}
        data = {
            "name": "test_model",
            "description": "Test model",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/upload", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "invalid model file" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_get_models(self, client, test_model, test_session):
        """Test getting all models"""
        response = client.get("/api/v1/models/")
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        assert len(result) >= 1
    
    @pytest.mark.asyncio
    async def test_get_models_with_filters(self, client, test_model, test_session):
        """Test getting models with filters"""
        response = client.get("/api/v1/models/", params={"model_type": test_model.model_type, "skip": 0, "limit": 10})
        
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_model_success(self, client, test_model, test_session):
        """Test getting a specific model"""
        response = client.get(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 200
        result = response.json()
        assert "model" in result
        assert result["model"]["id"] == test_model.id
    
    @pytest.mark.asyncio
    async def test_get_model_not_found(self, client):
        """Test getting non-existent model"""
        response = client.get("/api/v1/models/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_create_model(self, client, test_session):
        """Test creating a model without file upload"""
        model_data = {
            "name": "new_model",
            "description": "New model",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/", json=model_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        result = response.json()
        assert "model" in result
        assert result["model"]["name"] == "new_model"
    
    @pytest.mark.asyncio
    async def test_update_model_success(self, client, test_model, test_session):
        """Test updating a model"""
        update_data = {
            "name": "updated_model",
            "description": "Updated description"
        }
        
        response = client.patch(f"/api/v1/models/{test_model.id}", json=update_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "model" in result
        assert result["model"]["name"] == "updated_model"
    
    @pytest.mark.asyncio
    async def test_update_model_not_found(self, client):
        """Test updating non-existent model"""
        update_data = {"name": "updated"}
        
        response = client.patch("/api/v1/models/nonexistent", json=update_data)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_delete_model_success(self, client, test_model, test_session):
        """Test deleting a model"""
        response = client.delete(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
    
    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, client):
        """Test deleting non-existent model"""
        response = client.delete("/api/v1/models/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.ModelValidator.validate_model_input_async', new_callable=AsyncMock)
    async def test_validate_model_input_success(self, mock_validate, client, test_model):
        """Test validating model input"""
        mock_validate.return_value = True
        
        input_data = {"feature1": 1.0, "feature2": 2.0}
        response = client.post(f"/api/v1/models/{test_model.id}/validate", json=input_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["is_valid"] is True
        mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.ModelValidator.validate_model_input_async', new_callable=AsyncMock)
    async def test_validate_model_input_invalid(self, mock_validate, client, test_model):
        """Test validating invalid model input"""
        mock_validate.return_value = False
        
        input_data = {"invalid": "data"}
        response = client.post(f"/api/v1/models/{test_model.id}/validate", json=input_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["is_valid"] is False
        mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.get_model_performance_metrics', new_callable=AsyncMock)
    async def test_get_model_metrics(self, mock_metrics, client, test_model):
        """Test getting model metrics"""
        from app.schemas.monitoring import ModelPerformanceMetrics
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        mock_metrics_data = ModelPerformanceMetrics(
            model_id=test_model.id,
            time_window_start=now - timedelta(hours=1),
            time_window_end=now,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            avg_latency_ms=50.0
        )
        mock_metrics.return_value = mock_metrics_data
        
        response = client.get(f"/api/v1/models/{test_model.id}/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == test_model.id
        assert result["total_requests"] == 100
        mock_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.update_model_metrics', new_callable=AsyncMock)
    async def test_update_model_metrics(self, mock_update, client, test_model):
        """Test updating model metrics"""
        mock_update.return_value = True
        
        metrics_data = {
            "prediction_count": 100,
            "avg_response_time": 50.0
        }
        
        response = client.patch(f"/api/v1/models/{test_model.id}/metrics", json=metrics_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        mock_update.assert_called_once()

