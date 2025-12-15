"""
Comprehensive tests for lifecycle routes
Tests retraining jobs and model card endpoints
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestLifecycleRoutes:
    """Test lifecycle route endpoints"""
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.create_retraining_job', new_callable=AsyncMock)
    async def test_create_retraining_job_success(self, mock_create, client):
        """Test creating a retraining job"""
        mock_create.return_value = "job_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/retraining/jobs",
            params={"trigger_type": "scheduled"},
            json={"schedule": "daily"}
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "job_123"
        assert "message" in result
        mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.create_retraining_job', new_callable=AsyncMock)
    async def test_create_retraining_job_error(self, mock_create, client):
        """Test creating retraining job with error"""
        mock_create.side_effect = Exception("Model not found")
        
        response = client.post(
            "/api/v1/monitoring/models/nonexistent/retraining/jobs",
            params={"trigger_type": "scheduled"}
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Model not found" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.get_model_card', new_callable=AsyncMock)
    async def test_get_model_card_success(self, mock_get_card, client):
        """Test getting model card"""
        mock_card_data = {
            "model_id": "test_model",
            "model_name": "Test Model",
            "version": "1.0.0",
            "description": "Test model card",
            "performance_metrics": {},
            "training_info": {},
            "created_at": "2024-01-01T00:00:00"
        }
        mock_get_card.return_value = mock_card_data
        
        response = client.get("/api/v1/monitoring/models/test_model/card")
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == "test_model"
        mock_get_card.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.get_model_card', new_callable=AsyncMock)
    async def test_get_model_card_error(self, mock_get_card, client):
        """Test getting model card with error"""
        mock_get_card.side_effect = Exception("Model not found")
        
        response = client.get("/api/v1/monitoring/models/nonexistent/card")
        
        assert response.status_code == 500
        result = response.json()
        assert "Model not found" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.configure_retraining_trigger', new_callable=AsyncMock)
    async def test_configure_retraining_trigger_success(self, mock_configure, client):
        """Test configuring retraining trigger"""
        mock_configure.return_value = "trigger_123"
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/retraining/triggers",
            params={"trigger_type": "performance_degradation"},
            json={"threshold": 0.1}
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "trigger_123"
        assert "message" in result
        mock_configure.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.configure_retraining_trigger', new_callable=AsyncMock)
    async def test_configure_retraining_trigger_error(self, mock_configure, client):
        """Test configuring retraining trigger with error"""
        mock_configure.side_effect = Exception("Invalid trigger type")
        
        response = client.post(
            "/api/v1/monitoring/models/test_model/retraining/triggers",
            params={"trigger_type": "invalid"}
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Invalid trigger type" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.generate_model_card', new_callable=AsyncMock)
    async def test_generate_model_card_success(self, mock_generate, client):
        """Test generating model card"""
        mock_card_data = {
            "model_id": "test_model",
            "model_name": "Test Model",
            "version": "1.0.0",
            "description": "Generated model card",
            "performance_metrics": {},
            "training_info": {},
            "created_at": "2024-01-01T00:00:00"
        }
        mock_generate.return_value = mock_card_data
        
        response = client.post("/api/v1/monitoring/models/test_model/card/generate")
        
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == "test_model"
        mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring_service.monitoring_service.generate_model_card', new_callable=AsyncMock)
    async def test_generate_model_card_error(self, mock_generate, client):
        """Test generating model card with error"""
        mock_generate.side_effect = Exception("Model not found")
        
        response = client.post("/api/v1/monitoring/models/nonexistent/card/generate")
        
        assert response.status_code == 500
        result = response.json()
        assert "Model not found" in result["error"]["message"]

