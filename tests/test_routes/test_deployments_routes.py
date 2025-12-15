"""
Comprehensive tests for deployment routes
Tests all deployment endpoints with various scenarios
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from app.schemas.model import ModelDeploymentResponse, DeploymentStatus, ModelDeploymentCreate, ModelDeploymentUpdate
from fastapi import status


class TestDeploymentRoutes:
    """Test deployment route endpoints"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.create_deployment', new_callable=AsyncMock)
    async def test_create_deployment_success(self, mock_create, client, test_model):
        """Test successful deployment creation"""
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",
            description="Test deployment",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_create.return_value = (True, "Deployment created", mock_response)
        
        deployment_data = {
            "model_id": test_model.id,
            "name": "test_deployment",
            "description": "Test deployment",
            "config": {}
        }
        
        response = client.post("/api/v1/deployments/", json=deployment_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        result = response.json()
        assert result["id"] == "deploy_123"
        assert result["status"] == DeploymentStatus.ACTIVE.value
        mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.create_deployment', new_callable=AsyncMock)
    async def test_create_deployment_failure(self, mock_create, client):
        """Test deployment creation failure"""
        mock_create.return_value = (False, "Model not found", None)
        
        deployment_data = {
            "model_id": "nonexistent",
            "name": "test_deployment",
            "config": {}
        }
        
        response = client.post("/api/v1/deployments/", json=deployment_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Model not found" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.list_deployments', new_callable=AsyncMock)
    async def test_list_deployments(self, mock_list, client):
        """Test listing deployments"""
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id="model_123",
            name="test_deployment",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_list.return_value = [mock_response]
        
        response = client.get("/api/v1/deployments/")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["id"] == "deploy_123"
        mock_list.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.list_deployments', new_callable=AsyncMock)
    async def test_list_deployments_with_filters(self, mock_list, client):
        """Test listing deployments with filters"""
        mock_list.return_value = []
        
        response = client.get(
            "/api/v1/deployments/",
            params={"model_id": "model_123", "status": "active", "limit": 10, "offset": 0}
        )
        
        assert response.status_code == 200
        mock_list.assert_called_once_with(
            model_id="model_123",
            status=DeploymentStatus.ACTIVE,
            limit=10,
            offset=0
        )
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.get_deployment', new_callable=AsyncMock)
    async def test_get_deployment_success(self, mock_get, client):
        """Test getting a deployment"""
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id="model_123",
            name="test_deployment",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_get.return_value = mock_response
        
        response = client.get("/api/v1/deployments/deploy_123")
        
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "deploy_123"
        mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.get_deployment', new_callable=AsyncMock)
    async def test_get_deployment_not_found(self, mock_get, client):
        """Test getting non-existent deployment"""
        mock_get.return_value = None
        
        response = client.get("/api/v1/deployments/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_update_deployment_patch(self, mock_update, client):
        """Test PATCH update deployment"""
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id="model_123",
            name="updated_deployment",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={"replicas": 3},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_update.return_value = (True, "Updated", mock_response)
        
        update_data = {"name": "updated_deployment", "config": {"replicas": 3}}
        response = client.patch("/api/v1/deployments/deploy_123", json=update_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "updated_deployment"
        mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_update_deployment_put(self, mock_update, client):
        """Test PUT update deployment"""
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id="model_123",
            name="updated_deployment",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_update.return_value = (True, "Updated", mock_response)
        
        update_data = {"name": "updated_deployment"}
        response = client.put("/api/v1/deployments/deploy_123", json=update_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "updated_deployment"
        mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_update_deployment_failure(self, mock_update, client):
        """Test update deployment failure"""
        mock_update.return_value = (False, "Deployment not found", None)
        
        update_data = {"name": "updated"}
        response = client.patch("/api/v1/deployments/nonexistent", json=update_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Deployment not found" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.delete_deployment', new_callable=AsyncMock)
    async def test_delete_deployment_success(self, mock_delete, client):
        """Test deleting a deployment"""
        mock_delete.return_value = (True, "Deleted")
        
        response = client.delete("/api/v1/deployments/deploy_123")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_delete.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.delete_deployment', new_callable=AsyncMock)
    async def test_delete_deployment_not_found(self, mock_delete, client):
        """Test deleting non-existent deployment"""
        mock_delete.return_value = (False, "Deployment nonexistent not found")
        
        response = client.delete("/api/v1/deployments/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.delete_deployment', new_callable=AsyncMock)
    async def test_delete_deployment_other_error(self, mock_delete, client):
        """Test deleting deployment with other error"""
        mock_delete.return_value = (False, "Cannot delete active deployment")
        
        response = client.delete("/api/v1/deployments/deploy_123")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Cannot delete" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.get_deployment_status', new_callable=AsyncMock)
    async def test_get_deployment_status_success(self, mock_status, client):
        """Test getting deployment status"""
        mock_status_data = {
            "deployment_id": "deploy_123",
            "deployment_status": DeploymentStatus.ACTIVE,
            "service_status": {"status": "active"},
            "endpoint_url": "http://localhost:3001",
            "service_name": "test_service",
            "framework": "sklearn",
            "endpoints": ["predict"],
            "last_check": datetime.utcnow().isoformat()
        }
        mock_status.return_value = mock_status_data
        
        response = client.get("/api/v1/deployments/deploy_123/status")
        
        assert response.status_code == 200
        result = response.json()
        assert result["deployment_id"] == "deploy_123"
        assert result["deployment_status"] == DeploymentStatus.ACTIVE.value
        mock_status.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.get_deployment_status', new_callable=AsyncMock)
    async def test_get_deployment_status_not_found(self, mock_status, client):
        """Test getting status for non-existent deployment"""
        mock_status.return_value = None
        
        response = client.get("/api/v1/deployments/nonexistent/status")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.test_deployment', new_callable=AsyncMock)
    async def test_test_deployment_success(self, mock_test, client):
        """Test testing a deployment"""
        mock_test_result = {
            "deployment_id": "deploy_123",
            "predictions": [0.75, 0.25],
            "test_successful": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        mock_test.return_value = (True, "Test successful", mock_test_result)
        
        test_data = {"data": [1.0, 2.0, 3.0]}
        response = client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["test_successful"] is True
        assert "predictions" in result
        mock_test.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.test_deployment', new_callable=AsyncMock)
    async def test_test_deployment_failure(self, mock_test, client):
        """Test testing deployment failure"""
        mock_test.return_value = (False, "Deployment not active", None)
        
        test_data = {"data": [1.0, 2.0]}
        response = client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Deployment not active" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.get_deployment_metrics', new_callable=AsyncMock)
    async def test_get_deployment_metrics_success(self, mock_metrics, client):
        """Test getting deployment metrics"""
        mock_metrics_data = {
            "deployment_id": "deploy_123",
            "requests_count": 150,
            "average_latency_ms": 45.2,
            "error_rate": 0.02,
            "uptime_percentage": 99.8
        }
        mock_metrics.return_value = mock_metrics_data
        
        response = client.get("/api/v1/deployments/deploy_123/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert result["deployment_id"] == "deploy_123"
        assert result["requests_count"] == 150
        mock_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.get_deployment_metrics', new_callable=AsyncMock)
    async def test_get_deployment_metrics_not_found(self, mock_metrics, client):
        """Test getting metrics for non-existent deployment"""
        mock_metrics.return_value = None
        
        response = client.get("/api/v1/deployments/nonexistent/metrics")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "not found" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.start_deployment', new_callable=AsyncMock)
    async def test_start_deployment_success(self, mock_start, client):
        """Test starting a deployment"""
        mock_start.return_value = (True, "Deployment started successfully")
        
        response = client.post("/api/v1/deployments/deploy_123/start")
        
        assert response.status_code == 200
        result = response.json()
        assert "started successfully" in result["message"].lower()
        mock_start.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.start_deployment', new_callable=AsyncMock)
    async def test_start_deployment_failure(self, mock_start, client):
        """Test starting deployment failure"""
        mock_start.return_value = (False, "Deployment not found")
        
        response = client.post("/api/v1/deployments/nonexistent/start")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Deployment not found" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.stop_deployment', new_callable=AsyncMock)
    async def test_stop_deployment_success(self, mock_stop, client):
        """Test stopping a deployment"""
        mock_stop.return_value = (True, "Deployment stopped successfully")
        
        response = client.post("/api/v1/deployments/deploy_123/stop")
        
        assert response.status_code == 200
        result = response.json()
        assert "stopped successfully" in result["message"].lower()
        mock_stop.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.stop_deployment', new_callable=AsyncMock)
    async def test_stop_deployment_failure(self, mock_stop, client):
        """Test stopping deployment failure"""
        mock_stop.return_value = (False, "Deployment not found")
        
        response = client.post("/api/v1/deployments/nonexistent/stop")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Deployment not found" in result["error"]["message"]
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.scale_deployment', new_callable=AsyncMock)
    async def test_scale_deployment_success(self, mock_scale, client):
        """Test scaling a deployment"""
        mock_scale.return_value = (True, "Deployment scaled successfully")
        
        scale_data = {"replicas": 5}
        response = client.post("/api/v1/deployments/deploy_123/scale", json=scale_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "scaled successfully" in result["message"].lower()
        mock_scale.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.scale_deployment', new_callable=AsyncMock)
    async def test_scale_deployment_failure(self, mock_scale, client):
        """Test scaling deployment failure"""
        mock_scale.return_value = (False, "Deployment not found")
        
        scale_data = {"replicas": 3}
        response = client.post("/api/v1/deployments/nonexistent/scale", json=scale_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Deployment not found" in result["error"]["message"]

