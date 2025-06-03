"""
Comprehensive tests for deployment routes
Tests all deployment REST API endpoints including creation, management, testing, and metrics
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.models.model import ModelDeployment
from app.schemas.model import ModelDeploymentCreate, DeploymentStatus


@pytest.fixture
def deployment_client():
    """Create test client for deployment routes"""
    return TestClient(app)


@pytest.fixture
def sample_deployment_data():
    """Sample deployment creation data"""
    return {
        "model_id": "test_model_123",
        "service_name": "test_service",
        "deployment_name": "test_deployment",  
        "resource_requirements": {
            "cpu": "500m",
            "memory": "1Gi"
        },
        "scaling_config": {
            "min_replicas": 1,
            "max_replicas": 3
        }
    }


class TestCreateDeployment:
    """Test deployment creation endpoint"""
    
    @patch('app.services.deployment_service.deployment_service.create_deployment')
    def test_create_deployment_success(self, mock_create, deployment_client, sample_deployment_data, test_model):
        """Test successful deployment creation"""
        # Mock successful deployment creation
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.PENDING,
            configuration={},
            replicas=1
        )
        mock_create.return_value = (True, "Deployment created successfully", mock_deployment)
        
        response = deployment_client.post("/api/v1/deployments/", json=sample_deployment_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        result = response.json()
        assert result["id"] == "deploy_123"
        assert result["deployment_name"] == "test_deployment"
        assert result["status"] == DeploymentStatus.PENDING
        mock_create.assert_called_once()
    
    @patch('app.services.deployment_service.deployment_service.create_deployment')
    def test_create_deployment_failure(self, mock_create, deployment_client, sample_deployment_data):
        """Test deployment creation failure"""
        mock_create.return_value = (False, "Model not found", None)
        
        response = deployment_client.post("/api/v1/deployments/", json=sample_deployment_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Model not found" in result["detail"]
    
    def test_create_deployment_invalid_data(self, deployment_client):
        """Test deployment creation with invalid data"""
        invalid_data = {"invalid": "data"}
        
        response = deployment_client.post("/api/v1/deployments/", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGetDeployments:
    """Test deployment listing endpoints"""
    
    @patch('app.database.get_session')
    def test_get_deployments_empty(self, mock_get_session, deployment_client):
        """Test getting empty deployment list"""
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = []
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = deployment_client.get("/api/v1/deployments/")
        
        assert response.status_code == 200
        result = response.json()
        assert result == []
    
    @patch('app.database.get_session')
    def test_get_deployments_with_data(self, mock_get_session, deployment_client, test_model):
        """Test getting deployments with data"""
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.RUNNING,
            configuration={},
            replicas=1
        )
        
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [mock_deployment]
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = deployment_client.get("/api/v1/deployments/")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["id"] == "deploy_123"


class TestGetDeploymentById:
    """Test individual deployment retrieval"""
    
    @patch('app.database.get_session')
    def test_get_deployment_success(self, mock_get_session, deployment_client, test_model):
        """Test successful deployment retrieval"""
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.RUNNING,
            configuration={},
            replicas=1
        )
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_deployment
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = deployment_client.get("/api/v1/deployments/deploy_123")
        
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "deploy_123"
        assert result["status"] == DeploymentStatus.RUNNING
    
    @patch('app.database.get_session')
    def test_get_deployment_not_found(self, mock_get_session, deployment_client):
        """Test deployment not found"""
        mock_session = MagicMock()
        mock_session.get.return_value = None
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = deployment_client.get("/api/v1/deployments/nonexistent")
        
        assert response.status_code == 404


class TestUpdateDeployment:
    """Test deployment update endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.update_deployment')
    @patch('app.database.get_session')
    def test_update_deployment_success(self, mock_get_session, mock_update, deployment_client, test_model):
        """Test successful deployment update"""
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.RUNNING,
            configuration={},
            replicas=1
        )
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_deployment
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_update.return_value = (True, "Updated successfully", mock_deployment)
        
        update_data = {"replicas": 3}
        response = deployment_client.put("/api/v1/deployments/deploy_123", json=update_data)
        
        assert response.status_code == 200
        mock_update.assert_called_once()
    
    @patch('app.database.get_session')
    def test_update_deployment_not_found(self, mock_get_session, deployment_client):
        """Test updating non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get.return_value = None
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        update_data = {"replicas": 3}
        response = deployment_client.put("/api/v1/deployments/nonexistent", json=update_data)
        
        assert response.status_code == 404


class TestDeleteDeployment:
    """Test deployment deletion endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.delete_deployment')
    @patch('app.database.get_session')
    def test_delete_deployment_success(self, mock_get_session, mock_delete, deployment_client, test_model):
        """Test successful deployment deletion"""
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment", 
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.STOPPED,
            configuration={},
            replicas=1
        )
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_deployment
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_delete.return_value = (True, "Deleted successfully")
        
        response = deployment_client.delete("/api/v1/deployments/deploy_123")
        
        assert response.status_code == 204
        mock_delete.assert_called_once()
    
    @patch('app.database.get_session')
    def test_delete_deployment_not_found(self, mock_get_session, deployment_client):
        """Test deleting non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get.return_value = None
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = deployment_client.delete("/api/v1/deployments/nonexistent")
        
        assert response.status_code == 404


class TestDeploymentControl:
    """Test deployment control endpoints (start/stop/scale)"""
    
    @patch('app.services.deployment_service.deployment_service.start_deployment')
    @patch('app.database.get_session')
    def test_start_deployment(self, mock_get_session, mock_start, deployment_client, test_model):
        """Test starting a deployment"""
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.STOPPED,
            configuration={},
            replicas=1
        )
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_deployment
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_start.return_value = (True, "Started successfully")
        
        response = deployment_client.post("/api/v1/deployments/deploy_123/start")
        
        assert response.status_code == 200
        mock_start.assert_called_once()
    
    @patch('app.services.deployment_service.deployment_service.stop_deployment')
    @patch('app.database.get_session')
    def test_stop_deployment(self, mock_get_session, mock_stop, deployment_client, test_model):
        """Test stopping a deployment"""
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.RUNNING,
            configuration={},
            replicas=1
        )
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_deployment
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_stop.return_value = (True, "Stopped successfully")
        
        response = deployment_client.post("/api/v1/deployments/deploy_123/stop")
        
        assert response.status_code == 200
        mock_stop.assert_called_once()
    
    @patch('app.services.deployment_service.deployment_service.scale_deployment')
    @patch('app.database.get_session')
    def test_scale_deployment(self, mock_get_session, mock_scale, deployment_client, test_model):
        """Test scaling a deployment"""
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status=DeploymentStatus.RUNNING,
            configuration={},
            replicas=1
        )
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_deployment
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_scale.return_value = (True, "Scaled successfully")
        
        scale_data = {"replicas": 3}
        response = deployment_client.post("/api/v1/deployments/deploy_123/scale", json=scale_data)
        
        assert response.status_code == 200
        mock_scale.assert_called_once()


class TestDeploymentTesting:
    """Test deployment testing endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.test_deployment')
    def test_test_deployment_success(self, mock_test, deployment_client):
        """Test deployment testing with sample data"""
        mock_result = {
            "deployment_id": "deploy_123",
            "predictions": [0.75, 0.25],
            "test_successful": True
        }
        mock_test.return_value = (True, "Test successful", mock_result)
        
        test_data = {"data": {"feature1": 0.5, "feature2": "test"}}
        response = deployment_client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["test_successful"] is True
        assert "predictions" in result
    
    @patch('app.services.deployment_service.deployment_service.test_deployment')
    def test_test_deployment_failure(self, mock_test, deployment_client):
        """Test deployment testing failure"""
        mock_test.return_value = (False, "Deployment not active", None)
        
        test_data = {"data": {"feature1": 0.5}}
        response = deployment_client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        
        assert response.status_code == 400
        result = response.json()
        assert "Deployment not active" in result["detail"]


class TestDeploymentMetrics:
    """Test deployment metrics endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.get_deployment_metrics')
    def test_get_deployment_metrics_success(self, mock_metrics, deployment_client):
        """Test getting deployment metrics""" 
        mock_metrics_data = {
            "deployment_id": "deploy_123",
            "requests_count": 150,
            "average_latency_ms": 45.2,
            "error_rate": 0.02,
            "uptime_percentage": 99.8
        }
        mock_metrics.return_value = mock_metrics_data
        
        response = deployment_client.get("/api/v1/deployments/deploy_123/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert result["deployment_id"] == "deploy_123"
        assert result["requests_count"] == 150
        assert result["average_latency_ms"] == 45.2
    
    @patch('app.services.deployment_service.deployment_service.get_deployment_metrics')
    def test_get_deployment_metrics_not_found(self, mock_metrics, deployment_client):
        """Test getting metrics for non-existent deployment"""
        mock_metrics.return_value = None
        
        response = deployment_client.get("/api/v1/deployments/nonexistent/metrics")
        
        assert response.status_code == 404


class TestDeploymentErrorHandling:
    """Test error handling in deployment routes"""
    
    @patch('app.services.deployment_service.deployment_service.create_deployment')
    def test_create_deployment_service_error(self, mock_create, deployment_client, sample_deployment_data):
        """Test handling service errors during deployment creation"""
        mock_create.side_effect = Exception("Service unavailable")
        
        response = deployment_client.post("/api/v1/deployments/", json=sample_deployment_data)
        
        assert response.status_code == 500
    
    @patch('app.database.get_session')
    def test_database_error_handling(self, mock_get_session, deployment_client):
        """Test handling database errors"""
        mock_get_session.side_effect = Exception("Database connection failed")
        
        response = deployment_client.get("/api/v1/deployments/")
        
        assert response.status_code == 500
    
    def test_invalid_json_request(self, deployment_client):
        """Test handling invalid JSON in requests"""
        response = deployment_client.post(
            "/api/v1/deployments/deploy_123/test", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestDeploymentIntegration:
    """Integration tests for deployment workflows"""
    
    @patch('app.services.deployment_service.deployment_service.create_deployment')
    @patch('app.services.deployment_service.deployment_service.start_deployment')
    @patch('app.services.deployment_service.deployment_service.test_deployment')
    def test_complete_deployment_lifecycle(self, mock_test, mock_start, mock_create, 
                                         deployment_client, sample_deployment_data, test_model):
        """Test complete deployment lifecycle: create -> start -> test"""
        # Mock deployment creation
        mock_deployment = ModelDeployment(
            id="deploy_123",
            deployment_name="test_deployment",
            model_id=test_model.id,
            deployment_url="http://localhost:3001", 
            status=DeploymentStatus.PENDING,
            configuration={},
            replicas=1
        )
        mock_create.return_value = (True, "Created", mock_deployment)
        mock_start.return_value = (True, "Started")
        mock_test.return_value = (True, "Test successful", {"predictions": [0.8, 0.2]})
        
        # Create deployment
        create_response = deployment_client.post("/api/v1/deployments/", json=sample_deployment_data)
        assert create_response.status_code == 201
        
        # Start deployment  
        start_response = deployment_client.post("/api/v1/deployments/deploy_123/start")
        assert start_response.status_code == 200
        
        # Test deployment
        test_data = {"data": {"feature1": 0.5}}
        test_response = deployment_client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        assert test_response.status_code == 200
        
        # Verify all services were called
        mock_create.assert_called_once()
        mock_start.assert_called_once()
        mock_test.assert_called_once() 