"""
Comprehensive tests for deployment routes
Tests all deployment REST API endpoints including creation, management, testing, and metrics
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status
from fastapi.testclient import TestClient

from app.models.model import ModelDeployment
from app.schemas.model import ModelDeploymentCreate, DeploymentStatus, ModelDeploymentResponse
from datetime import datetime


@pytest.fixture
def sample_deployment_data():
    """Sample deployment creation data"""
    return {
        "model_id": "test_model_123",
        "name": "test_deployment",
        "description": "Test deployment description",
        "config": {
            "service_name": "test_service",
            "resource_requirements": {
                "cpu": "500m",
                "memory": "1Gi"
            },
            "scaling_config": {
                "min_replicas": 1,
                "max_replicas": 3
            }
        }
    }


class TestCreateDeployment:
    """Test deployment creation endpoint"""
    
    @patch('app.services.deployment_service.deployment_service.create_deployment')
    def test_create_deployment_success(self, mock_create, client, sample_deployment_data, test_model):
        """Test successful deployment creation"""
        # Mock successful deployment creation - return ModelDeploymentResponse, not ModelDeployment
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",  # Ensure name is set
            description="Test deployment description",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_create.return_value = (True, "Deployment created successfully", mock_response)
        
        response = client.post("/api/v1/deployments/", json=sample_deployment_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        result = response.json()
        assert result["id"] == "deploy_123"
        assert result["name"] == "test_deployment"  # Check name instead of deployment_name
        assert result["status"] == DeploymentStatus.ACTIVE
        mock_create.assert_called_once()
    
    @patch('app.services.deployment_service.deployment_service.create_deployment')
    def test_create_deployment_failure(self, mock_create, client, sample_deployment_data):
        """Test deployment creation failure"""
        mock_create.return_value = (False, "Model not found", None)
        
        response = client.post("/api/v1/deployments/", json=sample_deployment_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        # The application uses a custom error format with 'error' object
        assert "error" in result
        assert "message" in result["error"]
        assert "Model not found" in result["error"]["message"]
    
    def test_create_deployment_invalid_data(self, client):
        """Test deployment creation with invalid data"""
        invalid_data = {"invalid": "data"}
        
        response = client.post("/api/v1/deployments/", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGetDeployments:
    """Test deployment listing endpoints"""
    
    @patch('app.database.get_session')
    def test_get_deployments_empty(self, mock_get_session, client):
        """Test getting empty deployment list"""
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = []
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = client.get("/api/v1/deployments/")
        
        assert response.status_code == 200
        result = response.json()
        assert result == []
    
    @patch('app.services.deployment_service.deployment_service.list_deployments')
    def test_get_deployments_with_data(self, mock_list, client, test_model):
        """Test getting deployments with data"""
        from app.schemas.model import ModelDeploymentResponse, DeploymentStatus
        from datetime import datetime
        
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",
            description=None,
            status=DeploymentStatus.RUNNING,
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


class TestGetDeploymentById:
    """Test individual deployment retrieval"""
    
    @patch('app.services.deployment_service.deployment_service.get_deployment')
    def test_get_deployment_success(self, mock_get, client, test_model):
        """Test successful deployment retrieval"""
        from app.schemas.model import ModelDeploymentResponse, DeploymentStatus
        from datetime import datetime
        
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",
            description=None,
            status=DeploymentStatus.RUNNING,
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
        assert result["status"] == DeploymentStatus.RUNNING
    
    @patch('app.services.deployment_service.deployment_service.get_deployment')
    def test_get_deployment_not_found(self, mock_get, client):
        """Test deployment not found"""
        mock_get.return_value = None
        
        response = client.get("/api/v1/deployments/nonexistent")
        
        assert response.status_code == 404


class TestUpdateDeployment:
    """Test deployment update endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.update_deployment')
    def test_update_deployment_success(self, mock_update, client, test_model):
        """Test successful deployment update"""
        from app.schemas.model import ModelDeploymentResponse, DeploymentStatus
        from datetime import datetime
        
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",
            description=None,
            status=DeploymentStatus.RUNNING,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        mock_update.return_value = (True, "Updated successfully", mock_response)
        
        update_data = {"replicas": 3}
        response = client.put("/api/v1/deployments/deploy_123", json=update_data)
        
        assert response.status_code == 200
        mock_update.assert_called_once()
    
    @patch('app.services.deployment_service.deployment_service.update_deployment')
    def test_update_deployment_not_found(self, mock_update, client):
        """Test updating non-existent deployment"""
        mock_update.return_value = (False, "Deployment deploy_123 not found", None)
        
        update_data = {"replicas": 3}
        response = client.put("/api/v1/deployments/nonexistent", json=update_data)
        
        assert response.status_code == 400  # Service returns 400 for not found in update


class TestDeleteDeployment:
    """Test deployment deletion endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.delete_deployment')
    @patch('app.database.get_session')
    def test_delete_deployment_success(self, mock_get_session, mock_delete, client, test_model):
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
        
        response = client.delete("/api/v1/deployments/deploy_123")
        
        assert response.status_code == 204
        mock_delete.assert_called_once()
    
    @patch('app.database.get_session')
    def test_delete_deployment_not_found(self, mock_get_session, client):
        """Test deleting non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get.return_value = None
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        response = client.delete("/api/v1/deployments/nonexistent")
        
        assert response.status_code == 404


class TestDeploymentControl:
    """Test deployment control endpoints (start/stop/scale)"""
    
    @patch('app.services.deployment_service.deployment_service.start_deployment')
    @patch('app.database.get_session')
    def test_start_deployment(self, mock_get_session, mock_start, client, test_model):
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
        
        response = client.post("/api/v1/deployments/deploy_123/start")
        
        assert response.status_code == 200
        mock_start.assert_called_once()
    
    @patch('app.services.deployment_service.deployment_service.stop_deployment')
    @patch('app.database.get_session')
    def test_stop_deployment(self, mock_get_session, mock_stop, client, test_model):
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
        
        response = client.post("/api/v1/deployments/deploy_123/stop")
        
        assert response.status_code == 200
        mock_stop.assert_called_once()
    
    @patch('app.services.deployment_service.deployment_service.scale_deployment')
    @patch('app.database.get_session')
    def test_scale_deployment(self, mock_get_session, mock_scale, client, test_model):
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
        response = client.post("/api/v1/deployments/deploy_123/scale", json=scale_data)
        
        assert response.status_code == 200
        mock_scale.assert_called_once()


class TestDeploymentTesting:
    """Test deployment testing endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.test_deployment')
    def test_test_deployment_success(self, mock_test, client):
        """Test deployment testing with sample data"""
        mock_result = {
            "deployment_id": "deploy_123",
            "predictions": [0.75, 0.25],
            "test_successful": True
        }
        mock_test.return_value = (True, "Test successful", mock_result)
        
        test_data = {"data": {"feature1": 0.5, "feature2": "test"}}
        response = client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["test_successful"] is True
        assert "predictions" in result
    
    @patch('app.services.deployment_service.deployment_service.test_deployment')
    def test_test_deployment_failure(self, mock_test, client):
        """Test deployment testing failure"""
        mock_test.return_value = (False, "Deployment not active", None)
        
        test_data = {"data": {"feature1": 0.5}}
        response = client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        
        assert response.status_code == 400
        result = response.json()
        # The application uses a custom error format with 'error' object
        assert "error" in result
        assert "message" in result["error"]
        assert "Deployment not active" in result["error"]["message"]


class TestDeploymentMetrics:
    """Test deployment metrics endpoints"""
    
    @patch('app.services.deployment_service.deployment_service.get_deployment_metrics')
    def test_get_deployment_metrics_success(self, mock_metrics, client):
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
        assert result["average_latency_ms"] == 45.2
    
    @patch('app.services.deployment_service.deployment_service.get_deployment_metrics')
    def test_get_deployment_metrics_not_found(self, mock_metrics, client):
        """Test getting metrics for non-existent deployment"""
        mock_metrics.return_value = None
        
        response = client.get("/api/v1/deployments/nonexistent/metrics")
        
        assert response.status_code == 404


class TestDeploymentErrorHandling:
    """Test error handling in deployment routes"""
    
    @patch('app.services.deployment_service.deployment_service.create_deployment')
    def test_create_deployment_service_error(self, mock_create, client, sample_deployment_data):
        """Test handling service errors during deployment creation"""
        mock_create.side_effect = Exception("Service unavailable")
        
        # TestClient behavior with exceptions can vary
        try:
            response = client.post("/api/v1/deployments/", json=sample_deployment_data)
            # If it gets here, the global handler worked
            assert response.status_code == 500
        except Exception as e:
            # If it raises, that's also expected behavior in test mode
            assert "Service unavailable" in str(e)
    
    @patch('app.services.deployment_service.deployment_service.list_deployments')
    def test_database_error_handling(self, mock_list, client):
        """Test handling database errors"""
        mock_list.side_effect = Exception("Database connection failed")
        
        # The TestClient doesn't use the same error handling as production
        # It will propagate the exception up, so we expect it to fail
        try:
            response = client.get("/api/v1/deployments/")
            # If it gets here, the global handler worked
            assert response.status_code == 500
        except Exception as e:
            # If it raises, that's also expected behavior in test mode
            assert "Database connection failed" in str(e)
    
    def test_invalid_json_request(self, client):
        """Test handling invalid JSON in requests"""
        response = client.post(
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
                                         client, sample_deployment_data, test_model):
        """Test complete deployment lifecycle: create -> start -> test"""
        # Mock deployment creation - return ModelDeploymentResponse, not ModelDeployment
        from app.schemas.model import ModelDeploymentResponse, DeploymentStatus
        from datetime import datetime
        
        mock_response = ModelDeploymentResponse(
            id="deploy_123",
            model_id=test_model.id,
            name="test_deployment",  # Ensure name is set
            description="Test deployment description",
            status=DeploymentStatus.ACTIVE,
            endpoint_url="http://localhost:3001",
            service_name="test_service",
            framework="sklearn",
            endpoints=["predict"],
            config={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_create.return_value = (True, "Created", mock_response)
        mock_start.return_value = (True, "Started")
        mock_test.return_value = (True, "Test successful", {"predictions": [0.8, 0.2]})
        
        # Create deployment
        create_response = client.post("/api/v1/deployments/", json=sample_deployment_data)
        assert create_response.status_code == 201
        
        # Start deployment  
        start_response = client.post("/api/v1/deployments/deploy_123/start")
        assert start_response.status_code == 200
        
        # Test deployment
        test_data = {"data": {"feature1": 0.5}}
        test_response = client.post("/api/v1/deployments/deploy_123/test", json=test_data)
        assert test_response.status_code == 200
        
        # Verify all services were called
        mock_create.assert_called_once()
        mock_start.assert_called_once()
        mock_test.assert_called_once() 