"""
Comprehensive tests for deployment service
Tests all deployment service methods including update, start, stop, scale, test, and metrics
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from app.services.deployment_service import deployment_service
from app.models.model import Model, ModelDeployment
from app.schemas.model import (
    ModelDeploymentCreate,
    ModelDeploymentUpdate,
    ModelDeploymentResponse,
    ModelStatus,
    DeploymentStatus
)


@pytest.fixture
def sample_deployment(test_model, test_session):
    """Create a sample deployment for testing"""
    import uuid
    deployment = ModelDeployment(
        id=str(uuid.uuid4()),
        model_id=test_model.id,
        name="test_deployment",
        deployment_name="test_deployment",
        deployment_url="http://localhost:3000/service",
        status=DeploymentStatus.ACTIVE.value,
        framework="sklearn",
        endpoints=["predict"],
        configuration={"replicas": 1},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    test_session.add(deployment)
    test_session.commit()
    test_session.refresh(deployment)
    return deployment


class TestDeploymentServiceUpdate:
    """Test deployment update functionality"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_update_deployment_name(self, mock_get_session, sample_deployment):
        """Test updating deployment name"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        update_data = ModelDeploymentUpdate(name="updated_name")
        success, message, deployment = await deployment_service.update_deployment(
            sample_deployment.id, update_data
        )
        
        assert success is True
        assert deployment.name == "updated_name"
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.deploy_service', new_callable=AsyncMock)
    async def test_update_deployment_status_to_active(self, mock_deploy, mock_get_session, sample_deployment):
        """Test updating deployment status to active"""
        sample_deployment.status = DeploymentStatus.STOPPED.value
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_deploy.return_value = (True, "Deployed", {"endpoint_url": "http://localhost:3000/new"})
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        update_data = ModelDeploymentUpdate(status=DeploymentStatus.ACTIVE)
        success, message, deployment = await deployment_service.update_deployment(
            sample_deployment.id, update_data
        )
        
        assert success is True
        assert deployment.status == DeploymentStatus.ACTIVE
        mock_deploy.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.undeploy_service', new_callable=AsyncMock)
    async def test_update_deployment_status_to_stopped(self, mock_undeploy, mock_get_session, sample_deployment):
        """Test updating deployment status to stopped"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        update_data = ModelDeploymentUpdate(status=DeploymentStatus.STOPPED)
        success, message, deployment = await deployment_service.update_deployment(
            sample_deployment.id, update_data
        )
        
        assert success is True
        assert deployment.status == DeploymentStatus.STOPPED
        mock_undeploy.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_update_deployment_not_found(self, mock_get_session):
        """Test updating non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        update_data = ModelDeploymentUpdate(name="new_name")
        success, message, deployment = await deployment_service.update_deployment(
            "nonexistent", update_data
        )
        
        assert success is False
        assert "not found" in message.lower()
        assert deployment is None
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_update_deployment_config(self, mock_get_session, sample_deployment):
        """Test updating deployment configuration"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        new_config = {"replicas": 3, "memory": "2Gi"}
        update_data = ModelDeploymentUpdate(config=new_config)
        success, message, deployment = await deployment_service.update_deployment(
            sample_deployment.id, update_data
        )
        
        assert success is True
        assert deployment.config == new_config


class TestDeploymentServiceList:
    """Test deployment listing functionality"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_list_deployments_all(self, mock_get_session, sample_deployment):
        """Test listing all deployments"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [sample_deployment]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        deployments = await deployment_service.list_deployments()
        
        assert len(deployments) == 1
        assert deployments[0].id == sample_deployment.id
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_list_deployments_filter_by_model(self, mock_get_session, sample_deployment):
        """Test listing deployments filtered by model_id"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [sample_deployment]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        deployments = await deployment_service.list_deployments(model_id=sample_deployment.model_id)
        
        assert len(deployments) == 1
        assert deployments[0].model_id == sample_deployment.model_id
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_list_deployments_filter_by_status(self, mock_get_session, sample_deployment):
        """Test listing deployments filtered by status"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [sample_deployment]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        deployments = await deployment_service.list_deployments(status=DeploymentStatus.ACTIVE)
        
        assert len(deployments) == 1
        assert deployments[0].status == DeploymentStatus.ACTIVE
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_list_deployments_with_pagination(self, mock_get_session, sample_deployment):
        """Test listing deployments with pagination"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [sample_deployment]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        deployments = await deployment_service.list_deployments(limit=10, offset=0)
        
        assert len(deployments) <= 10


class TestDeploymentServiceControl:
    """Test deployment control operations"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_start_deployment_success(self, mock_update, sample_deployment):
        """Test starting a deployment"""
        from app.schemas.model import ModelDeploymentResponse
        mock_update.return_value = (
            True, 
            "Deployment updated successfully",
            ModelDeploymentResponse(
                id=sample_deployment.id,
                model_id=sample_deployment.model_id,
                name=sample_deployment.name,
                status=DeploymentStatus.ACTIVE,
                endpoint_url=sample_deployment.deployment_url,
                service_name=sample_deployment.deployment_name,
                framework=sample_deployment.framework,
                endpoints=sample_deployment.endpoints,
                config=sample_deployment.configuration,
                created_at=sample_deployment.created_at,
                updated_at=sample_deployment.updated_at
            )
        )
        
        success, message = await deployment_service.start_deployment(sample_deployment.id)
        
        assert success is True
        assert "updated" in message.lower() or "started" in message.lower()
        mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_start_deployment_not_found(self, mock_get_session):
        """Test starting non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message = await deployment_service.start_deployment("nonexistent")
        
        assert success is False
        assert "not found" in message.lower()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_stop_deployment_success(self, mock_update, sample_deployment):
        """Test stopping a deployment"""
        from app.schemas.model import ModelDeploymentResponse
        mock_update.return_value = (
            True, 
            "Deployment updated successfully",
            ModelDeploymentResponse(
                id=sample_deployment.id,
                model_id=sample_deployment.model_id,
                name=sample_deployment.name,
                status=DeploymentStatus.STOPPED,
                endpoint_url=sample_deployment.deployment_url,
                service_name=sample_deployment.deployment_name,
                framework=sample_deployment.framework,
                endpoints=sample_deployment.endpoints,
                config=sample_deployment.configuration,
                created_at=sample_deployment.created_at,
                updated_at=sample_deployment.updated_at
            )
        )
        
        success, message = await deployment_service.stop_deployment(sample_deployment.id)
        
        assert success is True
        assert "updated" in message.lower() or "stopped" in message.lower()
        mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_stop_deployment_not_found(self, mock_get_session):
        """Test stopping non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message = await deployment_service.stop_deployment("nonexistent")
        
        assert success is False
        assert "not found" in message.lower()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_scale_deployment_success(self, mock_update, sample_deployment):
        """Test scaling a deployment"""
        from app.schemas.model import ModelDeploymentResponse
        mock_update.return_value = (
            True, 
            "Deployment updated successfully",
            ModelDeploymentResponse(
                id=sample_deployment.id,
                model_id=sample_deployment.model_id,
                name=sample_deployment.name,
                status=DeploymentStatus.ACTIVE,
                endpoint_url=sample_deployment.deployment_url,
                service_name=sample_deployment.deployment_name,
                framework=sample_deployment.framework,
                endpoints=sample_deployment.endpoints,
                config={"replicas": 5},
                created_at=sample_deployment.created_at,
                updated_at=sample_deployment.updated_at
            )
        )
        
        scale_data = {"replicas": 5}
        success, message = await deployment_service.scale_deployment(sample_deployment.id, scale_data)
        
        assert success is True
        assert "updated" in message.lower() or "scaled" in message.lower()
        mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_scale_deployment_not_found(self, mock_get_session):
        """Test scaling non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        scale_data = {"replicas": 3}
        success, message = await deployment_service.scale_deployment("nonexistent", scale_data)
        
        assert success is False
        assert "not found" in message.lower()


class TestDeploymentServiceTesting:
    """Test deployment testing functionality"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_test_deployment_success(self, mock_get_session, sample_deployment):
        """Test testing a deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        test_data = {"data": [1.0, 2.0, 3.0]}
        success, message, result = await deployment_service.test_deployment(
            sample_deployment.id, test_data
        )
        
        assert success is True
        assert "successful" in message.lower()
        assert result is not None
        assert result["deployment_id"] == sample_deployment.id
        assert "predictions" in result
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_test_deployment_not_found(self, mock_get_session):
        """Test testing non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        test_data = {"data": [1, 2, 3]}
        success, message, result = await deployment_service.test_deployment("nonexistent", test_data)
        
        assert success is False
        assert "not found" in message.lower()
        assert result is None
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_test_deployment_inactive(self, mock_get_session, sample_deployment):
        """Test testing inactive deployment"""
        sample_deployment.status = DeploymentStatus.STOPPED.value
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        test_data = {"data": [1, 2, 3]}
        success, message, result = await deployment_service.test_deployment(
            sample_deployment.id, test_data
        )
        
        assert success is False
        assert "not active" in message.lower()
        assert result is None


class TestDeploymentServiceMetrics:
    """Test deployment metrics functionality"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_get_deployment_metrics_success(self, mock_get_session, sample_deployment):
        """Test getting deployment metrics"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        metrics = await deployment_service.get_deployment_metrics(sample_deployment.id)
        
        assert metrics is not None
        assert metrics["deployment_id"] == sample_deployment.id
        assert "requests_count" in metrics
        assert "average_latency_ms" in metrics
        assert "error_rate" in metrics
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_get_deployment_metrics_not_found(self, mock_get_session):
        """Test getting metrics for non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        metrics = await deployment_service.get_deployment_metrics("nonexistent")
        
        assert metrics is None


class TestDeploymentServiceGet:
    """Test deployment retrieval functionality"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_get_deployment_success(self, mock_get_session, sample_deployment):
        """Test getting a deployment by ID"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        deployment = await deployment_service.get_deployment(sample_deployment.id)
        
        assert deployment is not None
        assert deployment.id == sample_deployment.id
        assert isinstance(deployment, ModelDeploymentResponse)
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_get_deployment_not_found(self, mock_get_session):
        """Test getting non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        deployment = await deployment_service.get_deployment("nonexistent")
        
        assert deployment is None


class TestDeploymentServiceDelete:
    """Test deployment deletion functionality"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.undeploy_service', new_callable=AsyncMock)
    async def test_delete_deployment_with_other_deployments(self, mock_undeploy, mock_get_session, 
                                                           sample_deployment, test_model):
        """Test deleting deployment when other deployments exist"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[sample_deployment, test_model])
        mock_result = MagicMock()
        mock_result.first.return_value = MagicMock()  # Other deployment exists
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message = await deployment_service.delete_deployment(sample_deployment.id)
        
        assert success is True
        # Model status should not change if other deployments exist
        mock_session.delete.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.undeploy_service', new_callable=AsyncMock)
    async def test_delete_deployment_last_one(self, mock_undeploy, mock_get_session,
                                             sample_deployment, test_model):
        """Test deleting the last deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[sample_deployment, test_model])
        mock_result = MagicMock()
        mock_result.first.return_value = None  # No other deployments
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success, message = await deployment_service.delete_deployment(sample_deployment.id)
        
        assert success is True
        # Model status should be updated to VALIDATED
        mock_session.delete.assert_called_once()


class TestDeploymentServiceStatus:
    """Test deployment status functionality"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    @patch('app.services.deployment_service.bentoml_service_manager.get_service_status', new_callable=AsyncMock)
    async def test_get_deployment_status_success(self, mock_get_status, mock_get_session, sample_deployment):
        """Test getting deployment status"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_get_status.return_value = {"status": "running", "replicas": 1}
        
        status = await deployment_service.get_deployment_status(sample_deployment.id)
        
        assert status is not None
        assert status["deployment_id"] == sample_deployment.id
        assert status["deployment_status"] == sample_deployment.status
        assert "service_status" in status
        assert "endpoint_url" in status
        mock_get_status.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_get_deployment_status_not_found(self, mock_get_session):
        """Test getting status for non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        status = await deployment_service.get_deployment_status("nonexistent")
        
        assert status is None


class TestDeploymentServicePrivateMethods:
    """Test private deployment service methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager.undeploy_service', new_callable=AsyncMock)
    async def test_stop_bento_server_success(self, mock_undeploy):
        """Test stopping BentoML server"""
        mock_undeploy.return_value = True
        
        result = await deployment_service._stop_bento_server("test_service")
        
        assert result is True
        mock_undeploy.assert_called_once_with("test_service")
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.bentoml_service_manager.undeploy_service', new_callable=AsyncMock)
    async def test_stop_bento_server_failure(self, mock_undeploy):
        """Test stopping BentoML server with error"""
        mock_undeploy.side_effect = Exception("Service not found")
        
        result = await deployment_service._stop_bento_server("test_service")
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_update_scaling_success(self, mock_get_session, sample_deployment):
        """Test updating scaling configuration"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await deployment_service._update_scaling(sample_deployment.id, 5)
        
        assert result is True
        assert sample_deployment.replicas == 5
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_update_scaling_not_found(self, mock_get_session):
        """Test updating scaling for non-existent deployment"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await deployment_service._update_scaling("nonexistent", 5)
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_update_scaling_error(self, mock_get_session, sample_deployment):
        """Test updating scaling with error"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_deployment)
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await deployment_service._update_scaling(sample_deployment.id, 5)
        
        assert result is False


class TestDeploymentServiceConfig:
    """Test deployment configuration methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_update_deployment_config_success(self, mock_update, sample_deployment):
        """Test updating deployment configuration"""
        from app.schemas.model import ModelDeploymentResponse
        new_config = {"replicas": 3, "resources": {"cpu": "2"}}
        mock_update.return_value = (
            True,
            "Configuration updated",
            ModelDeploymentResponse(
                id=sample_deployment.id,
                model_id=sample_deployment.model_id,
                name=sample_deployment.name,
                status=DeploymentStatus.ACTIVE,
                endpoint_url=sample_deployment.deployment_url,
                service_name=sample_deployment.deployment_name,
                framework=sample_deployment.framework,
                endpoints=sample_deployment.endpoints,
                config=new_config,
                created_at=sample_deployment.created_at,
                updated_at=sample_deployment.updated_at
            )
        )
        
        success, message, deployment = await deployment_service.update_deployment_config(
            sample_deployment.id, new_config
        )
        
        assert success is True
        assert deployment.config == new_config
        mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.update_deployment', new_callable=AsyncMock)
    async def test_update_deployment_config_error(self, mock_update):
        """Test updating deployment configuration with error"""
        mock_update.side_effect = Exception("Update failed")
        
        success, message, deployment = await deployment_service.update_deployment_config(
            "test_id", {"replicas": 3}
        )
        
        assert success is False
        assert deployment is None


class TestDeploymentServiceLogs:
    """Test deployment logs functionality"""
    
    @pytest.mark.asyncio
    async def test_get_deployment_logs_success(self, sample_deployment):
        """Test getting deployment logs"""
        logs = await deployment_service.get_deployment_logs(sample_deployment, lines=10)
        
        assert isinstance(logs, list)
        assert len(logs) > 0
        assert all(isinstance(log, str) for log in logs)
    
    @pytest.mark.asyncio
    async def test_get_deployment_logs_custom_lines(self, sample_deployment):
        """Test getting deployment logs with custom line count"""
        logs = await deployment_service.get_deployment_logs(sample_deployment, lines=2)
        
        assert isinstance(logs, list)
        assert len(logs) <= 2


class TestDeploymentServiceAlias:
    """Test deployment service alias methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.deployment_service.create_deployment', new_callable=AsyncMock)
    async def test_deploy_model_alias(self, mock_create):
        """Test deploy_model alias method"""
        from app.schemas.model import ModelDeploymentCreate, ModelDeploymentResponse
        deployment_data = ModelDeploymentCreate(
            name="test",
            model_id="model_123",
            config={"replicas": 1}
        )
        mock_create.return_value = (True, "Deployed", None)
        
        result = await deployment_service.deploy_model(deployment_data)
        
        assert result[0] is True
        mock_create.assert_called_once_with(deployment_data)


class TestDeploymentServiceErrorHandling:
    """Test error handling in deployment service"""
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_get_deployment_error(self, mock_get_session):
        """Test get_deployment with database error"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=Exception("Database error"))
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await deployment_service.get_deployment("test_id")
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('app.services.deployment_service.get_session')
    async def test_list_deployments_error(self, mock_get_session):
        """Test list_deployments with database error"""
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("Database error"))
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await deployment_service.list_deployments()
        
        assert result == []

