"""
Comprehensive tests for Canary Deployment Service
Tests additional methods and edge cases to increase coverage
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import uuid

from app.services.monitoring.canary import CanaryDeploymentService
from app.models.monitoring import CanaryDeploymentDB
from app.schemas.monitoring import CanaryDeploymentStatus, CanaryMetrics


@pytest.fixture
def canary_service():
    """Create canary deployment service instance"""
    return CanaryDeploymentService()


@pytest.fixture
def sample_canary(test_model, test_session):
    """Create a sample canary deployment in database"""
    canary_db = CanaryDeploymentDB(
        id=str(uuid.uuid4()),
        deployment_name="test_canary",
        model_id=test_model.id,
        production_deployment_id="prod_deploy_123",
        canary_deployment_id="canary_deploy_456",
        current_traffic_percentage=10.0,
        target_traffic_percentage=100.0,
        rollout_step_size=10.0,
        rollout_step_duration_minutes=60,
        max_error_rate_threshold=5.0,
        max_latency_increase_pct=50.0,
        status="rolling_out",
        current_step=1,
        total_steps=10,
        started_at=datetime.utcnow() - timedelta(minutes=30),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    test_session.add(canary_db)
    test_session.commit()
    test_session.refresh(canary_db)
    return canary_db


class TestCanaryServiceMethods:
    """Test additional canary service methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_store_canary_deployment_with_all_fields(self, mock_get_session, canary_service, test_model):
        """Test storing canary deployment with all fields"""
        from app.schemas.monitoring import CanaryDeployment
        
        canary = CanaryDeployment(
            deployment_name="Full Canary Test",
            model_id=test_model.id,
            production_deployment_id="prod_123",
            canary_deployment_id="canary_456",
            current_traffic_percentage=0.0,
            target_traffic_percentage=100.0,
            rollout_step_size=10.0,
            rollout_step_duration_minutes=60,
            max_error_rate_threshold=5.0,
            max_latency_increase_pct=50.0,
            min_health_check_duration_minutes=5,
            health_check_window_minutes=15,
            status=CanaryDeploymentStatus.PENDING,
            current_step=0,
            total_steps=10,
            created_by="admin",
            config={"custom": "value"}
        )
        
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        canary_id = await canary_service.store_canary_deployment(canary)
        
        assert canary_id is not None
        assert isinstance(canary_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_start_canary_rollout_not_found(self, mock_get_session, canary_service):
        """Test starting non-existent canary rollout"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        with pytest.raises(ValueError, match="not found"):
            await canary_service.start_canary_rollout("nonexistent")
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_start_canary_rollout_invalid_status(self, mock_get_session, canary_service, sample_canary):
        """Test starting canary rollout with invalid status"""
        sample_canary.status = "completed"
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        with pytest.raises(ValueError, match="Cannot start canary"):
            await canary_service.start_canary_rollout(sample_canary.id)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    @patch('app.services.monitoring.canary.CanaryDeploymentService.calculate_canary_metrics', new_callable=AsyncMock)
    async def test_check_canary_health_unhealthy(self, mock_calc_metrics, mock_get_session, 
                                                  canary_service, sample_canary):
        """Test checking canary health when unhealthy"""
        sample_canary.min_health_check_duration_minutes = 1
        sample_canary.started_at = datetime.utcnow() - timedelta(minutes=10)
        
        mock_metrics = CanaryMetrics(
            canary_deployment_id=sample_canary.id,
            time_window_start=datetime.utcnow() - timedelta(minutes=15),
            time_window_end=datetime.utcnow(),
            canary_error_rate=10.0,  # Exceeds threshold
            production_error_rate=2.0,
            canary_avg_latency_ms=100.0,
            production_avg_latency_ms=50.0,
            health_status="unhealthy",
            health_check_passed=False
        )
        mock_calc_metrics.return_value = mock_metrics
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        is_healthy, health_status, reason = await canary_service.check_canary_health(sample_canary.id)
        
        assert is_healthy is False
        assert health_status == "unhealthy"
        assert reason is not None
        assert "Error rate" in reason
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    @patch('app.services.monitoring.canary.CanaryDeploymentService.calculate_canary_metrics', new_callable=AsyncMock)
    async def test_check_canary_health_degraded(self, mock_calc_metrics, mock_get_session,
                                                canary_service, sample_canary):
        """Test checking canary health when degraded"""
        sample_canary.min_health_check_duration_minutes = 1
        sample_canary.started_at = datetime.utcnow() - timedelta(minutes=10)
        sample_canary.max_latency_increase_pct = 50.0
        
        mock_metrics = CanaryMetrics(
            canary_deployment_id=sample_canary.id,
            time_window_start=datetime.utcnow() - timedelta(minutes=15),
            time_window_end=datetime.utcnow(),
            canary_error_rate=2.0,
            production_error_rate=1.0,
            canary_avg_latency_ms=100.0,
            production_avg_latency_ms=50.0,
            latency_increase_pct=100.0,  # Exceeds threshold
            health_status="degraded",
            health_check_passed=False
        )
        mock_calc_metrics.return_value = mock_metrics
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        is_healthy, health_status, reason = await canary_service.check_canary_health(sample_canary.id)
        
        assert is_healthy is False
        assert health_status == "degraded"
        assert reason is not None
        assert "Latency increase" in reason
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_advance_canary_rollout_not_rolling_out(self, mock_get_session, canary_service, sample_canary):
        """Test advancing canary rollout when not in rolling_out status"""
        sample_canary.status = "pending"
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await canary_service.advance_canary_rollout(sample_canary.id)
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_advance_canary_rollout_not_time_yet(self, mock_get_session, canary_service, sample_canary):
        """Test advancing canary rollout when not time for next step"""
        sample_canary.next_step_time = datetime.utcnow() + timedelta(minutes=30)
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await canary_service.advance_canary_rollout(sample_canary.id)
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    @patch('app.services.monitoring.canary.CanaryDeploymentService.check_canary_health', new_callable=AsyncMock)
    async def test_advance_canary_rollout_completes(self, mock_check_health, mock_get_session,
                                                    canary_service, sample_canary):
        """Test advancing canary rollout to completion"""
        sample_canary.current_traffic_percentage = 90.0
        sample_canary.target_traffic_percentage = 100.0
        sample_canary.rollout_step_size = 10.0
        sample_canary.next_step_time = datetime.utcnow() - timedelta(minutes=1)
        
        mock_check_health.return_value = (True, "healthy", None)
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await canary_service.advance_canary_rollout(sample_canary.id)
        
        assert result is True
        assert sample_canary.status == "completed"
        assert sample_canary.current_traffic_percentage == 100.0
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_rollback_canary_not_found(self, mock_get_session, canary_service):
        """Test rolling back non-existent canary"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        with pytest.raises(ValueError, match="not found"):
            await canary_service.rollback_canary("nonexistent", "test reason")
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_store_canary_metrics(self, mock_get_session, canary_service, sample_canary):
        """Test storing canary metrics"""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        metrics = CanaryMetrics(
            canary_deployment_id=sample_canary.id,
            time_window_start=datetime.utcnow() - timedelta(hours=1),
            time_window_end=datetime.utcnow(),
            canary_total_requests=100,
            canary_successful_requests=95,
            canary_failed_requests=5,
            production_total_requests=500,
            production_successful_requests=490,
            production_failed_requests=10,
            canary_avg_latency_ms=50.0,
            production_avg_latency_ms=45.0,
            health_status="healthy",
            health_check_passed=True
        )
        
        metrics_id = await canary_service.store_canary_metrics(metrics)
        
        assert metrics_id is not None
        assert isinstance(metrics_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    @patch('app.services.monitoring.canary.CanaryDeploymentService.calculate_canary_metrics', new_callable=AsyncMock)
    async def test_check_canary_health_too_early(self, mock_calc_metrics, mock_get_session,
                                                 canary_service, sample_canary):
        """Test checking canary health before minimum duration"""
        sample_canary.min_health_check_duration_minutes = 60
        sample_canary.started_at = datetime.utcnow() - timedelta(minutes=5)  # Too early
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        is_healthy, health_status, reason = await canary_service.check_canary_health(sample_canary.id)
        
        assert is_healthy is True
        assert health_status == "healthy"
        assert reason is None
        mock_calc_metrics.assert_not_called()  # Should not calculate metrics yet
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.canary.get_session')
    async def test_check_canary_health_not_rolling_out(self, mock_get_session, canary_service, sample_canary):
        """Test checking canary health when not rolling out"""
        sample_canary.status = "completed"
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_canary)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        is_healthy, health_status, reason = await canary_service.check_canary_health(sample_canary.id)
        
        assert is_healthy is True
        assert health_status == "healthy"
        assert reason is None

