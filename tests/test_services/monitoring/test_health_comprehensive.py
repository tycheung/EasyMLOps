"""
Comprehensive tests for System Health Service
Tests additional methods and edge cases to increase coverage
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.services.monitoring.health import SystemHealthService
from app.schemas.monitoring import SystemHealthMetric, SystemComponent, MetricType, SystemStatus


@pytest.fixture
def health_service():
    """Create system health service instance"""
    return SystemHealthService()


class TestSystemHealthServiceMethods:
    """Test additional health service methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.PSUTIL_AVAILABLE', True)
    @patch('asyncio.to_thread')
    async def test_collect_system_health_metrics_with_psutil(self, mock_to_thread, health_service):
        """Test collecting system health metrics with psutil available"""
        # Mock psutil functions
        mock_uname = MagicMock()
        mock_uname.node = "test-node"
        mock_to_thread.side_effect = [
            mock_uname,  # uname
            45.5,  # cpu_percent
            MagicMock(percent=65.0),  # virtual_memory
            MagicMock(percent=75.0),  # disk_usage('/')
            MagicMock(percent=80.0),  # disk_usage(models_dir)
            MagicMock(bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20),  # net_io_counters
        ]
        
        with patch('app.services.monitoring.health.psutil') as mock_psutil:
            mock_psutil.uname.return_value = mock_uname
            
            metrics = await health_service.collect_system_health_metrics()
            
            assert len(metrics) > 0
            assert any(m.component == SystemComponent.API_SERVER for m in metrics)
            assert any(m.component == SystemComponent.DATABASE for m in metrics)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.PSUTIL_AVAILABLE', False)
    async def test_collect_system_health_metrics_without_psutil(self, health_service):
        """Test collecting system health metrics without psutil"""
        metrics = await health_service.collect_system_health_metrics()
        
        assert len(metrics) > 0
        # Should still have database health metric
        assert any(m.component == SystemComponent.DATABASE for m in metrics)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_store_health_metric(self, mock_get_session, health_service):
        """Test storing a health metric"""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        metric = SystemHealthMetric(
            component=SystemComponent.API_SERVER,
            status=SystemStatus.OPERATIONAL,
            message="Test metric",
            metric_type=MetricType.CPU_USAGE,
            value=50.0,
            unit="percent"
        )
        
        metric_id = await health_service.store_health_metric(metric)
        
        assert metric_id is not None
        assert isinstance(metric_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_store_health_metric_with_all_fields(self, mock_get_session, health_service):
        """Test storing health metric with all fields"""
        metric = SystemHealthMetric(
            component=SystemComponent.API_SERVER,
            status=SystemStatus.OPERATIONAL,
            message="Test metric with all fields",
            metric_type=MetricType.CPU_USAGE,
            value=50.0,
            unit="percent",
            host="test-host",
            mount_point="/",
            threshold_warning=75.0,
            threshold_critical=90.0
        )
        
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        metric_id = await health_service.store_health_metric(metric)
        
        assert metric_id is not None
        assert isinstance(metric_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_store_model_resource_usage(self, mock_get_session, health_service, test_model):
        """Test storing model resource usage"""
        from app.schemas.monitoring import ModelResourceUsage
        
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        resource_usage = ModelResourceUsage(
            model_id=test_model.id,
            time_window_start=datetime.utcnow() - timedelta(minutes=5),
            time_window_end=datetime.utcnow(),
            cpu_usage_percent=45.0,
            memory_usage_mb=512.0
        )
        
        usage_id = await health_service.store_model_resource_usage(resource_usage)
        
        assert usage_id is not None
        assert isinstance(usage_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_store_model_resource_usage_with_all_fields(self, mock_get_session, health_service, test_model):
        """Test storing model resource usage with all fields"""
        from app.schemas.monitoring import ModelResourceUsage
        
        resource_usage = ModelResourceUsage(
            model_id=test_model.id,
            time_window_start=datetime.utcnow() - timedelta(minutes=5),
            time_window_end=datetime.utcnow(),
            cpu_usage_percent=45.0,
            memory_usage_mb=512.0,
            gpu_usage_percent=30.0,
            disk_io_read_mb=100.0,
            disk_io_write_mb=50.0,
            network_io_sent_mb=200.0,
            network_io_received_mb=150.0
        )
        
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        usage_id = await health_service.store_model_resource_usage(resource_usage)
        
        assert usage_id is not None
        assert isinstance(usage_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_get_system_health(self, mock_get_session, health_service):
        """Test getting system health summary"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        health = await health_service.get_system_health()
        
        assert health is not None
        assert isinstance(health, dict)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_get_system_health_status(self, mock_get_session, health_service):
        """Test getting system health status"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        status = await health_service.get_system_health_status()
        
        assert status is not None
        assert hasattr(status, 'overall_status')
        assert hasattr(status, 'components')
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_check_system_health(self, mock_get_session, health_service):
        """Test checking system health"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        health = await health_service.check_system_health()
        
        assert health is not None
        assert isinstance(health, dict)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_check_bentoml_system_health(self, mock_get_session, health_service):
        """Test checking BentoML system health"""
        mock_session = MagicMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        is_healthy, message = await health_service.check_bentoml_system_health()
        
        assert isinstance(is_healthy, bool)
        assert isinstance(message, str)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.health.get_session')
    async def test_collect_model_resource_usage(self, mock_get_session, health_service, test_model):
        """Test collecting model resource usage"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        resource_usage = await health_service.collect_model_resource_usage(
            model_id=test_model.id,
            time_window_start=datetime.utcnow() - timedelta(minutes=5),
            time_window_end=datetime.utcnow()
        )
        
        assert resource_usage is not None
        assert resource_usage.model_id == test_model.id

