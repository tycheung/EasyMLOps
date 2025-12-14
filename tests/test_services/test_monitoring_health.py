"""
Tests for System Health Service
Tests system health monitoring and health metrics collection
"""

import pytest
from unittest.mock import patch
from datetime import datetime, timedelta

from app.services.monitoring_service import monitoring_service


class TestSystemHealthService:
    """Test system health service functionality"""
    
    @pytest.mark.asyncio
    async def test_get_system_health(self):
        """Test system health monitoring"""
        # Test without mocking psutil (it's optional)
        health = await monitoring_service.get_system_health()
        
        assert health is not None
        assert isinstance(health, dict)
        # Health dict should have some basic keys
        assert 'status' in health or 'overall_status' in health or 'cpu_usage' in health
    
    @pytest.mark.asyncio
    async def test_get_system_health_status(self):
        """Test getting system health status"""
        status = await monitoring_service.get_system_health_status()
        
        assert status is not None
        assert hasattr(status, 'overall_status')
        assert hasattr(status, 'components')
    
    @pytest.mark.asyncio
    async def test_collect_system_health_metrics(self):
        """Test collecting system health metrics"""
        metrics = await monitoring_service.collect_system_health_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, list)
        # Should have at least CPU and memory metrics
        assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_store_health_metric(self):
        """Test storing a health metric"""
        from app.schemas.monitoring import SystemHealthMetric, SystemComponent, MetricType, SystemStatus
        
        metric = SystemHealthMetric(
            component=SystemComponent.API_SERVER,
            status=SystemStatus.OPERATIONAL,
            message="Test metric",
            metric_type=MetricType.CPU_USAGE,
            value=50.0,
            unit="percent"
        )
        
        metric_id = await monitoring_service.store_health_metric(metric)
        assert metric_id is not None
        assert isinstance(metric_id, str)
    
    @pytest.mark.asyncio
    async def test_check_system_health(self):
        """Test checking system health"""
        health = await monitoring_service.check_system_health()
        
        assert health is not None
        assert isinstance(health, dict)
        assert 'status' in health or 'overall_status' in health
    
    @pytest.mark.asyncio
    async def test_collect_model_resource_usage(self, test_model):
        """Test collecting model resource usage"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=5)
        end_time = now
        
        resource_usage = await monitoring_service.collect_model_resource_usage(
            model_id=test_model.id,
            time_window_start=start_time,
            time_window_end=end_time
        )
        
        assert resource_usage is not None
        assert resource_usage.model_id == test_model.id
        assert resource_usage.time_window_start == start_time
        assert resource_usage.time_window_end == end_time
    
    @pytest.mark.asyncio
    async def test_store_model_resource_usage(self, test_model):
        """Test storing model resource usage"""
        from app.schemas.monitoring import ModelResourceUsage
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        resource_usage = ModelResourceUsage(
            model_id=test_model.id,
            time_window_start=now - timedelta(minutes=5),
            time_window_end=now,
            cpu_usage_percent=45.0,
            memory_usage_mb=512.0
        )
        
        usage_id = await monitoring_service.store_model_resource_usage(resource_usage)
        assert usage_id is not None
        assert isinstance(usage_id, str)
    
    @pytest.mark.asyncio
    async def test_check_bentoml_system_health(self):
        """Test checking BentoML system health"""
        is_healthy, message = await monitoring_service.check_bentoml_system_health()
        
        assert isinstance(is_healthy, bool)
        assert isinstance(message, str)

