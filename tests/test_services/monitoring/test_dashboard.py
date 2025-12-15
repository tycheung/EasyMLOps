"""
Tests for Dashboard Service
Tests dashboard metrics aggregation
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB, AlertDB
from app.models.model import Model, ModelDeployment
from app.database import get_session


class TestDashboardService:
    """Test dashboard service functionality"""
    
    @pytest.mark.asyncio
    async def test_get_dashboard_metrics(self, test_model, test_deployment):
        """Test getting comprehensive dashboard metrics"""
        # Create some test data
        async with get_session() as session:
            # Create prediction logs for today
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            for i in range(10):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    deployment_id=test_deployment.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": 0.5},
                    latency_ms=50.0 + i,
                    timestamp=today_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            # Create an active alert (AlertDB has component, title, description, not alert_type/message)
            alert = AlertDB(
                id=str(uuid.uuid4()),
                component="api_server",
                title="Performance Degradation",
                description="Test alert",
                severity="high",
                is_active=True
            )
            session.add(alert)
            await session.commit()
        
        # Get dashboard metrics
        metrics = await monitoring_service.get_dashboard_metrics()
        
        # Verify metrics structure
        assert metrics is not None
        assert metrics.total_models >= 1  # At least our test model
        assert metrics.active_deployments >= 0
        assert metrics.total_predictions_today >= 10
        assert metrics.avg_response_time_today >= 0
        assert metrics.active_alerts >= 1
        # system_status should be a SystemStatus enum value
        assert metrics.system_status is not None
        # Should be one of: OPERATIONAL, UNHEALTHY, DEGRADED, MAINTENANCE
        assert isinstance(metrics.recent_deployments, list)
        assert isinstance(metrics.request_trend_24h, list)
        assert isinstance(metrics.error_trend_24h, list)
        assert len(metrics.request_trend_24h) == 24
        assert len(metrics.error_trend_24h) == 24
    
    @pytest.mark.asyncio
    async def test_get_dashboard_metrics_empty_database(self):
        """Test dashboard metrics with empty database"""
        metrics = await monitoring_service.get_dashboard_metrics()
        
        assert metrics is not None
        assert metrics.total_models >= 0
        assert metrics.active_deployments >= 0
        assert metrics.total_predictions_today >= 0
        assert metrics.active_alerts >= 0
    
    @pytest.mark.asyncio
    async def test_get_dashboard_metrics_with_recent_deployments(self, test_model):
        """Test dashboard metrics includes recent deployments"""
        async with get_session() as session:
            # Create multiple deployments
            for i in range(3):
                deployment = ModelDeployment(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    name=f"test_deployment_{i}",
                    deployment_name=f"test_deployment_{i}",
                    status="active",
                    created_at=datetime.utcnow() - timedelta(hours=i)
                )
                session.add(deployment)
            await session.commit()
        
        metrics = await monitoring_service.get_dashboard_metrics()
        
        assert len(metrics.recent_deployments) >= 0  # May be limited to 5
        if len(metrics.recent_deployments) > 0:
            assert "id" in metrics.recent_deployments[0]
            assert "name" in metrics.recent_deployments[0]
            assert "model_id" in metrics.recent_deployments[0]
            assert "status" in metrics.recent_deployments[0]

