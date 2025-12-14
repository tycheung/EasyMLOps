"""
Basic monitoring service tests
Tests prediction logging, metrics calculation, health checks, and alerts
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from app.services.monitoring_service import MonitoringService
from app.schemas.monitoring import AlertSeverity, SystemComponent


class TestMonitoringService:
    """Test MonitoringService functionality"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service instance"""
        return MonitoringService()
    
    @pytest.mark.asyncio
    async def test_log_prediction(self, monitoring_service, test_session, test_model):
        """Test prediction logging"""
        input_data = {"feature1": 0.5, "feature2": "test"}
        output_data = {"class": "A", "probability": 0.85}
        latency_ms = 150.5
        
        await monitoring_service.log_prediction(
            model_id=test_model.id,
            deployment_id=None,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            api_endpoint="/test/predict",
            success=True
        )
        
        # Verify log was created
        from app.models.monitoring import PredictionLogDB
        logs = test_session.query(PredictionLogDB).filter(
            PredictionLogDB.model_id == test_model.id
        ).all()
        
        assert len(logs) == 1
        log = logs[0]
        assert log.input_data == input_data
        assert log.output_data == output_data
        assert log.latency_ms == latency_ms
        assert log.success == True
    
    @pytest.mark.asyncio
    async def test_calculate_metrics(self, monitoring_service, test_session, test_model):
        """Test metrics calculation"""
        from app.models.monitoring import PredictionLogDB
        import uuid
        
        logs = [
            PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=test_model.id,
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                input_data={"test": "data"},
                output_data={"result": "A"},
                latency_ms=100.0,
                api_endpoint="/predict",
                success=True
            ),
            PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=test_model.id,
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                input_data={"test": "data"},
                output_data={"result": "B"},
                latency_ms=200.0,
                api_endpoint="/predict",
                success=True
            ),
            PredictionLogDB(
                id=str(uuid.uuid4()),
                model_id=test_model.id,
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                input_data={"test": "data"},
                output_data={"result": "error"},
                latency_ms=50.0,
                api_endpoint="/predict",
                success=False
            )
        ]
        
        for log in logs:
            test_session.add(log)
        test_session.commit()
        
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        metrics = await monitoring_service.get_model_performance_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.avg_latency_ms == (100.0 + 200.0 + 50.0) / 3
        assert metrics.success_rate == (2 / 3) * 100
        assert metrics.error_rate == (1 / 3) * 100
    
    @pytest.mark.asyncio
    async def test_check_system_health(self, monitoring_service):
        """Test system health checking"""
        with patch('psutil.cpu_percent', return_value=45.2):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.5
                
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk.return_value.percent = 30.0
                    
                    health = await monitoring_service.check_system_health()
        
        assert "cpu_usage" in health
        assert "memory_usage" in health
        assert "disk_usage" in health
        assert health["cpu_usage"] == 45.2
        assert health["memory_usage"] == 60.5
        assert health["disk_usage"] == 30.0
    
    @pytest.mark.asyncio
    async def test_create_alert(self, monitoring_service, test_session):
        """Test alert creation"""
        from app.models.monitoring import AlertDB
        test_session.query(AlertDB).delete()
        test_session.commit()
        
        created_alert = await monitoring_service.create_alert(
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            title="High CPU Usage",
            description="CPU usage exceeded 80%",
            metric_value=85.0
        )
        
        assert created_alert.severity == AlertSeverity.WARNING
        assert created_alert.component == SystemComponent.API_SERVER
        assert created_alert.title == "High CPU Usage"
        assert created_alert.description == "CPU usage exceeded 80%"
        assert created_alert.metric_value == 85.0
        assert created_alert.is_active is True

        db_alerts = test_session.query(AlertDB).all()
        
        assert len(db_alerts) == 1
        db_alert = db_alerts[0]
        assert db_alert.severity == "warning"
        assert db_alert.component == "api_server"
        assert db_alert.title == "High CPU Usage"
        assert db_alert.description == "CPU usage exceeded 80%"
        assert db_alert.metric_value == 85.0


class TestServiceIntegration:
    """Integration tests for service interactions (monitoring-related)"""
    
    @pytest.mark.asyncio
    async def test_monitoring_deployment_integration(self, monitoring_service, deployment_service):
        """Test integration between monitoring and deployment services"""
        from app.schemas.model import ModelDeploymentCreate
        
        deployment_data = ModelDeploymentCreate(
            model_id="test_model_123",
            name="test_deployment",
            description="Integration test deployment"
        )
        
        with patch.object(deployment_service, 'create_deployment') as mock_create:
            with patch.object(monitoring_service, 'log_prediction') as mock_log:
                mock_create.return_value = (True, "Success", None)
                mock_log.return_value = "prediction_123"
                
                success, message, deployment = await deployment_service.create_deployment(deployment_data)
                assert success is True
                
                prediction_id = await monitoring_service.log_prediction(
                    model_id="test_model_123",
                    deployment_id="deploy_123",
                    input_data={"feature1": 0.5},
                    output_data={"class": "A", "probability": 0.85},
                    latency_ms=45.2,
                    api_endpoint="/test/predict",
                    success=True
                )
                
                assert prediction_id == "prediction_123"


class TestServiceErrorHandling:
    """Test service error handling scenarios (monitoring-related)"""
    
    @pytest.mark.asyncio
    async def test_monitoring_service_db_error(self):
        """Test monitoring service with database errors"""
        monitoring_service = MonitoringService()
        
        mock_session = MagicMock()
        mock_session.add.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            await monitoring_service.log_prediction(
                mock_session, 1, {}, {}, 100.0, "success"
            )

