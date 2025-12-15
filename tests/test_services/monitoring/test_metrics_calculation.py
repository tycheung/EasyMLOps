"""
Comprehensive tests for metrics calculation service
Tests performance and confidence metrics calculation
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.services.monitoring.performance.metrics_calculation import MetricsCalculationService
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import ModelPerformanceMetrics, ModelConfidenceMetrics


@pytest.fixture
def metrics_service():
    """Create metrics calculation service instance"""
    return MetricsCalculationService()


@pytest.fixture
def sample_prediction_logs(test_model, test_session):
    """Create sample prediction logs"""
    import uuid
    logs = []
    base_time = datetime.utcnow() - timedelta(hours=1)
    
    for i in range(100):
        log = PredictionLogDB(
            id=f"log_{i}",
            model_id=test_model.id,
            deployment_id=None,
            request_id=str(uuid.uuid4()),  # Required field
            input_data={"feature": i},
            output_data={"prediction": i % 2},
            latency_ms=50.0 + (i % 10),
            success=True,
            timestamp=base_time + timedelta(minutes=i),
            api_endpoint="/api/v1/predict/test",  # Required field
            confidence_score=0.8 + (i % 20) / 100.0 if i % 2 == 0 else None
        )
        logs.append(log)
        test_session.add(log)
    
    test_session.commit()
    return logs


class TestMetricsCalculationService:
    """Test metrics calculation service methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.performance.metrics_calculation.get_session')
    async def test_get_model_performance_metrics_with_data(self, mock_get_session, metrics_service, 
                                                          test_model, sample_prediction_logs):
        """Test calculating performance metrics with data"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_prediction_logs
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await metrics_service.get_model_performance_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics is not None
        assert metrics.model_id == test_model.id
        assert metrics.total_requests == 100
        assert metrics.successful_requests == 100
        assert metrics.failed_requests == 0
        assert metrics.avg_latency_ms > 0
        assert metrics.p50_latency_ms is not None
        assert metrics.p95_latency_ms is not None
        assert metrics.p99_latency_ms is not None
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.performance.metrics_calculation.get_session')
    async def test_get_model_performance_metrics_no_data(self, mock_get_session, metrics_service, test_model):
        """Test calculating performance metrics with no data"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await metrics_service.get_model_performance_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics is not None
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.avg_latency_ms == 0.0
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.performance.metrics_calculation.get_session')
    async def test_get_model_performance_metrics_with_deployment(self, mock_get_session, metrics_service,
                                                                 test_model, sample_prediction_logs):
        """Test calculating performance metrics with deployment filter"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_prediction_logs[:50]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await metrics_service.get_model_performance_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time,
            deployment_id="deploy_123"
        )
        
        assert metrics is not None
        assert metrics.deployment_id == "deploy_123"
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.performance.metrics_calculation.get_session')
    async def test_get_model_performance_metrics_with_failures(self, mock_get_session, metrics_service, test_model):
        """Test calculating performance metrics with failed requests"""
        logs = []
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        import uuid
        for i in range(20):
            log = PredictionLogDB(
                id=f"log_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data={"feature": i},
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=i % 5 != 0,  # 20% failure rate
                timestamp=base_time + timedelta(minutes=i),
                api_endpoint="/api/v1/predict/test"  # Required field
            )
            logs.append(log)
        
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = logs
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await metrics_service.get_model_performance_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics is not None
        assert metrics.total_requests == 20
        assert metrics.successful_requests == 16
        assert metrics.failed_requests == 4
        assert metrics.error_rate > 0
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.performance.metrics_calculation.get_session')
    async def test_calculate_confidence_metrics_with_data(self, mock_get_session, metrics_service, test_model):
        """Test calculating confidence metrics with data"""
        logs = []
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        import uuid
        for i in range(50):
            log = PredictionLogDB(
                id=f"log_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data={"feature": i},
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=True,
                timestamp=base_time + timedelta(minutes=i),
                api_endpoint="/api/v1/predict/test",  # Required field
                confidence_score=0.7 + (i % 30) / 100.0
            )
            logs.append(log)
        
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = logs
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await metrics_service.calculate_confidence_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics is not None
        assert metrics.model_id == test_model.id
        assert metrics.avg_confidence_score is not None
        assert metrics.avg_confidence_score > 0
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.performance.metrics_calculation.get_session')
    async def test_calculate_confidence_metrics_insufficient_data(self, mock_get_session, metrics_service, test_model):
        """Test calculating confidence metrics with insufficient data"""
        logs = []
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        import uuid
        for i in range(5):  # Less than 10 required
            log = PredictionLogDB(
                id=f"log_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data={"feature": i},
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=True,
                timestamp=base_time + timedelta(minutes=i),
                api_endpoint="/api/v1/predict/test",  # Required field
                confidence_score=0.8
            )
            logs.append(log)
        
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = logs
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await metrics_service.calculate_confidence_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics is not None
        assert metrics.total_predictions == 5
        assert metrics.avg_confidence_score is None  # Insufficient data
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.performance.metrics_calculation.get_session')
    async def test_calculate_confidence_metrics_low_confidence(self, mock_get_session, metrics_service, test_model):
        """Test calculating confidence metrics with low confidence threshold"""
        logs = []
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        import uuid
        for i in range(50):
            log = PredictionLogDB(
                id=f"log_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data={"feature": i},
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=True,
                timestamp=base_time + timedelta(minutes=i),
                api_endpoint="/api/v1/predict/test",  # Required field
                confidence_score=0.3 + (i % 20) / 100.0  # Low confidence scores
            )
            logs.append(log)
        
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = logs
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = await metrics_service.calculate_confidence_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time,
            low_confidence_threshold=0.5
        )
        
        assert metrics is not None
        assert metrics.low_confidence_count > 0
        assert metrics.low_confidence_percentage > 0

