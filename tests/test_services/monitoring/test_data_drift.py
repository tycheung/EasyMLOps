"""
Comprehensive tests for data drift detection service
Tests data drift detection including schema changes and data quality
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.services.monitoring.drift.data_drift import DataDriftService
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import DriftType, DriftSeverity


@pytest.fixture
def data_drift_service():
    """Create data drift service instance"""
    return DataDriftService()


@pytest.fixture
def baseline_logs(test_model, test_session):
    """Create baseline prediction logs"""
    import uuid
    logs = []
    base_time = datetime.utcnow() - timedelta(days=7)
    
    for i in range(100):
        log = PredictionLogDB(
            id=f"baseline_log_{i}",
            model_id=test_model.id,
            request_id=str(uuid.uuid4()),  # Required field
            input_data={"feature1": float(i), "feature2": float(i * 2), "category": "A"},
            output_data={"prediction": i % 2},
            latency_ms=50.0,
            success=True,
            timestamp=base_time + timedelta(hours=i),
            api_endpoint="/api/v1/predict/test"  # Required field
        )
        logs.append(log)
        test_session.add(log)
    
    test_session.commit()
    return logs


@pytest.fixture
def current_logs(test_model, test_session):
    """Create current prediction logs"""
    import uuid
    logs = []
    base_time = datetime.utcnow() - timedelta(days=1)
    
    for i in range(100):
        log = PredictionLogDB(
            id=f"current_log_{i}",
            model_id=test_model.id,
            request_id=str(uuid.uuid4()),  # Required field
            input_data={"feature1": float(i), "feature2": float(i * 2), "category": "A"},
            output_data={"prediction": i % 2},
            latency_ms=50.0,
            success=True,
            timestamp=base_time + timedelta(hours=i),
            api_endpoint="/api/v1/predict/test"  # Required field
        )
        logs.append(log)
        test_session.add(log)
    
    test_session.commit()
    return logs


class TestDataDriftService:
    """Test data drift detection service methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.drift.data_drift.get_session')
    async def test_detect_data_drift_no_drift(self, mock_get_session, data_drift_service, 
                                              test_model, baseline_logs, current_logs):
        """Test detecting data drift when no drift exists"""
        mock_session = MagicMock()
        baseline_result = MagicMock()
        baseline_result.scalars.return_value.all.return_value = baseline_logs
        current_result = MagicMock()
        current_result.scalars.return_value.all.return_value = current_logs
        mock_session.execute = AsyncMock(side_effect=[baseline_result, current_result])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        baseline_start = datetime.utcnow() - timedelta(days=7)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        current_start = datetime.utcnow() - timedelta(days=1)
        current_end = datetime.utcnow()
        
        drift_result = await data_drift_service.detect_data_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end
        )
        
        assert drift_result is not None
        assert drift_result.model_id == test_model.id
        assert drift_result.drift_type == DriftType.DATA
        assert drift_result.detection_method == "schema_quality_analysis"
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.drift.data_drift.get_session')
    async def test_detect_data_drift_schema_changes(self, mock_get_session, data_drift_service, test_model):
        """Test detecting data drift with schema changes"""
        baseline_logs = []
        current_logs = []
        base_time = datetime.utcnow() - timedelta(days=7)
        
        import uuid
        # Baseline with feature1, feature2
        for i in range(50):
            log = PredictionLogDB(
                id=f"baseline_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data={"feature1": float(i), "feature2": float(i * 2)},
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=True,
                timestamp=base_time + timedelta(hours=i)
            )
            baseline_logs.append(log)
        
        # Current with feature1, feature2, feature3 (new field)
        current_base = datetime.utcnow() - timedelta(days=1)
        for i in range(50):
            log = PredictionLogDB(
                id=f"current_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data={"feature1": float(i), "feature2": float(i * 2), "feature3": float(i * 3)},
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=True,
                timestamp=current_base + timedelta(hours=i),
                api_endpoint="/api/v1/predict/test"  # Required field
            )
            current_logs.append(log)
        
        mock_session = MagicMock()
        baseline_result = MagicMock()
        baseline_result.scalars.return_value.all.return_value = baseline_logs
        current_result = MagicMock()
        current_result.scalars.return_value.all.return_value = current_logs
        mock_session.execute = AsyncMock(side_effect=[baseline_result, current_result])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        baseline_start = datetime.utcnow() - timedelta(days=7)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        current_start = datetime.utcnow() - timedelta(days=1)
        current_end = datetime.utcnow()
        
        drift_result = await data_drift_service.detect_data_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end
        )
        
        assert drift_result is not None
        assert drift_result.drift_detected is True
        assert drift_result.schema_changes is not None
        assert "new_fields" in drift_result.schema_changes
        assert "feature3" in drift_result.schema_changes["new_fields"]
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.drift.data_drift.get_session')
    async def test_detect_data_drift_missing_values(self, mock_get_session, data_drift_service, test_model):
        """Test detecting data drift with missing values"""
        baseline_logs = []
        current_logs = []
        base_time = datetime.utcnow() - timedelta(days=7)
        
        import uuid
        # Baseline with no missing values
        for i in range(50):
            log = PredictionLogDB(
                id=f"baseline_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data={"feature1": float(i), "feature2": float(i * 2)},
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=True,
                timestamp=base_time + timedelta(hours=i),
                api_endpoint="/api/v1/predict/test"  # Required field
            )
            baseline_logs.append(log)
        
        # Current with missing values
        current_base = datetime.utcnow() - timedelta(days=1)
        for i in range(50):
            input_data = {"feature1": float(i)}
            if i % 10 != 0:  # 10% missing feature2
                input_data["feature2"] = float(i * 2)
            log = PredictionLogDB(
                id=f"current_{i}",
                model_id=test_model.id,
                request_id=str(uuid.uuid4()),  # Required field
                input_data=input_data,
                output_data={"prediction": i % 2},
                latency_ms=50.0,
                success=True,
                timestamp=current_base + timedelta(hours=i),
                api_endpoint="/api/v1/predict/test"  # Required field
            )
            current_logs.append(log)
        
        mock_session = MagicMock()
        baseline_result = MagicMock()
        baseline_result.scalars.return_value.all.return_value = baseline_logs
        current_result = MagicMock()
        current_result.scalars.return_value.all.return_value = current_logs
        mock_session.execute = AsyncMock(side_effect=[baseline_result, current_result])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        baseline_start = datetime.utcnow() - timedelta(days=7)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        current_start = datetime.utcnow() - timedelta(days=1)
        current_end = datetime.utcnow()
        
        drift_result = await data_drift_service.detect_data_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end
        )
        
        assert drift_result is not None
        assert drift_result.drift_detected is True
        assert drift_result.data_quality_metrics is not None
        assert "missing_values" in drift_result.data_quality_metrics
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.drift.data_drift.get_session')
    async def test_detect_data_drift_no_baseline_data(self, mock_get_session, data_drift_service, test_model):
        """Test detecting data drift with no baseline data"""
        mock_session = MagicMock()
        baseline_result = MagicMock()
        baseline_result.scalars.return_value.all.return_value = []
        current_result = MagicMock()
        current_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(side_effect=[baseline_result, current_result])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        baseline_start = datetime.utcnow() - timedelta(days=7)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        current_start = datetime.utcnow() - timedelta(days=1)
        current_end = datetime.utcnow()
        
        drift_result = await data_drift_service.detect_data_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end
        )
        
        assert drift_result is not None
        assert drift_result.drift_detected is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.drift.data_drift.get_session')
    async def test_detect_data_drift_with_deployment(self, mock_get_session, data_drift_service, 
                                                     test_model, baseline_logs, current_logs):
        """Test detecting data drift with deployment filter"""
        mock_session = MagicMock()
        baseline_result = MagicMock()
        baseline_result.scalars.return_value.all.return_value = baseline_logs[:50]
        current_result = MagicMock()
        current_result.scalars.return_value.all.return_value = current_logs[:50]
        mock_session.execute = AsyncMock(side_effect=[baseline_result, current_result])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        baseline_start = datetime.utcnow() - timedelta(days=7)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        current_start = datetime.utcnow() - timedelta(days=1)
        current_end = datetime.utcnow()
        
        drift_result = await data_drift_service.detect_data_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end,
            deployment_id="deploy_123"
        )
        
        assert drift_result is not None
        assert drift_result.deployment_id == "deploy_123"

