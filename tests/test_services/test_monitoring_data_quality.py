"""
Tests for Data Quality Service
Tests outlier detection, anomaly detection, and data quality metrics calculation
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import (
    OutlierDetectionMethod, OutlierType, AnomalyType
)
from app.database import get_session


class TestDataQualityService:
    """Test data quality service functionality"""
    
    @pytest.mark.asyncio
    async def test_detect_outliers_z_score(self, test_model):
        """Test outlier detection using Z-score method"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        
        # Create historical logs with normal values
        async with get_session() as session:
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": float(i), "feature2": float(i * 2)},  # Normal range
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Test with normal input (should not be outlier)
        normal_input = {"feature1": 50.0, "feature2": 100.0}
        detection = await monitoring_service.detect_outliers(
            model_id=test_model.id,
            input_data=normal_input,
            detection_method=OutlierDetectionMethod.Z_SCORE,
            outlier_type=OutlierType.INPUT
        )
        
        assert detection is not None
        assert detection.model_id == test_model.id
        assert detection.detection_method == OutlierDetectionMethod.Z_SCORE
        assert detection.outlier_type == OutlierType.INPUT
        
        # Test with outlier input (should be outlier)
        outlier_input = {"feature1": 1000.0, "feature2": 2000.0}  # Far from normal range
        detection2 = await monitoring_service.detect_outliers(
            model_id=test_model.id,
            input_data=outlier_input,
            detection_method=OutlierDetectionMethod.Z_SCORE,
            outlier_type=OutlierType.INPUT,
            z_score_threshold=3.0
        )
        
        assert detection2 is not None
        # Should detect as outlier if threshold is exceeded
        assert detection2.outlier_score is not None
        assert 0 <= detection2.outlier_score <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_data_quality_metrics(self, test_model):
        """Test calculating data quality metrics"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create logs with various quality levels
        async with get_session() as session:
            # Valid logs
            for i in range(80):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": float(i), "feature2": float(i * 2)},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            
            # Logs with missing values
            for i in range(10):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": None, "feature2": float(i * 2)},  # Missing feature1
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i + 80),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            
            # Invalid logs (non-dict input)
            for i in range(10):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data="invalid",  # Invalid format
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i + 90),
                    api_endpoint="/models/predict",
                    success=False
                )
                session.add(log)
            
            await session.commit()
        
        # Calculate data quality metrics
        metrics = await monitoring_service.calculate_data_quality_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics is not None
        assert metrics.model_id == test_model.id
        assert metrics.total_samples == 100
        assert metrics.valid_samples == 80
        assert metrics.invalid_samples == 20
        assert metrics.missing_value_count >= 10
        assert metrics.completeness_score is not None
        assert 0 <= metrics.completeness_score <= 1
        assert metrics.validity_score is not None
        assert 0 <= metrics.validity_score <= 1
        assert metrics.overall_quality_score is not None
    
    @pytest.mark.asyncio
    async def test_detect_anomaly(self, test_model):
        """Test anomaly detection"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        
        # Create baseline logs
        async with get_session() as session:
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": float(i), "feature2": float(i * 2)},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Detect anomaly with unusual pattern
        anomaly_input = {"feature1": 1000.0, "feature2": -500.0}  # Unusual pattern
        anomaly = await monitoring_service.detect_anomaly(
            model_id=test_model.id,
            input_data=anomaly_input,
            anomaly_type=AnomalyType.INPUT,
            detection_method="statistical"
        )
        
        assert anomaly is not None
        assert anomaly.model_id == test_model.id
        assert anomaly.anomaly_type == AnomalyType.INPUT
        assert anomaly.anomaly_score is not None
        assert 0 <= anomaly.anomaly_score <= 1
        assert anomaly.is_anomaly is not None

