"""
Tests for Drift Detection Service
Tests feature drift, prediction drift, and data drift detection
"""

import pytest
from unittest.mock import patch
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import DriftType
from app.database import get_session


class TestDriftDetectionService:
    """Test drift detection service functionality"""
    
    @pytest.mark.asyncio
    async def test_feature_drift_detection(self, test_model):
        """Test feature drift detection using KS test and PSI"""
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=7)
        baseline_end = now - timedelta(days=6)
        current_start = now - timedelta(hours=1)
        current_end = now
        
        async with get_session() as session:
            # Clean up existing logs
            from sqlalchemy import delete
            delete_stmt = delete(PredictionLogDB).where(
                PredictionLogDB.model_id == test_model.id
            )
            await session.execute(delete_stmt)
            await session.commit()
            
            # Create baseline logs with normal distribution
            baseline_logs = []
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={
                        "feature1": 50.0 + (i % 20),  # 50-70 range
                        "feature2": 100.0 + (i % 30),  # 100-130 range
                        "feature3": 0.5 + (i % 10) * 0.1  # 0.5-1.4 range
                    },
                    output_data={"prediction": 1.0},
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                baseline_logs.append(log)
                session.add(log)
            
            # Create current logs with shifted distribution (drift)
            current_logs = []
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={
                        "feature1": 80.0 + (i % 20),  # 80-100 range (shifted!)
                        "feature2": 100.0 + (i % 30),  # Same as baseline
                        "feature3": 0.5 + (i % 10) * 0.1  # Same as baseline
                    },
                    output_data={"prediction": 1.0},
                    latency_ms=50.0,
                    timestamp=current_start + timedelta(minutes=i),
                    api_endpoint="/predict",
                    success=True
                )
                current_logs.append(log)
                session.add(log)
            
            await session.commit()
            
            # Detect feature drift
            drift_result = await monitoring_service.detect_feature_drift(
                model_id=test_model.id,
                baseline_window_start=baseline_start,
                baseline_window_end=baseline_end,
                current_window_start=current_start,
                current_window_end=current_end
            )
        
        # Verify drift detection
        assert drift_result.model_id == test_model.id
        assert drift_result.drift_type == DriftType.FEATURE
        assert drift_result.detection_method == "ks_test_psi"
        assert drift_result.drift_detected is True  # Should detect drift in feature1
        assert drift_result.drift_score > 0
        assert drift_result.drift_severity is not None
        assert drift_result.feature_drift_scores is not None
        assert "feature1" in drift_result.feature_drift_scores
        # feature1 should have high drift score (shifted distribution)
        assert drift_result.feature_drift_scores["feature1"] > 0.2
        # feature2 and feature3 should have lower drift scores
        assert drift_result.feature_drift_details is not None
        assert "feature1" in drift_result.feature_drift_details
        assert drift_result.p_value is not None
        assert drift_result.p_value < 0.05  # Should be statistically significant
    
    @pytest.mark.asyncio
    async def test_prediction_drift_detection(self, test_model):
        """Test prediction drift detection"""
        from app.models.monitoring import PredictionLogDB
        from datetime import datetime, timedelta
        import uuid
        
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=7)
        baseline_end = now - timedelta(days=6)
        current_start = now - timedelta(hours=1)
        current_end = now
        
        async with get_session() as session:
            # Clean up existing logs
            from sqlalchemy import delete
            delete_stmt = delete(PredictionLogDB).where(
                PredictionLogDB.model_id == test_model.id
            )
            await session.execute(delete_stmt)
            await session.commit()
            
            # Create baseline logs with predictions centered around 0.5
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": 0.5 + (i % 10) * 0.01},  # 0.5-0.59 range
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            # Create current logs with shifted predictions (drift)
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": 0.7 + (i % 10) * 0.01},  # 0.7-0.79 range (shifted!)
                    latency_ms=50.0,
                    timestamp=current_start + timedelta(minutes=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
            
            # Detect prediction drift
            drift_result = await monitoring_service.detect_prediction_drift(
                model_id=test_model.id,
                baseline_window_start=baseline_start,
                baseline_window_end=baseline_end,
                current_window_start=current_start,
                current_window_end=current_end
            )
        
        # Verify drift detection
        assert drift_result.model_id == test_model.id
        assert drift_result.drift_type == DriftType.PREDICTION
        assert drift_result.drift_detected is True
        assert drift_result.prediction_mean_shift is not None
        assert drift_result.prediction_mean_shift > 0  # Should detect upward shift
        assert drift_result.prediction_distribution_shift is not None
        assert drift_result.prediction_distribution_shift > 0
    
    @pytest.mark.asyncio
    async def test_data_drift_detection(self, test_model):
        """Test data drift detection (schema changes, missing values, etc.)"""
        from app.models.monitoring import PredictionLogDB
        from datetime import datetime, timedelta
        import uuid
        
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=7)
        baseline_end = now - timedelta(days=6)
        current_start = now - timedelta(hours=1)
        current_end = now
        
        async with get_session() as session:
            # Clean up existing logs
            from sqlalchemy import delete
            delete_stmt = delete(PredictionLogDB).where(
                PredictionLogDB.model_id == test_model.id
            )
            await session.execute(delete_stmt)
            await session.commit()
            
            # Create baseline logs with consistent schema
            for i in range(100):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={
                        "feature1": 50.0 + i,
                        "feature2": "value" + str(i % 10),
                        "feature3": 0.5
                    },
                    output_data={"prediction": 1.0},
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            # Create current logs with schema changes (new feature, missing feature)
            for i in range(100):
                input_data = {
                    "feature1": 50.0 + i,
                    "feature4": "new_feature",  # New feature not in baseline
                    # feature2 missing - schema change
                }
                if i % 2 == 0:
                    input_data["feature2"] = "value" + str(i % 10)  # Sometimes present
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data=input_data,
                    output_data={"prediction": 1.0},
                    latency_ms=50.0,
                    timestamp=current_start + timedelta(minutes=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
            
            # Detect data drift
            drift_result = await monitoring_service.detect_data_drift(
                model_id=test_model.id,
                baseline_window_start=baseline_start,
                baseline_window_end=baseline_end,
                current_window_start=current_start,
                current_window_end=current_end
            )
        
        # Verify drift detection
        assert drift_result.model_id == test_model.id
        assert drift_result.drift_type == DriftType.DATA
        assert drift_result.drift_detected is True
        assert drift_result.schema_changes is not None
        assert drift_result.data_quality_metrics is not None
    
    @pytest.mark.asyncio
    async def test_drift_detection_no_data(self, test_model):
        """Test drift detection with no prediction logs"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=7)
        baseline_end = now - timedelta(days=6)
        current_start = now - timedelta(hours=1)
        current_end = now
        
        # Try to detect drift with no data
        drift_result = await monitoring_service.detect_feature_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end
        )
        
        # Should return a result but with drift_detected = False
        assert drift_result.model_id == test_model.id
        assert drift_result.drift_detected is False
    
    @pytest.mark.asyncio
    async def test_store_drift_detection(self, test_model):
        """Test storing drift detection results"""
        from app.schemas.monitoring import ModelDriftDetection, DriftType
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        drift_result = ModelDriftDetection(
            model_id=test_model.id,
            drift_type=DriftType.FEATURE,
            detection_method="ks_test_psi",
            baseline_window_start=now - timedelta(days=7),
            baseline_window_end=now - timedelta(days=6),
            current_window_start=now - timedelta(hours=1),
            current_window_end=now,
            drift_detected=True,
            drift_score=0.35,
            drift_severity="high"
        )
        
        drift_id = await monitoring_service.store_drift_detection(drift_result)
        assert drift_id is not None
        assert isinstance(drift_id, str)
        assert len(drift_id) == 36  # UUID format
    
    @pytest.mark.asyncio
    async def test_detect_feature_drift_with_deployment_id(self, test_model, test_deployment):
        """Test feature drift detection with deployment_id filter"""
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=7)
        baseline_end = now - timedelta(days=6)
        current_start = now - timedelta(hours=1)
        current_end = now
        
        async with get_session() as session:
            from sqlalchemy import delete
            delete_stmt = delete(PredictionLogDB).where(
                PredictionLogDB.model_id == test_model.id
            )
            await session.execute(delete_stmt)
            await session.commit()
            
            # Create logs with deployment_id
            for i in range(50):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    deployment_id=test_deployment.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": 50.0 + i},
                    output_data={"prediction": 1.0},
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
        
        drift_result = await monitoring_service.detect_feature_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end,
            deployment_id=test_deployment.id
        )
        
        assert drift_result.model_id == test_model.id
        assert drift_result.deployment_id == test_deployment.id
    
    @pytest.mark.asyncio
    async def test_detect_prediction_drift_insufficient_data(self, test_model):
        """Test prediction drift detection with insufficient data"""
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=7)
        baseline_end = now - timedelta(days=6)
        current_start = now - timedelta(hours=1)
        current_end = now
        
        async with get_session() as session:
            from sqlalchemy import delete
            delete_stmt = delete(PredictionLogDB).where(
                PredictionLogDB.model_id == test_model.id
            )
            await session.execute(delete_stmt)
            await session.commit()
            
            # Create only 5 logs (insufficient for drift detection)
            for i in range(5):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": 0.5},
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
        
        drift_result = await monitoring_service.detect_prediction_drift(
            model_id=test_model.id,
            baseline_window_start=baseline_start,
            baseline_window_end=baseline_end,
            current_window_start=current_start,
            current_window_end=current_end
        )
        
        assert drift_result.drift_detected is False

