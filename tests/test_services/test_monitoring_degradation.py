"""
Tests for Performance Degradation Service
Tests performance degradation detection for classification and regression models
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.database import get_session


class TestPerformanceDegradationService:
    """Test performance degradation service functionality"""
    
    @pytest.mark.asyncio
    async def test_log_prediction_with_ground_truth(self, test_model):
        """Test logging prediction with ground truth"""
        # Log prediction with ground truth
        log_id = await monitoring_service.log_prediction_with_ground_truth(
            model_id=test_model.id,
            deployment_id=None,
            input_data={"feature1": 1.0, "feature2": 2.0},
            output_data={"prediction": 1},
            ground_truth=1,
            latency_ms=50.0,
            api_endpoint="/models/predict",
            success=True
        )
        
        assert log_id is not None
        
        # Verify it was stored
        async with get_session() as session:
            from sqlalchemy import select
            stmt = select(PredictionLogDB).where(PredictionLogDB.id == log_id)
            result = await session.execute(stmt)
            log = result.scalar_one_or_none()
            
            assert log is not None
            assert log.ground_truth == 1
            assert log.ground_truth_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_calculate_classification_metrics(self, test_model):
        """Test classification metrics calculation"""
        from datetime import datetime, timedelta
        
        # Skip if sklearn is not available
        try:
            from sklearn.metrics import accuracy_score
        except ImportError:
            pytest.skip("scikit-learn not available - skipping classification metrics test")
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create prediction logs with ground truth for binary classification
        async with get_session() as session:
            for i in range(100):
                # Create balanced dataset: 50% class 0, 50% class 1
                # With 80% accuracy (80 correct, 20 wrong)
                true_label = i % 2
                pred_label = true_label if i < 80 else (1 - true_label)  # 80% accuracy
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": pred_label},
                    ground_truth=true_label,
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(minutes=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
            
            # Calculate classification metrics
            metrics = await monitoring_service.calculate_classification_metrics(
                model_id=test_model.id,
                start_time=start_time,
                end_time=end_time
            )
        
        # Verify metrics
        assert metrics.model_id == test_model.id
        assert metrics.model_type == "classification"
        assert metrics.accuracy == 0.8  # 80/100
        assert metrics.total_samples == 100
        assert metrics.samples_with_ground_truth == 100
        assert metrics.confusion_matrix is not None
        # For binary classification: TP=40, TN=40, FP=10, FN=10
        assert sum(sum(metrics.confusion_matrix)) == 100
    
    @pytest.mark.asyncio
    async def test_calculate_regression_metrics(self, test_model):
        """Test regression metrics calculation"""
        from datetime import datetime, timedelta
        
        # Skip if sklearn is not available
        try:
            from sklearn.metrics import mean_absolute_error
        except ImportError:
            pytest.skip("scikit-learn not available - skipping regression metrics test")
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create prediction logs with ground truth for regression
        async with get_session() as session:
            for i in range(100):
                true_value = 100.0 + i * 2.0  # 100, 102, 104, ...
                pred_value = true_value + (i % 10) - 5  # Add some error
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": pred_value},
                    ground_truth=true_value,
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(minutes=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
            
            # Calculate regression metrics
            metrics = await monitoring_service.calculate_regression_metrics(
                model_id=test_model.id,
                start_time=start_time,
                end_time=end_time
            )
        
        # Verify metrics
        assert metrics.model_id == test_model.id
        assert metrics.model_type == "regression"
        assert metrics.total_samples == 100
        assert metrics.samples_with_ground_truth == 100
        assert metrics.mae is not None
        assert metrics.mae >= 0
        assert metrics.mse is not None
        assert metrics.mse >= 0
        assert metrics.rmse is not None
        assert metrics.rmse >= 0
        assert metrics.r2_score is not None
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection_classification(self, test_model):
        """Test performance degradation detection for classification models"""
        from datetime import datetime, timedelta
        
        # Skip if sklearn is not available
        try:
            from sklearn.metrics import accuracy_score
        except ImportError:
            pytest.skip("scikit-learn not available - skipping degradation detection test")
        
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
            
            # Create baseline logs with 90% accuracy
            for i in range(100):
                true_label = i % 2
                pred_label = true_label if i < 90 else (1 - true_label)  # 90% accuracy
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": pred_label},
                    ground_truth=true_label,
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            # Create current logs with 70% accuracy (degradation!)
            for i in range(100):
                true_label = i % 2
                pred_label = true_label if i < 70 else (1 - true_label)  # 70% accuracy
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": pred_label},
                    ground_truth=true_label,
                    latency_ms=50.0,
                    timestamp=current_start + timedelta(minutes=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
            
            # Detect degradation
            degradation = await monitoring_service.detect_performance_degradation(
                model_id=test_model.id,
                baseline_window_start=baseline_start,
                baseline_window_end=baseline_end,
                current_window_start=current_start,
                current_window_end=current_end,
                degradation_threshold=0.15  # 15% drop threshold
            )
        
        # Verify degradation detection
        assert degradation.model_id == test_model.id
        assert degradation.performance_degraded is True  # Should detect 20% drop
        assert degradation.accuracy_delta < 0  # Negative delta means degradation
        assert abs(degradation.accuracy_delta) > 0.15  # More than threshold
        assert degradation.degradation_severity is not None
        assert degradation.p_value is not None
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection_regression(self, test_model):
        """Test performance degradation detection for regression models"""
        from datetime import datetime, timedelta
        
        # Skip if sklearn is not available
        try:
            from sklearn.metrics import mean_absolute_error
        except ImportError:
            pytest.skip("scikit-learn not available - skipping regression degradation test")
        
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
            
            # Create baseline logs with low MAE (good performance)
            for i in range(100):
                true_value = 100.0 + i * 2.0
                pred_value = true_value + 2.0  # Small, consistent error
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": pred_value},
                    ground_truth=true_value,
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(hours=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            # Create current logs with high MAE (degradation!)
            for i in range(100):
                true_value = 100.0 + i * 2.0
                pred_value = true_value + 20.0  # Large error (degradation)
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": i},
                    output_data={"prediction": pred_value},
                    ground_truth=true_value,
                    latency_ms=50.0,
                    timestamp=current_start + timedelta(minutes=i),
                    api_endpoint="/predict",
                    success=True
                )
                session.add(log)
            
            await session.commit()
            
            # Detect degradation
            degradation = await monitoring_service.detect_performance_degradation(
                model_id=test_model.id,
                baseline_window_start=baseline_start,
                baseline_window_end=baseline_end,
                current_window_start=current_start,
                current_window_end=current_end,
                degradation_threshold=0.5  # 50% increase in MAE threshold
            )
        
        # Verify degradation detection
        assert degradation.model_id == test_model.id
        assert degradation.performance_degraded is True
        assert degradation.mae_delta > 0  # Positive delta means worse MAE
        assert degradation.mae_delta > 2.0  # Should be significant increase
        assert degradation.degradation_severity is not None
    
    @pytest.mark.asyncio
    async def test_performance_metrics_insufficient_data(self, test_model):
        """Test performance metrics calculation with insufficient data"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Try to calculate metrics with no data
        metrics = await monitoring_service.calculate_classification_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Should return metrics but with zero/None values
        assert metrics.model_id == test_model.id
        assert metrics.total_samples == 0
        assert metrics.samples_with_ground_truth == 0
        assert metrics.accuracy is None
    
    @pytest.mark.asyncio
    async def test_store_performance_history(self, test_model):
        """Test storing performance history"""
        from app.schemas.monitoring import ModelPerformanceHistory
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        perf_history = ModelPerformanceHistory(
            model_id=test_model.id,
            time_window_start=now - timedelta(hours=1),  # Required field
            time_window_end=now,  # Required field
            model_type="classification",  # Required field
            total_samples=100,  # Required field
            samples_with_ground_truth=100,  # Required field
            accuracy=0.85,
            performance_degraded=False
        )
        
        history_id = await monitoring_service.store_performance_history(perf_history)
        assert history_id is not None
        assert isinstance(history_id, str)
    
    @pytest.mark.asyncio
    async def test_calculate_classification_metrics_insufficient_data(self, test_model):
        """Test classification metrics with insufficient data"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Calculate with no data
        metrics = await monitoring_service.calculate_classification_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics.model_id == test_model.id
        assert metrics.total_samples == 0
        assert metrics.accuracy is None
    
    @pytest.mark.asyncio
    async def test_calculate_regression_metrics_insufficient_data(self, test_model):
        """Test regression metrics with insufficient data"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Calculate with no data
        metrics = await monitoring_service.calculate_regression_metrics(
            model_id=test_model.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics.model_id == test_model.id
        assert metrics.total_samples == 0
        assert metrics.mae is None

