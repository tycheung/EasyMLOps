"""
Tests for Bias and Fairness Service
Tests protected attribute configuration, fairness metrics calculation, and demographic distribution
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import ProtectedAttributeType
from app.database import get_session


class TestBiasFairnessService:
    """Test bias and fairness service functionality"""
    
    @pytest.mark.asyncio
    async def test_configure_protected_attribute(self, test_model):
        """Test configuring a protected attribute"""
        config = await monitoring_service.configure_protected_attribute(
            model_id=test_model.id,
            attribute_name="gender",
            attribute_type=ProtectedAttributeType.CATEGORICAL,
            attribute_values=["male", "female", "other"]
        )
        
        assert config is not None
        assert config.attribute_name == "gender"
        assert config.attribute_type == ProtectedAttributeType.CATEGORICAL
        assert config.attribute_values == ["male", "female", "other"]
        assert config.is_active is True
    
    @pytest.mark.asyncio
    async def test_calculate_fairness_metrics(self, test_model):
        """Test calculating fairness metrics"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create prediction logs with protected attribute and ground truth
        async with get_session() as session:
            # Group A: 80% positive rate
            for i in range(100):
                true_label = 1 if i < 80 else 0  # 80% positive
                pred_label = 1 if i < 75 else 0  # 75% positive predictions
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"gender": "male", "feature": i},
                    output_data={"prediction": pred_label},
                    ground_truth=true_label,
                    ground_truth_timestamp=now,
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            
            # Group B: 50% positive rate
            for i in range(100):
                true_label = 1 if i < 50 else 0  # 50% positive
                pred_label = 1 if i < 45 else 0  # 45% positive predictions
                
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"gender": "female", "feature": i},
                    output_data={"prediction": pred_label},
                    ground_truth=true_label,
                    ground_truth_timestamp=now,
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i + 100),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Calculate fairness metrics
        metrics = await monitoring_service.calculate_fairness_metrics(
            model_id=test_model.id,
            protected_attribute="gender",
            start_time=start_time,
            end_time=end_time
        )
        
        assert metrics is not None
        assert metrics.protected_attribute == "gender"
        assert len(metrics.protected_attribute_values) == 2
        assert metrics.demographic_parity_score is not None
        assert metrics.equalized_odds_score is not None
        assert metrics.equal_opportunity_score is not None
        assert metrics.overall_bias_score is not None
        assert 0 <= metrics.overall_bias_score <= 1
        assert metrics.sample_size == 200
        assert len(metrics.group_metrics) == 2
    
    @pytest.mark.asyncio
    async def test_calculate_demographic_distribution(self, test_model):
        """Test calculating demographic distribution"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create prediction logs with protected attribute
        async with get_session() as session:
            for i in range(200):
                gender = "male" if i < 120 else "female"
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"gender": gender, "feature": i},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Calculate demographic distribution
        distribution = await monitoring_service.calculate_demographic_distribution(
            model_id=test_model.id,
            protected_attribute="gender",
            start_time=start_time,
            end_time=end_time
        )
        
        assert distribution is not None
        assert distribution.protected_attribute == "gender"
        assert distribution.total_samples == 200
        assert "male" in distribution.group_distribution
        assert "female" in distribution.group_distribution
        assert distribution.group_distribution["male"] == 120
        assert distribution.group_distribution["female"] == 80
        assert "male" in distribution.group_percentages
        assert distribution.group_percentages["male"] == 60.0
        assert distribution.group_percentages["female"] == 40.0
    
    @pytest.mark.asyncio
    async def test_fairness_violation_alert(self, test_model):
        """Test that fairness violations trigger alerts"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create logs with significant bias
        async with get_session() as session:
            # Group A: 90% positive rate
            for i in range(100):
                pred_label = 1 if i < 90 else 0
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"gender": "male", "feature": i},
                    output_data={"prediction": pred_label},
                    ground_truth=1 if i < 80 else 0,
                    ground_truth_timestamp=now,
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            
            # Group B: 10% positive rate (huge bias!)
            for i in range(100):
                pred_label = 1 if i < 10 else 0
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"gender": "female", "feature": i},
                    output_data={"prediction": pred_label},
                    ground_truth=1 if i < 50 else 0,
                    ground_truth_timestamp=now,
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i + 100),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Calculate fairness metrics with low threshold
        metrics = await monitoring_service.calculate_fairness_metrics(
            model_id=test_model.id,
            protected_attribute="gender",
            start_time=start_time,
            end_time=end_time,
            fairness_threshold=0.5  # Low threshold to trigger violation
        )
        
        assert metrics is not None
        assert metrics.fairness_violation_detected is True
        assert metrics.bias_alert_triggered is True
        assert metrics.alert_id is not None

