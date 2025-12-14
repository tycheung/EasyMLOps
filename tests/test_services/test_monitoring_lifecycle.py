"""
Tests for Model Lifecycle Service
Tests retraining job creation, trigger configuration, and model card generation
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import (
    RetrainingTriggerType, RetrainingJobStatus
)
from app.database import get_session


class TestModelLifecycleService:
    """Test model lifecycle service functionality"""
    
    @pytest.mark.asyncio
    async def test_configure_retraining_trigger(self, test_model):
        """Test configuring a retraining trigger"""
        config = await monitoring_service.configure_retraining_trigger(
            model_id=test_model.id,
            trigger_type=RetrainingTriggerType.PERFORMANCE,
            performance_threshold=0.8,
            performance_metric="accuracy",
            degradation_window_hours=24
        )
        
        assert config is not None
        assert config.model_id == test_model.id
        assert config.trigger_type == RetrainingTriggerType.PERFORMANCE
        assert config.performance_threshold == 0.8
        assert config.performance_metric == "accuracy"
        assert config.is_enabled is True
    
    @pytest.mark.asyncio
    async def test_create_retraining_job(self, test_model):
        """Test creating a retraining job"""
        job = await monitoring_service.create_retraining_job(
            model_id=test_model.id,
            trigger_type=RetrainingTriggerType.MANUAL,
            job_name="test_retraining_job",
            trigger_reason="Manual retraining request",
            triggered_by="test_user"
        )
        
        assert job is not None
        assert job.model_id == test_model.id
        assert job.trigger_type == RetrainingTriggerType.MANUAL
        assert job.job_name == "test_retraining_job"
        assert job.status == RetrainingJobStatus.PENDING
        assert job.trigger_reason == "Manual retraining request"
        assert job.triggered_by == "test_user"
    
    @pytest.mark.asyncio
    async def test_generate_model_card(self, test_model):
        """Test generating a model card"""
        from datetime import datetime, timedelta
        
        # Create some prediction logs for usage statistics
        now = datetime.utcnow()
        async with get_session() as session:
            for i in range(10):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": float(i)},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=now - timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Generate model card
        card = await monitoring_service.generate_model_card(
            model_id=test_model.id,
            include_performance=True,
            include_training_info=True
        )
        
        assert card is not None
        assert card.model_id == test_model.id
        assert card.card_content is not None
        assert "model_details" in card.card_content
        assert "usage_statistics" in card.card_content
        assert card.card_content["model_details"]["name"] == test_model.name
        assert card.card_content["usage_statistics"]["total_predictions"] == 10
    
    @pytest.mark.asyncio
    async def test_get_model_card(self, test_model):
        """Test retrieving a model card"""
        # Generate and store a model card
        card = await monitoring_service.generate_model_card(
            model_id=test_model.id
        )
        
        # Retrieve it
        retrieved = await monitoring_service.get_model_card(test_model.id)
        
        assert retrieved is not None
        assert retrieved.id == card.id
        assert retrieved.model_id == test_model.id
        assert retrieved.card_content == card.card_content

