"""
Tests for Model Explainability Service
Tests feature importance calculation and model explanation storage/retrieval
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import (
    ModelExplanation, ExplanationType, FeatureImportance, ImportanceType
)
from app.database import get_session


class TestExplainabilityService:
    """Test explainability service functionality"""
    
    @pytest.mark.asyncio
    async def test_calculate_global_feature_importance(self, test_model):
        """Test calculating global feature importance"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        # Create prediction logs with various features
        async with get_session() as session:
            for i in range(50):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature1": float(i), "feature2": float(i * 2), "feature3": float(i * 0.5)},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Calculate global feature importance
        importance = await monitoring_service.calculate_global_feature_importance(
            model_id=test_model.id,
            calculation_method="variance",
            start_time=start_time,
            end_time=end_time
        )
        
        assert importance is not None
        assert importance.model_id == test_model.id
        assert importance.importance_type == ImportanceType.GLOBAL
        assert importance.calculation_method == "variance"
        assert len(importance.feature_importance_scores) == 3
        assert importance.total_features == 3
        assert importance.sample_size == 50
        assert all(score >= 0 for score in importance.feature_importance_scores.values())
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_explanation(self, test_model):
        """Test storing and retrieving explanations"""
        input_data = {"feature1": 1.0, "feature2": 2.0}
        
        # Create a simple explanation
        explanation = ModelExplanation(
            model_id=test_model.id,
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            explanation_method="builtin",
            input_data=input_data,
            feature_importance={"feature1": 0.6, "feature2": 0.4},
            computation_time_ms=10.0
        )
        
        # Store explanation
        stored_id = await monitoring_service.store_explanation(explanation)
        assert stored_id is not None
        
        # Retrieve explanation
        retrieved = await monitoring_service.get_explanation(stored_id)
        assert retrieved is not None
        assert retrieved.id == stored_id
        assert retrieved.model_id == test_model.id
        assert retrieved.feature_importance == {"feature1": 0.6, "feature2": 0.4}

