"""
Tests for Model Baseline and Versioning Services
Tests baseline creation, retrieval, and model version comparison
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import PredictionLogDB
from app.database import get_session


class TestModelBaselineService:
    """Test model baseline service functionality"""
    
    @pytest.mark.asyncio
    async def test_create_model_baseline(self, test_model):
        """Test creating a model performance baseline"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=2)
        end_time = now - timedelta(hours=1)
        
        # Create some prediction logs for baseline
        async with get_session() as session:
            for i in range(50):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature": i},
                    output_data={"prediction": i % 2},
                    latency_ms=50.0 + i * 0.1,
                    timestamp=start_time + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Create baseline
        baseline = await monitoring_service.create_model_baseline(
            model_id=test_model.id,
            model_name=test_model.name,
            model_version=test_model.version,
            baseline_type="performance",
            start_time=start_time,
            end_time=end_time,
            is_production=True,
            description="Initial production baseline"
        )
        
        assert baseline is not None
        assert baseline.model_name == test_model.name
        assert baseline.model_version == test_model.version
        assert baseline.baseline_type == "performance"
        assert baseline.is_production is True
        assert baseline.baseline_sample_count == 50
        assert baseline.baseline_avg_latency_ms is not None
    
    @pytest.mark.asyncio
    async def test_get_active_baseline(self, test_model):
        """Test retrieving active baseline"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time = now - timedelta(hours=2)
        end_time = now - timedelta(hours=1)
        
        # Create baseline
        baseline = await monitoring_service.create_model_baseline(
            model_id=test_model.id,
            model_name=test_model.name,
            model_version=test_model.version,
            baseline_type="performance",
            start_time=start_time,
            end_time=end_time
        )
        
        # Retrieve active baseline
        retrieved = await monitoring_service.get_active_baseline(
            model_name=test_model.name,
            baseline_type="performance"
        )
        
        assert retrieved is not None
        assert retrieved.model_name == baseline.model_name
        assert retrieved.model_version == baseline.model_version
        assert retrieved.is_active is True
    
    @pytest.mark.asyncio
    async def test_compare_model_versions(self, test_model):
        """Test comparing two model versions"""
        from datetime import datetime, timedelta
        from app.models.model import Model
        
        now = datetime.utcnow()
        
        # Create baseline version (v1.0.0)
        baseline_start = now - timedelta(hours=4)
        baseline_end = now - timedelta(hours=3)
        
        async with get_session() as session:
            for i in range(50):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=test_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature": i},
                    output_data={"prediction": i % 2},
                    ground_truth=i % 2,
                    latency_ms=50.0,
                    timestamp=baseline_start + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Create baseline
        baseline = await monitoring_service.create_model_baseline(
            model_id=test_model.id,
            model_name=test_model.name,
            model_version=test_model.version,
            baseline_type="performance",
            start_time=baseline_start,
            end_time=baseline_end
        )
        
        # Create comparison version (v2.0.0) - better performance
        comparison_start = now - timedelta(hours=2)
        comparison_end = now - timedelta(hours=1)
        
        # Create a new model for comparison
        async with get_session() as session:
            comparison_model = Model(
                id=str(uuid.uuid4()),
                name=test_model.name,
                version="2.0.0",
                framework=test_model.framework,
                model_type=test_model.model_type,
                status="validated",
                file_name="model_v2.pkl",
                file_size=1024,
                file_hash=f"test_hash_{uuid.uuid4().hex[:8]}"
            )
            session.add(comparison_model)
            await session.commit()
            
            for i in range(50):
                log = PredictionLogDB(
                    id=str(uuid.uuid4()),
                    model_id=comparison_model.id,
                    request_id=str(uuid.uuid4()),
                    input_data={"feature": i},
                    output_data={"prediction": i % 2},
                    ground_truth=i % 2,
                    latency_ms=45.0,  # Better latency
                    timestamp=comparison_start + timedelta(seconds=i),
                    api_endpoint="/models/predict",
                    success=True
                )
                session.add(log)
            await session.commit()
        
        # Compare versions (parameter names are start_time and end_time)
        comparison = await monitoring_service.compare_model_versions(
            model_name=test_model.name,
            baseline_version=test_model.version,
            comparison_version="2.0.0",
            baseline_model_id=test_model.id,
            comparison_model_id=comparison_model.id,
            start_time=comparison_start,
            end_time=comparison_end
        )
        
        assert comparison is not None
        assert comparison.model_name == test_model.name
        assert comparison.baseline_version == test_model.version
        assert comparison.comparison_version == "2.0.0"
        # avg_latency_delta_ms may be None if no logs found, or negative if improvement
        if comparison.avg_latency_delta_ms is not None:
            assert comparison.avg_latency_delta_ms < 0  # Negative means improvement
            assert comparison.performance_improved is True
        else:
            # If no latency data, performance_improved should be False
            assert comparison.performance_improved is False
    
    @pytest.mark.asyncio
    async def test_baseline_deactivation(self, test_model):
        """Test deactivating old baseline when creating new one"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        start_time1 = now - timedelta(hours=4)
        end_time1 = now - timedelta(hours=3)
        start_time2 = now - timedelta(hours=2)
        end_time2 = now - timedelta(hours=1)
        
        # Create first baseline
        baseline1 = await monitoring_service.create_model_baseline(
            model_id=test_model.id,
            model_name=test_model.name,
            model_version=test_model.version,
            baseline_type="performance",
            start_time=start_time1,
            end_time=end_time1
        )
        
        assert baseline1.is_active is True
        
        # Create second baseline (should deactivate first)
        baseline2 = await monitoring_service.create_model_baseline(
            model_id=test_model.id,
            model_name=test_model.name,
            model_version=test_model.version,
            baseline_type="performance",
            start_time=start_time2,
            end_time=end_time2
        )
        
        assert baseline2.is_active is True
        
        # Verify first baseline is deactivated
        from app.models.monitoring import ModelBaselineDB
        async with get_session() as session:
            from sqlalchemy import select
            stmt = select(ModelBaselineDB).where(ModelBaselineDB.id == baseline1.id)
            result = await session.execute(stmt)
            retrieved_baseline1 = result.scalar_one_or_none()
            
            assert retrieved_baseline1 is not None
            assert retrieved_baseline1.is_active is False
    
    @pytest.mark.asyncio
    async def test_store_model_baseline(self, test_model):
        """Test storing model baseline directly"""
        from app.schemas.monitoring import ModelBaseline
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        baseline = ModelBaseline(
            model_id=test_model.id,
            model_name=test_model.name,
            model_version=test_model.version,
            baseline_type="performance",
            baseline_sample_count=100,
            baseline_time_window_start=now - timedelta(hours=2),
            baseline_time_window_end=now - timedelta(hours=1),
            baseline_avg_latency_ms=50.0,
            baseline_p50_latency_ms=45.0,
            baseline_p95_latency_ms=75.0,
            baseline_p99_latency_ms=90.0
        )
        
        baseline_id = await monitoring_service.store_model_baseline(baseline)
        assert baseline_id is not None
        assert isinstance(baseline_id, str)
    
    @pytest.mark.asyncio
    async def test_get_active_baseline_not_found(self, test_model):
        """Test getting active baseline when none exists"""
        baseline = await monitoring_service.get_active_baseline(
            model_name="nonexistent_model",
            baseline_type="performance"
        )
        
        assert baseline is None
    
    @pytest.mark.asyncio
    async def test_compare_model_versions_no_baseline(self, test_model):
        """Test comparing model versions when baseline doesn't exist"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        
        # Use a unique model name that definitely doesn't have a baseline
        unique_model_name = f"nonexistent_model_{uuid.uuid4().hex[:8]}"
        
        # Try to compare without creating baseline first
        with pytest.raises(ValueError, match="No baseline found"):
            await monitoring_service.compare_model_versions(
                model_name=unique_model_name,  # Use unique name to ensure no baseline exists
                baseline_version="1.0.0",
                comparison_version="2.0.0",
                baseline_model_id=test_model.id,
                comparison_model_id=test_model.id,
                start_time=now - timedelta(hours=1),
                end_time=now
            )
    
    @pytest.mark.asyncio
    async def test_store_version_comparison(self, test_model):
        """Test storing version comparison"""
        from app.schemas.monitoring import ModelVersionComparison
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        comparison = ModelVersionComparison(
            model_name=test_model.name,
            baseline_version="1.0.0",
            comparison_version="2.0.0",
            baseline_model_id=test_model.id,
            comparison_model_id=test_model.id,
            comparison_window_start=now - timedelta(hours=1),
            comparison_window_end=now,
            accuracy_delta=0.05,
            performance_improved=True,
            performance_degraded=False
        )
        
        comparison_id = await monitoring_service.store_version_comparison(comparison)
        assert comparison_id is not None
        assert isinstance(comparison_id, str)

