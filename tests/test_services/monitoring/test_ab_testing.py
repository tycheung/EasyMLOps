"""
Tests for A/B Testing Service
Tests A/B test creation, variant assignment, metrics calculation, and test management
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.services.monitoring_service import monitoring_service
from app.schemas.monitoring import ABTestStatus


class TestABTestingService:
    """Test A/B testing service functionality"""
    
    @pytest.mark.asyncio
    async def test_create_ab_test(self, test_model):
        """Test creating an A/B test"""
        # Create a second model for variant B
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        # Create A/B test
        ab_test = await monitoring_service.create_ab_test(
            test_name="Model Version Test",
            model_name=test_model.name,
            variant_a_model_id=test_model.id,
            variant_b_model_id=variant_b_model.id,
            variant_a_percentage=50.0,
            variant_b_percentage=50.0,
            primary_metric="accuracy",
            min_sample_size=100,
            description="Test new model version",
            created_by="admin"
        )
        
        assert ab_test is not None
        assert ab_test.test_name == "Model Version Test"
        assert ab_test.model_name == test_model.name
        assert ab_test.variant_a_model_id == test_model.id
        assert ab_test.variant_b_model_id == variant_b_model.id
        assert ab_test.variant_a_percentage == 50.0
        assert ab_test.variant_b_percentage == 50.0
        assert ab_test.status == ABTestStatus.DRAFT
        assert ab_test.primary_metric == "accuracy"
    
    @pytest.mark.asyncio
    async def test_assign_variant(self, test_model):
        """Test assigning a variant to a session"""
        # Create A/B test
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        ab_test = await monitoring_service.create_ab_test(
            test_name="Test",
            model_name=test_model.name,
            variant_a_model_id=test_model.id,
            variant_b_model_id=variant_b_model.id,
            use_sticky_sessions=True
        )
        
        # Start the test
        await monitoring_service.start_ab_test(ab_test.id)
        
        # Assign variant
        variant = await monitoring_service.assign_variant(
            test_id=ab_test.id,
            session_id="test_session_123"
        )
        
        assert variant in ["variant_a", "variant_b"]
        
        # Test sticky assignment - should get same variant
        variant2 = await monitoring_service.assign_variant(
            test_id=ab_test.id,
            session_id="test_session_123"
        )
        
        assert variant2 == variant  # Should be same due to sticky sessions
    
    @pytest.mark.asyncio
    async def test_calculate_ab_test_metrics(self, test_model):
        """Test calculating A/B test metrics"""
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        # Create and start A/B test
        ab_test = await monitoring_service.create_ab_test(
            test_name="Test",
            model_name=test_model.name,
            variant_a_model_id=test_model.id,
            variant_b_model_id=variant_b_model.id
        )
        
        await monitoring_service.start_ab_test(ab_test.id)
        
        # Log some predictions for both variants
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        end_time = now
        
        for i in range(50):
            await monitoring_service.log_prediction(
                model_id=test_model.id,
                deployment_id=None,
                input_data={"feature": i},
                output_data={"prediction": 1},
                latency_ms=50.0,
                api_endpoint="/predict",
                success=True
            )
        
        for i in range(50):
            await monitoring_service.log_prediction(
                model_id=variant_b_model.id,
                deployment_id=None,
                input_data={"feature": i},
                output_data={"prediction": 1},
                latency_ms=45.0,
                api_endpoint="/predict",
                success=True
            )
        
        # Calculate metrics (returns ABTestComparison, not list)
        comparison = await monitoring_service.calculate_ab_test_metrics(
            test_id=ab_test.id,
            start_time=start_time,
            end_time=end_time
        )
        
        assert comparison is not None
        assert comparison.test_id == ab_test.id
        assert comparison.variant_a_metrics is not None
        assert comparison.variant_b_metrics is not None
        assert comparison.variant_a_metrics.variant == "variant_a"
        assert comparison.variant_b_metrics.variant == "variant_b"
    
    @pytest.mark.asyncio
    async def test_start_stop_ab_test(self, test_model):
        """Test starting and stopping an A/B test"""
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        # Create A/B test
        ab_test = await monitoring_service.create_ab_test(
            test_name="Test",
            model_name=test_model.name,
            variant_a_model_id=test_model.id,
            variant_b_model_id=variant_b_model.id
        )
        
        assert ab_test.status == ABTestStatus.DRAFT
        
        # Start test (returns bool)
        started = await monitoring_service.start_ab_test(ab_test.id)
        assert started is True
        
        # Verify test status was updated
        from app.database import get_session
        from app.models.monitoring import ABTestDB
        async with get_session() as session:
            test_db = await session.get(ABTestDB, ab_test.id)
            assert test_db.status == "running"
            assert test_db.start_time is not None
        
        # Stop test (returns bool)
        stopped = await monitoring_service.stop_ab_test(ab_test.id)
        assert stopped is True
        
        # Verify test status was updated
        async with get_session() as session:
            test_db = await session.get(ABTestDB, ab_test.id)
            assert test_db.status == "completed"
            assert test_db.end_time is not None
    
    @pytest.mark.asyncio
    async def test_create_ab_test_invalid_percentages(self, test_model):
        """Test creating A/B test with invalid percentages"""
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        # Test with percentages that don't sum to 100
        with pytest.raises(ValueError, match="must sum to 100"):
            await monitoring_service.create_ab_test(
                test_name="Test",
                model_name=test_model.name,
                variant_a_model_id=test_model.id,
                variant_b_model_id=variant_b_model.id,
                variant_a_percentage=60.0,
                variant_b_percentage=50.0  # Sums to 110, should fail
            )
    
    @pytest.mark.asyncio
    async def test_assign_variant_not_running(self, test_model):
        """Test assigning variant when test is not running"""
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        # Create test but don't start it
        ab_test = await monitoring_service.create_ab_test(
            test_name="Test",
            model_name=test_model.name,
            variant_a_model_id=test_model.id,
            variant_b_model_id=variant_b_model.id
        )
        
        # Try to assign variant (should fail because test is not running)
        with pytest.raises(ValueError, match="not running"):
            await monitoring_service.assign_variant(
                test_id=ab_test.id,
                session_id="test_session"
            )
    
    @pytest.mark.asyncio
    async def test_start_ab_test_invalid_status(self, test_model):
        """Test starting A/B test with invalid status"""
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        ab_test = await monitoring_service.create_ab_test(
            test_name="Test",
            model_name=test_model.name,
            variant_a_model_id=test_model.id,
            variant_b_model_id=variant_b_model.id
        )
        
        # Start test
        await monitoring_service.start_ab_test(ab_test.id)
        
        # Try to start again (should fail - already running)
        with pytest.raises(ValueError):
            await monitoring_service.start_ab_test(ab_test.id)
    
    @pytest.mark.asyncio
    async def test_store_ab_test_metrics(self, test_model):
        """Test storing A/B test metrics"""
        from app.schemas.monitoring import ABTestMetrics
        from datetime import datetime, timedelta
        
        from app.models.model import Model
        from app.database import get_session
        
        async with get_session() as session:
            variant_b_model = Model(
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
            session.add(variant_b_model)
            await session.commit()
        
        ab_test = await monitoring_service.create_ab_test(
            test_name="Test",
            model_name=test_model.name,
            variant_a_model_id=test_model.id,
            variant_b_model_id=variant_b_model.id
        )
        
        now = datetime.utcnow()
        metrics = ABTestMetrics(
            test_id=ab_test.id,
            variant="variant_a",
            model_id=test_model.id,  # Required field
            time_window_start=now - timedelta(hours=1),
            time_window_end=now,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            avg_latency_ms=50.0,
            accuracy=0.95
        )
        
        metrics_id = await monitoring_service.store_ab_test_metrics(metrics)
        assert metrics_id is not None
        assert isinstance(metrics_id, str)

