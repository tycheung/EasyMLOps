"""
A/B testing and canary deployment routes
Provides endpoints for A/B testing and canary deployment management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.monitoring import ABTest, ABTestMetrics, CanaryDeployment, CanaryMetrics
from app.services.monitoring_service import monitoring_service
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.get("/ab-tests", response_model=List[ABTest])
async def list_ab_tests(
    status: Optional[str] = Query(None, description="Filter by status"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, description="Maximum number of tests to return")
):
    """List all A/B tests"""
    try:
        async with get_session() as session:
            from app.models.monitoring import ABTestDB
            from sqlalchemy import select, desc
            
            stmt = select(ABTestDB)
            
            if status:
                stmt = stmt.where(ABTestDB.status == status)
            if model_id:
                stmt = stmt.where(
                    (ABTestDB.variant_a_model_id == model_id) | 
                    (ABTestDB.variant_b_model_id == model_id)
                )
            
            stmt = stmt.order_by(desc(ABTestDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            tests_db = result.scalars().all()
            
            return [
                ABTest(
                    id=test.id,
                    test_name=test.test_name,
                    description=test.description,
                    model_name=test.model_name,
                    variant_a_model_id=test.variant_a_model_id,
                    variant_b_model_id=test.variant_b_model_id,
                    variant_a_deployment_id=test.variant_a_deployment_id,
                    variant_b_deployment_id=test.variant_b_deployment_id,
                    variant_a_percentage=test.variant_a_percentage,
                    variant_b_percentage=test.variant_b_percentage,
                    use_sticky_sessions=test.use_sticky_sessions,
                    status=test.status,
                    scheduled_start=test.scheduled_start,
                    scheduled_end=test.scheduled_end,
                    min_sample_size=test.min_sample_size,
                    significance_level=test.significance_level,
                    primary_metric=test.primary_metric,
                    config=test.config or {},
                    created_by=test.created_by
                )
                for test in tests_db
            ]
    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-tests/{test_id}", response_model=ABTest)
async def get_ab_test(test_id: str):
    """Get a specific A/B test by ID"""
    try:
        async with get_session() as session:
            from app.models.monitoring import ABTestDB
            
            test = await session.get(ABTestDB, test_id)
            if not test:
                raise HTTPException(status_code=404, detail=f"A/B test {test_id} not found")
            
            return ABTest(
                id=test.id,
                test_name=test.test_name,
                description=test.description,
                model_name=test.model_name,
                variant_a_model_id=test.variant_a_model_id,
                variant_b_model_id=test.variant_b_model_id,
                variant_a_deployment_id=test.variant_a_deployment_id,
                variant_b_deployment_id=test.variant_b_deployment_id,
                variant_a_percentage=test.variant_a_percentage,
                variant_b_percentage=test.variant_b_percentage,
                use_sticky_sessions=test.use_sticky_sessions,
                status=test.status,
                scheduled_start=test.scheduled_start,
                scheduled_end=test.scheduled_end,
                min_sample_size=test.min_sample_size,
                significance_level=test.significance_level,
                primary_metric=test.primary_metric,
                config=test.config or {},
                created_by=test.created_by
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-tests", response_model=ABTest, status_code=201)
async def create_ab_test(test: ABTest):
    """Create a new A/B test"""
    try:
        ab_test = await monitoring_service.create_ab_test(
            test_name=test.test_name,
            description=test.description,
            model_name=test.model_name,
            variant_a_model_id=test.variant_a_model_id,
            variant_b_model_id=test.variant_b_model_id,
            variant_a_deployment_id=test.variant_a_deployment_id,
            variant_b_deployment_id=test.variant_b_deployment_id,
            variant_a_percentage=test.variant_a_percentage,
            variant_b_percentage=test.variant_b_percentage,
            use_sticky_sessions=test.use_sticky_sessions,
            min_sample_size=test.min_sample_size,
            significance_level=test.significance_level,
            primary_metric=test.primary_metric,
            scheduled_start=test.scheduled_start,
            scheduled_end=test.scheduled_end,
            config=test.config,
            created_by=test.created_by
        )
        await monitoring_service.store_ab_test(ab_test)
        return ab_test
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-tests/{test_id}/start", response_model=Dict[str, bool])
async def start_ab_test(test_id: str):
    """Start an A/B test"""
    try:
        success = await monitoring_service.start_ab_test(test_id=test_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error starting A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-tests/{test_id}/stop", response_model=Dict[str, bool])
async def stop_ab_test(test_id: str):
    """Stop an A/B test"""
    try:
        success = await monitoring_service.stop_ab_test(test_id=test_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error stopping A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-tests/{test_id}/metrics", response_model=ABTestMetrics)
async def get_ab_test_metrics(
    test_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Get A/B test metrics"""
    try:
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        metrics = await monitoring_service.calculate_ab_test_metrics(
            test_id=test_id,
            start_time=start_time,
            end_time=end_time
        )
        return metrics
    except Exception as e:
        logger.error(f"Error getting A/B test metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/canary", response_model=List[CanaryDeployment])
async def list_canary_deployments(
    status: Optional[str] = Query(None, description="Filter by status"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, description="Maximum number of canaries to return")
):
    """List all canary deployments"""
    try:
        async with get_session() as session:
            from app.models.monitoring import CanaryDeploymentDB
            from sqlalchemy import select, desc
            
            stmt = select(CanaryDeploymentDB)
            
            if status:
                stmt = stmt.where(CanaryDeploymentDB.status == status)
            if model_id:
                stmt = stmt.where(CanaryDeploymentDB.model_id == model_id)
            
            stmt = stmt.order_by(desc(CanaryDeploymentDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            canaries_db = result.scalars().all()
            
            return [
                CanaryDeployment(
                    id=canary.id,
                    deployment_name=canary.deployment_name,
                    model_id=canary.model_id,
                    production_deployment_id=canary.production_deployment_id,
                    canary_deployment_id=canary.canary_deployment_id,
                    current_traffic_percentage=canary.current_traffic_percentage,
                    target_traffic_percentage=canary.target_traffic_percentage,
                    rollout_step_size=canary.rollout_step_size,
                    rollout_step_duration_minutes=canary.rollout_step_duration_minutes,
                    max_error_rate_threshold=canary.max_error_rate_threshold,
                    max_latency_increase_pct=canary.max_latency_increase_pct,
                    min_health_check_duration_minutes=canary.min_health_check_duration_minutes,
                    health_check_window_minutes=canary.health_check_window_minutes,
                    status=canary.status,
                    current_step=canary.current_step,
                    total_steps=canary.total_steps,
                    config=canary.config or {},
                    created_by=canary.created_by
                )
                for canary in canaries_db
            ]
    except Exception as e:
        logger.error(f"Error listing canary deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/canary/{canary_id}", response_model=CanaryDeployment)
async def get_canary_deployment(canary_id: str):
    """Get a specific canary deployment by ID"""
    try:
        async with get_session() as session:
            from app.models.monitoring import CanaryDeploymentDB
            
            canary = await session.get(CanaryDeploymentDB, canary_id)
            if not canary:
                raise HTTPException(status_code=404, detail=f"Canary deployment {canary_id} not found")
            
            return CanaryDeployment(
                id=canary.id,
                deployment_name=canary.deployment_name,
                model_id=canary.model_id,
                production_deployment_id=canary.production_deployment_id,
                canary_deployment_id=canary.canary_deployment_id,
                current_traffic_percentage=canary.current_traffic_percentage,
                target_traffic_percentage=canary.target_traffic_percentage,
                rollout_step_size=canary.rollout_step_size,
                rollout_step_duration_minutes=canary.rollout_step_duration_minutes,
                max_error_rate_threshold=canary.max_error_rate_threshold,
                max_latency_increase_pct=canary.max_latency_increase_pct,
                min_health_check_duration_minutes=canary.min_health_check_duration_minutes,
                health_check_window_minutes=canary.health_check_window_minutes,
                status=canary.status,
                current_step=canary.current_step,
                total_steps=canary.total_steps,
                config=canary.config or {},
                created_by=canary.created_by
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting canary deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canary", response_model=CanaryDeployment, status_code=201)
async def create_canary_deployment(canary: CanaryDeployment):
    """Create a new canary deployment"""
    try:
        canary_deployment = await monitoring_service.create_canary_deployment(
            deployment_name=canary.deployment_name,
            model_id=canary.model_id,
            production_deployment_id=canary.production_deployment_id,
            canary_deployment_id=canary.canary_deployment_id,
            target_traffic_percentage=canary.target_traffic_percentage,
            rollout_step_size=canary.rollout_step_size or 10.0,
            rollout_step_duration_minutes=canary.rollout_step_duration_minutes or 60,
            max_error_rate_threshold=canary.max_error_rate_threshold or 5.0,
            max_latency_increase_pct=canary.max_latency_increase_pct or 50.0,
            min_health_check_duration_minutes=canary.min_health_check_duration_minutes or 5,
            health_check_window_minutes=canary.health_check_window_minutes or 15,
            config=canary.config or {},
            created_by=canary.created_by
        )
        return canary_deployment
    except Exception as e:
        logger.error(f"Error creating canary deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canary/{canary_id}/start", response_model=Dict[str, bool])
async def start_canary_rollout(canary_id: str):
    """Start canary rollout"""
    try:
        success = await monitoring_service.start_canary_rollout(canary_id=canary_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error starting canary rollout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canary/{canary_id}/advance", response_model=Dict[str, bool])
async def advance_canary_rollout(canary_id: str):
    """Advance canary rollout to next stage"""
    try:
        success = await monitoring_service.advance_canary_rollout(canary_id=canary_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error advancing canary rollout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canary/{canary_id}/rollback", response_model=Dict[str, bool])
async def rollback_canary(canary_id: str):
    """Rollback canary deployment"""
    try:
        success = await monitoring_service.rollback_canary(canary_id=canary_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error rolling back canary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/canary/{canary_id}/metrics", response_model=CanaryMetrics)
async def get_canary_metrics(
    canary_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Get canary deployment metrics"""
    try:
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        metrics = await monitoring_service.calculate_canary_metrics(
            canary_id=canary_id,
            start_time=start_time,
            end_time=end_time
        )
        return metrics
    except Exception as e:
        logger.error(f"Error getting canary metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/canary/{canary_id}/health", response_model=Dict[str, Any])
async def check_canary_health(canary_id: str):
    """Check canary deployment health"""
    try:
        is_healthy, status_message, rollback_reason = await monitoring_service.check_canary_health(
            canary_id=canary_id
        )
        return {
            "is_healthy": is_healthy,
            "status_message": status_message,
            "rollback_reason": rollback_reason
        }
    except Exception as e:
        logger.error(f"Error checking canary health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-tests/{test_id}/assign", response_model=Dict[str, str])
async def assign_ab_test_variant(
    test_id: str,
    user_id: Optional[str] = Query(None, description="User ID for sticky sessions"),
    session_id: Optional[str] = Query(None, description="Session ID")
):
    """Assign A/B test variant to a user/session"""
    try:
        variant_id = await monitoring_service.assign_variant(
            test_id=test_id,
            user_id=user_id,
            session_id=session_id
        )
        return {"variant_id": variant_id, "message": "Variant assigned successfully"}
    except Exception as e:
        logger.error(f"Error assigning variant: {e}")
        raise HTTPException(status_code=500, detail=str(e))

