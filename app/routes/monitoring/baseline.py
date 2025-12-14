"""
Model baseline and versioning routes
Provides endpoints for model baselines and version comparisons
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.monitoring import ModelBaseline, ModelVersionComparison
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/models/{model_id}/baseline", response_model=ModelBaseline, status_code=201)
async def create_model_baseline(
    model_id: str,
    baseline_name: str = Query(..., description="Baseline name"),
    description: Optional[str] = Query(None, description="Baseline description"),
    start_time: datetime = Query(..., description="Baseline period start"),
    end_time: datetime = Query(..., description="Baseline period end"),
    deployment_id: Optional[str] = Query(None)
):
    """Create a model baseline"""
    try:
        baseline = await monitoring_service.create_model_baseline(
            model_id=model_id,
            baseline_name=baseline_name,
            description=description,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id
        )
        await monitoring_service.store_model_baseline(baseline)
        return baseline
    except Exception as e:
        logger.error(f"Error creating model baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/baseline", response_model=Optional[ModelBaseline])
async def get_active_baseline(
    model_id: str,
    deployment_id: Optional[str] = Query(None)
):
    """Get the active baseline for a model"""
    try:
        baseline = await monitoring_service.get_active_baseline(
            model_id=model_id,
            deployment_id=deployment_id
        )
        return baseline
    except Exception as e:
        logger.error(f"Error getting active baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/versions/compare", response_model=ModelVersionComparison)
async def compare_model_versions(
    model_id: str,
    version_a_id: str = Query(..., description="First version ID"),
    version_b_id: str = Query(..., description="Second version ID"),
    start_time: datetime = Query(..., description="Comparison period start"),
    end_time: datetime = Query(..., description="Comparison period end")
):
    """Compare two model versions"""
    try:
        comparison = await monitoring_service.compare_model_versions(
            model_id=model_id,
            version_a_id=version_a_id,
            version_b_id=version_b_id,
            start_time=start_time,
            end_time=end_time
        )
        await monitoring_service.store_version_comparison(comparison)
        return comparison
    except Exception as e:
        logger.error(f"Error comparing model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

