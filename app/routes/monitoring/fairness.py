"""
Bias and fairness routes
Provides endpoints for calculating bias and fairness metrics
"""

from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
import logging

from app.schemas.monitoring import BiasFairnessMetrics
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/models/{model_id}/fairness/metrics", response_model=BiasFairnessMetrics)
async def calculate_fairness_metrics(
    model_id: str,
    protected_attribute: str = Query(..., description="Protected attribute name"),
    start_time: datetime = Query(..., description="Analysis period start"),
    end_time: datetime = Query(..., description="Analysis period end"),
    deployment_id: Optional[str] = Query(None)
):
    """Calculate bias and fairness metrics for a model"""
    try:
        metrics = await monitoring_service.calculate_fairness_metrics(
            model_id=model_id,
            protected_attribute=protected_attribute,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id
        )
        await monitoring_service.store_bias_fairness_metrics(metrics)
        return metrics
    except Exception as e:
        logger.error(f"Error calculating fairness metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/fairness/attributes", response_model=Dict[str, str], status_code=201)
async def configure_protected_attribute(
    model_id: str,
    attribute_name: str = Query(..., description="Protected attribute name"),
    attribute_type: str = Query(..., description="Attribute type"),
    config: Dict[str, Any] = Body(..., description="Attribute configuration")
):
    """Configure protected attribute for fairness monitoring"""
    try:
        config_id = await monitoring_service.configure_protected_attribute(
            model_id=model_id,
            attribute_name=attribute_name,
            attribute_type=attribute_type,
            config=config
        )
        return {"id": config_id, "message": "Protected attribute configured successfully"}
    except Exception as e:
        logger.error(f"Error configuring protected attribute: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/fairness/demographics")
async def get_demographic_distribution(
    model_id: str,
    protected_attribute: str = Query(..., description="Protected attribute name"),
    start_time: datetime = Query(..., description="Analysis period start"),
    end_time: datetime = Query(..., description="Analysis period end"),
    deployment_id: Optional[str] = Query(None)
):
    """Calculate demographic distribution for a model"""
    try:
        distribution = await monitoring_service.calculate_demographic_distribution(
            model_id=model_id,
            protected_attribute=protected_attribute,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id
        )
        return distribution
    except Exception as e:
        logger.error(f"Error calculating demographic distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

