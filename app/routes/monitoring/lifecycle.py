"""
Model lifecycle routes
Provides endpoints for retraining jobs and model cards
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
import logging

from app.schemas.monitoring import ModelCard
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/models/{model_id}/retraining/jobs", response_model=Dict[str, str], status_code=201)
async def create_retraining_job(
    model_id: str,
    trigger_type: str = Query(..., description="Retraining trigger type"),
    config: Optional[Dict[str, Any]] = Body(None)
):
    """Create a retraining job"""
    try:
        job_id = await monitoring_service.create_retraining_job(
            model_id=model_id,
            trigger_type=trigger_type,
            config=config or {}
        )
        return {"id": job_id, "message": "Retraining job created successfully"}
    except Exception as e:
        logger.error(f"Error creating retraining job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/card", response_model=ModelCard)
async def get_model_card(model_id: str):
    """Get model card for a model"""
    try:
        card = await monitoring_service.get_model_card(model_id=model_id)
        return ModelCard(**card)
    except Exception as e:
        logger.error(f"Error getting model card: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/retraining/triggers", response_model=Dict[str, str], status_code=201)
async def configure_retraining_trigger(
    model_id: str,
    trigger_type: str = Query(..., description="Trigger type"),
    config: Optional[Dict[str, Any]] = Body(None)
):
    """Configure retraining trigger for a model"""
    try:
        trigger_id = await monitoring_service.configure_retraining_trigger(
            model_id=model_id,
            trigger_type=trigger_type,
            config=config or {}
        )
        return {"id": trigger_id, "message": "Retraining trigger configured successfully"}
    except Exception as e:
        logger.error(f"Error configuring retraining trigger: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/card/generate", response_model=ModelCard)
async def generate_model_card(model_id: str):
    """Generate model card for a model"""
    try:
        card_data = await monitoring_service.generate_model_card(model_id=model_id)
        return ModelCard(**card_data)
    except Exception as e:
        logger.error(f"Error generating model card: {e}")
        raise HTTPException(status_code=500, detail=str(e))

