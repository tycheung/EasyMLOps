"""
Model explainability routes
Provides endpoints for SHAP, LIME, and feature importance explanations
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
import logging

from app.schemas.monitoring import ModelExplanation
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/models/{model_id}/explain/shap", response_model=ModelExplanation)
async def generate_shap_explanation(
    model_id: str,
    input_data: Dict[str, Any] = Body(..., description="Input data for explanation"),
    deployment_id: Optional[str] = Query(None)
):
    """Generate SHAP explanation for a prediction"""
    try:
        explanation = await monitoring_service.generate_shap_explanation(
            model_id=model_id,
            input_data=input_data,
            deployment_id=deployment_id
        )
        await monitoring_service.store_explanation(explanation)
        return explanation
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/explain/lime", response_model=ModelExplanation)
async def generate_lime_explanation(
    model_id: str,
    input_data: Dict[str, Any] = Body(..., description="Input data for explanation"),
    deployment_id: Optional[str] = Query(None)
):
    """Generate LIME explanation for a prediction"""
    try:
        explanation = await monitoring_service.generate_lime_explanation(
            model_id=model_id,
            input_data=input_data,
            deployment_id=deployment_id
        )
        await monitoring_service.store_explanation(explanation)
        return explanation
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/explain/importance", response_model=Dict[str, Any])
async def get_global_feature_importance(
    model_id: str,
    deployment_id: Optional[str] = Query(None)
):
    """Get global feature importance for a model"""
    try:
        importance = await monitoring_service.calculate_global_feature_importance(
            model_id=model_id,
            deployment_id=deployment_id
        )
        return importance
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

