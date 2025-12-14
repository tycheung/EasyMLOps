"""
Integration routes
Provides endpoints for external integrations and webhooks
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
import logging

from app.schemas.monitoring import ExternalIntegration, WebhookConfig
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/integrations", response_model=Dict[str, str], status_code=201)
async def create_external_integration(integration: ExternalIntegration):
    """Create external integration"""
    try:
        integration_id = await monitoring_service.create_external_integration(
            integration_type=integration.integration_type,
            integration_name=integration.integration_name,
            config=integration.config,
            is_active=integration.is_active
        )
        return {"id": integration_id, "message": "Integration created successfully"}
    except Exception as e:
        logger.error(f"Error creating integration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrations/webhooks", response_model=Dict[str, str], status_code=201)
async def create_webhook_config(webhook: WebhookConfig):
    """Create webhook configuration"""
    try:
        webhook_id = await monitoring_service.create_webhook_config(
            webhook_name=webhook.webhook_name,
            webhook_url=webhook.webhook_url,
            event_types=webhook.event_types,
            headers=webhook.headers,
            secret=webhook.secret,
            is_active=webhook.is_active
        )
        return {"id": webhook_id, "message": "Webhook created successfully"}
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrations/sampling", response_model=Dict[str, str], status_code=201)
async def create_sampling_config(config: Dict[str, Any]):
    """Create sampling configuration"""
    try:
        from app.schemas.monitoring import SamplingConfig
        config_obj = SamplingConfig(**config)
        config_id = await monitoring_service.create_sampling_config(
            config_name=config_obj.config_name,
            resource_type=config_obj.resource_type,
            sampling_strategy=config_obj.sampling_strategy,
            sampling_rate=config_obj.sampling_rate,
            min_volume_threshold=config_obj.min_volume_threshold,
            conditions=config_obj.conditions,
            model_id=config_obj.model_id,
            deployment_id=config_obj.deployment_id,
            is_active=config_obj.is_active,
            created_by=config_obj.created_by
        )
        return {"id": config_id, "message": "Sampling configuration created successfully"}
    except Exception as e:
        logger.error(f"Error creating sampling configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrations/aggregation", response_model=Dict[str, str], status_code=201)
async def create_metric_aggregation_config(config: Dict[str, Any]):
    """Create metric aggregation configuration"""
    try:
        from app.schemas.monitoring import MetricAggregationConfig
        config_obj = MetricAggregationConfig(**config)
        config_id = await monitoring_service.create_metric_aggregation_config(
            config_name=config_obj.config_name,
            metric_type=config_obj.metric_type,
            aggregation_window_seconds=config_obj.aggregation_window_seconds,
            aggregation_method=config_obj.aggregation_method,
            percentile=config_obj.percentile,
            raw_data_retention_hours=config_obj.raw_data_retention_hours,
            aggregated_data_retention_days=config_obj.aggregated_data_retention_days,
            model_id=config_obj.model_id,
            is_active=config_obj.is_active,
            created_by=config_obj.created_by
        )
        return {"id": config_id, "message": "Metric aggregation configuration created successfully"}
    except Exception as e:
        logger.error(f"Error creating metric aggregation configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

