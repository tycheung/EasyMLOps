"""
Integration routes
Provides endpoints for external integrations and webhooks
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Body, Query
import logging

from app.schemas.monitoring import ExternalIntegration, WebhookConfig
from app.services.monitoring_service import monitoring_service
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.get("/integrations", response_model=List[Dict[str, Any]])
async def list_integrations(
    integration_type: Optional[str] = Query(None, description="Filter by integration type"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, description="Maximum number of integrations to return")
):
    """List all external integrations"""
    try:
        async with get_session() as session:
            from app.models.monitoring import ExternalIntegrationDB
            from sqlalchemy import select, desc
            
            stmt = select(ExternalIntegrationDB)
            if integration_type:
                stmt = stmt.where(ExternalIntegrationDB.integration_type == integration_type)
            if is_active is not None:
                stmt = stmt.where(ExternalIntegrationDB.is_active == is_active)
            
            stmt = stmt.order_by(desc(ExternalIntegrationDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            integrations_db = result.scalars().all()
            
            return [
                {
                    "id": integration.id,
                    "integration_type": integration.integration_type,
                    "integration_name": integration.integration_name,
                    "description": integration.description,
                    "config": integration.config,
                    "endpoint_url": integration.endpoint_url,
                    "is_active": integration.is_active,
                    "sync_frequency_seconds": integration.sync_frequency_seconds,
                    "model_ids": integration.model_ids,
                    "metric_types": integration.metric_types,
                    "created_at": integration.created_at.isoformat() if integration.created_at else None,
                    "created_by": integration.created_by
                }
                for integration in integrations_db
            ]
    except Exception as e:
        logger.error(f"Error listing integrations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/integrations/webhooks", response_model=List[Dict[str, Any]])
async def list_webhooks(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, description="Maximum number of webhooks to return")
):
    """List all webhook configurations"""
    try:
        async with get_session() as session:
            from app.models.monitoring import WebhookConfigDB
            from sqlalchemy import select, desc
            
            stmt = select(WebhookConfigDB)
            if is_active is not None:
                stmt = stmt.where(WebhookConfigDB.is_active == is_active)
            
            stmt = stmt.order_by(desc(WebhookConfigDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            webhooks_db = result.scalars().all()
            
            return [
                {
                    "id": webhook.id,
                    "webhook_name": webhook.webhook_name,
                    "webhook_url": webhook.webhook_url,
                    "trigger_events": webhook.trigger_events,
                    "headers": webhook.headers,
                    "is_active": webhook.is_active,
                    "created_at": webhook.created_at.isoformat() if webhook.created_at else None
                }
                for webhook in webhooks_db
            ]
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
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

