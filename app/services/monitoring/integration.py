"""
Integration service
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select, and_

from app.database import get_session
from app.models.monitoring import (
    ExternalIntegrationDB, WebhookConfigDB, SamplingConfigDB, MetricAggregationConfigDB
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class IntegrationService(BaseMonitoringService):
    """Service for integrations"""
    
    async def create_external_integration(
        self,
        integration_type: str,
        integration_name: str,
        config: Dict[str, Any],
        is_active: bool = True
    ) -> str:
        """Create external integration"""
        try:
            integration_id = str(uuid.uuid4())
            integration_db = ExternalIntegrationDB(
                id=integration_id,
                integration_type=integration_type,
                integration_name=integration_name,
                description=config.get("description"),
                config=config.get("config", {}),
                endpoint_url=config.get("endpoint_url"),
                api_key=config.get("api_key"),
                auth_config=config.get("auth_config", {}),
                is_active=is_active,
                sync_frequency_seconds=config.get("sync_frequency_seconds"),
                model_ids=config.get("model_ids"),
                metric_types=config.get("metric_types"),
                created_by=config.get("created_by")
            )
            
            async with get_session() as session:
                session.add(integration_db)
                await session.commit()
                logger.info(f"Created external integration {integration_id}: {integration_name} ({integration_type})")
                return integration_id
        except Exception as e:
            logger.error(f"Error creating external integration: {e}", exc_info=True)
            raise
    
    async def create_webhook_config(
        self,
        webhook_name: str,
        webhook_url: str,
        event_types: List[str],
        headers: Dict[str, str],
        secret: Optional[str],
        is_active: bool = True
    ) -> str:
        """Create webhook configuration"""
        try:
            webhook_id = str(uuid.uuid4())
            webhook_db = WebhookConfigDB(
                id=webhook_id,
                webhook_name=webhook_name,
                webhook_url=webhook_url,
                trigger_events=event_types,
                headers=headers,
                secret_key=secret,
                is_active=is_active
            )
            
            async with get_session() as session:
                session.add(webhook_db)
                await session.commit()
                logger.info(f"Created webhook config {webhook_id}: {webhook_name}")
                return webhook_id
        except Exception as e:
            logger.error(f"Error creating webhook config: {e}", exc_info=True)
            raise
    
    async def create_sampling_config(
        self,
        config_name: str,
        resource_type: str,
        sampling_strategy: str,
        sampling_rate: float,
        min_volume_threshold: Optional[int],
        conditions: Dict[str, Any],
        model_id: Optional[str],
        deployment_id: Optional[str],
        is_active: bool,
        created_by: Optional[str]
    ) -> str:
        """Create sampling configuration"""
        try:
            config_id = str(uuid.uuid4())
            config_db = SamplingConfigDB(
                id=config_id,
                config_name=config_name,
                resource_type=resource_type,
                sampling_strategy=sampling_strategy,
                sampling_rate=sampling_rate,
                min_volume_threshold=min_volume_threshold,
                conditions=conditions,
                model_id=model_id,
                deployment_id=deployment_id,
                is_active=is_active,
                created_by=created_by
            )
            
            async with get_session() as session:
                session.add(config_db)
                await session.commit()
                logger.info(f"Created sampling config {config_id}: {config_name}")
                return config_id
        except Exception as e:
            logger.error(f"Error creating sampling config: {e}", exc_info=True)
            raise
    
    async def create_metric_aggregation_config(
        self,
        config_name: str,
        metric_type: str,
        aggregation_window_seconds: int,
        aggregation_method: str,
        percentile: Optional[float],
        raw_data_retention_hours: Optional[int],
        aggregated_data_retention_days: Optional[int],
        model_id: Optional[str],
        is_active: bool,
        created_by: Optional[str]
    ) -> str:
        """Create metric aggregation configuration"""
        try:
            config_id = str(uuid.uuid4())
            config_db = MetricAggregationConfigDB(
                id=config_id,
                config_name=config_name,
                metric_type=metric_type,
                aggregation_window_seconds=aggregation_window_seconds,
                aggregation_method=aggregation_method,
                percentile=percentile,
                raw_data_retention_hours=raw_data_retention_hours,
                aggregated_data_retention_days=aggregated_data_retention_days,
                model_id=model_id,
                is_active=is_active,
                created_by=created_by
            )
            
            async with get_session() as session:
                session.add(config_db)
                await session.commit()
                logger.info(f"Created metric aggregation config {config_id}: {config_name}")
                return config_id
        except Exception as e:
            logger.error(f"Error creating metric aggregation config: {e}", exc_info=True)
            raise
    
    async def should_sample(self, resource_type: str, model_id: Optional[str] = None) -> bool:
        """Check if should sample based on active sampling configs"""
        try:
            async with get_session() as session:
                stmt = select(SamplingConfigDB).where(
                    and_(
                        SamplingConfigDB.resource_type == resource_type,
                        SamplingConfigDB.is_active == True
                    )
                )
                if model_id:
                    stmt = stmt.where(
                        (SamplingConfigDB.model_id == model_id) | (SamplingConfigDB.model_id.is_(None))
                    )
                
                result = await session.execute(stmt)
                configs = result.scalars().all()
                
                if not configs:
                    return False
                
                # Use the first active config's sampling rate
                import random
                config = configs[0]
                return random.random() < config.sampling_rate
                
        except Exception as e:
            logger.error(f"Error checking sampling: {e}")
            return False
