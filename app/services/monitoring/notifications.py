"""
Notifications service
Handles notification channels and alert notifications
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import NotificationChannelDB, WebhookConfigDB
from app.schemas.monitoring import (
    Alert, NotificationChannel, NotificationChannelType
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class NotificationService(BaseMonitoringService):
    """Service for notifications"""
    
    async def create_notification_channel(
        self,
        channel_name: str,
        channel_type: NotificationChannelType,
        config: Dict[str, Any],
        email_recipients: Optional[List[str]] = None,
        slack_webhook_url: Optional[str] = None,
        slack_channel: Optional[str] = None,
        pagerduty_integration_key: Optional[str] = None,
        webhook_url: Optional[str] = None,
        severity_filter: Optional[List[str]] = None,
        component_filter: Optional[List[str]] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> NotificationChannel:
        """Create a notification channel"""
        try:
            channel = NotificationChannel(
                channel_name=channel_name,
                channel_type=channel_type,
                description=description,
                config=config,
                email_recipients=email_recipients or [],
                slack_webhook_url=slack_webhook_url,
                slack_channel=slack_channel,
                pagerduty_integration_key=pagerduty_integration_key,
                webhook_url=webhook_url,
                severity_filter=severity_filter or [],
                component_filter=component_filter or [],
                is_active=True,
                created_by=created_by
            )
            
            # Store channel
            stored_id = await self.store_notification_channel(channel)
            channel.id = stored_id
            
            logger.info(f"Created notification channel {channel_name} ({channel_type.value})")
            return channel
            
        except Exception as e:
            logger.error(f"Error creating notification channel: {e}", exc_info=True)
            raise
    
    async def store_notification_channel(self, channel: NotificationChannel) -> str:
        """Store notification channel in database"""
        try:
            channel_db = NotificationChannelDB(
                id=str(uuid.uuid4()),
                channel_name=channel.channel_name,
                channel_type=channel.channel_type.value,
                description=channel.description,
                config=channel.config,
                email_recipients=channel.email_recipients,
                email_template=channel.email_template if hasattr(channel, 'email_template') else None,
                slack_webhook_url=channel.slack_webhook_url,
                slack_channel=channel.slack_channel,
                pagerduty_integration_key=channel.pagerduty_integration_key,
                pagerduty_service_id=channel.pagerduty_service_id if hasattr(channel, 'pagerduty_service_id') else None,
                webhook_url=channel.webhook_url,
                webhook_headers=channel.webhook_headers if hasattr(channel, 'webhook_headers') else None,
                sms_provider=channel.sms_provider if hasattr(channel, 'sms_provider') else None,
                sms_recipients=channel.sms_recipients if hasattr(channel, 'sms_recipients') else None,
                sms_config=channel.sms_config if hasattr(channel, 'sms_config') else None,
                severity_filter=channel.severity_filter,
                component_filter=channel.component_filter,
                is_active=channel.is_active,
                last_sent_at=channel.last_sent_at if hasattr(channel, 'last_sent_at') else None,
                total_sent=channel.total_sent if hasattr(channel, 'total_sent') else 0,
                failure_count=channel.failure_count if hasattr(channel, 'failure_count') else 0,
                last_error=channel.last_error if hasattr(channel, 'last_error') else None,
                created_by=channel.created_by
            )
            
            async with get_session() as session:
                session.add(channel_db)
                await session.commit()
                return channel_db.id
                
        except Exception as e:
            logger.error(f"Error storing notification channel: {e}")
            raise
    
    async def send_alert_notification(
        self,
        alert: Alert,
        channel_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, bool]]:
        """Send alert notification through configured channels"""
        try:
            results = []
            
            # Get channels to use
            async with get_session() as session:
                if channel_ids:
                    channels = []
                    for channel_id in channel_ids:
                        channel_db = await session.get(NotificationChannelDB, channel_id)
                        if channel_db and channel_db.is_active:
                            channels.append(channel_db)
                else:
                    # Get all active channels that match alert filters
                    stmt = select(NotificationChannelDB).where(
                        NotificationChannelDB.is_active == True
                    )
                    result = await session.execute(stmt)
                    all_channels = result.scalars().all()
                    
                    # Filter by severity and component
                    channels = []
                    for channel in all_channels:
                        if channel.severity_filter and alert.severity.value not in channel.severity_filter:
                            continue
                        if channel.component_filter and alert.component.value not in channel.component_filter:
                            continue
                        channels.append(channel)
            
            # Send notifications
            for channel_db in channels:
                try:
                    # Prepare notification message
                    message = f"Alert: {alert.title}\n{alert.description}"
                    if alert.metric_value is not None:
                        message += f"\nMetric Value: {alert.metric_value}"
                    if alert.threshold_value is not None:
                        message += f"\nThreshold: {alert.threshold_value}"
                    
                    # Send based on channel type
                    channel_type = NotificationChannelType(channel_db.channel_type)
                    
                    if channel_type == NotificationChannelType.EMAIL:
                        logger.info(f"Would send email to {channel_db.email_recipients}: {message}")
                    
                    elif channel_type == NotificationChannelType.SLACK:
                        logger.info(f"Would send Slack message to {channel_db.slack_channel}: {message}")
                    
                    elif channel_type == NotificationChannelType.PAGERDUTY:
                        logger.info(f"Would trigger PagerDuty incident: {message}")
                    
                    elif channel_type == NotificationChannelType.WEBHOOK:
                        # Trigger webhook
                        await self.trigger_webhook(
                            event_name="alert_created",
                            event_data={
                                "alert_id": alert.id,
                                "severity": alert.severity.value,
                                "component": alert.component.value,
                                "title": alert.title,
                                "description": alert.description,
                                "message": message
                            },
                            model_id=alert.affected_models[0] if alert.affected_models else None
                        )
                    
                    # Update channel stats
                    channel_db.total_sent += 1
                    channel_db.last_sent_at = datetime.utcnow()
                    
                    async with get_session() as session:
                        session.add(channel_db)
                        await session.commit()
                    
                    results.append((channel_db.id, True))
                    
                except Exception as e:
                    logger.error(f"Error sending notification via {channel_db.channel_name}: {e}")
                    channel_db.failure_count += 1
                    channel_db.last_error = str(e)
                    
                    async with get_session() as session:
                        session.add(channel_db)
                        await session.commit()
                    
                    results.append((channel_db.id, False))
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}", exc_info=True)
            return []
    
    async def trigger_webhook(
        self,
        event_name: str,
        event_data: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> List[Tuple[str, bool]]:
        """Trigger webhooks for a given event"""
        try:
            # Get active webhooks for this event
            async with get_session() as session:
                stmt = select(WebhookConfigDB).where(
                    WebhookConfigDB.is_active == True
                )
                result = await session.execute(stmt)
                all_webhooks = result.scalars().all()
            
            # Filter webhooks that match the event
            matching_webhooks = []
            for webhook in all_webhooks:
                if event_name in (webhook.trigger_events or []):
                    matching_webhooks.append(webhook)
            
            results = []
            
            for webhook_db in matching_webhooks:
                try:
                    # Prepare payload
                    payload = event_data.copy()
                    payload["event"] = event_name
                    payload["timestamp"] = datetime.utcnow().isoformat()
                    if model_id:
                        payload["model_id"] = model_id
                    
                    # Apply payload template if provided
                    if webhook_db.payload_template:
                        # Simple template replacement
                        payload_str = webhook_db.payload_template
                        for key, value in payload.items():
                            payload_str = payload_str.replace(f"{{{{{key}}}}}", str(value))
                        payload = {"custom": payload_str}
                    
                    # Make HTTP request (would use httpx or requests in production)
                    logger.info(f"Would trigger webhook {webhook_db.webhook_name} to {webhook_db.webhook_url}")
                    
                    # Update webhook stats
                    webhook_db.total_triggers += 1
                    webhook_db.last_triggered_at = datetime.utcnow()
                    webhook_db.last_success = True
                    webhook_db.success_count += 1
                    
                    async with get_session() as session:
                        session.add(webhook_db)
                        await session.commit()
                    
                    results.append((webhook_db.id, True))
                    
                except Exception as e:
                    logger.error(f"Error triggering webhook {webhook_db.webhook_name}: {e}")
                    webhook_db.failure_count += 1
                    webhook_db.last_success = False
                    webhook_db.last_error = str(e)
                    
                    async with get_session() as session:
                        session.add(webhook_db)
                        await session.commit()
                    
                    results.append((webhook_db.id, False))
            
            return results
            
        except Exception as e:
            logger.error(f"Error triggering webhooks: {e}", exc_info=True)
            return []

