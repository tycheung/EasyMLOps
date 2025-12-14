"""
Alert management service
Handles alert grouping and escalation
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import AlertDB, AlertEscalationDB, AlertGroupDB
from app.schemas.monitoring import (
    Alert, AlertGroup, AlertEscalation, AlertSeverity, SystemComponent,
    EscalationTriggerCondition
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class AlertManagementService(BaseMonitoringService):
    """Service for alert management"""
    
    async def group_alerts(
        self,
        alert_ids: List[str],
        group_key: str,
        group_type: str = "custom"
    ) -> AlertGroup:
        """Group multiple alerts together"""
        try:
            # Get alerts
            async with get_session() as session:
                alerts = []
                first_alert_time = None
                last_alert_time = None
                
                for alert_id in alert_ids:
                    alert_db = await session.get(AlertDB, alert_id)
                    if alert_db:
                        alerts.append(alert_db)
                        if not first_alert_time or alert_db.triggered_at < first_alert_time:
                            first_alert_time = alert_db.triggered_at
                        if not last_alert_time or alert_db.triggered_at > last_alert_time:
                            last_alert_time = alert_db.triggered_at
                
                if not alerts:
                    raise ValueError("No valid alerts to group")
            
            group = AlertGroup(
                group_key=group_key,
                group_type=group_type,
                alert_ids=alert_ids,
                alert_count=len(alert_ids),
                first_alert_at=first_alert_time or datetime.utcnow(),
                last_alert_at=last_alert_time or datetime.utcnow(),
                is_active=True
            )
            
            # Store group
            stored_id = await self.store_alert_group(group)
            group.id = stored_id
            
            logger.info(f"Grouped {len(alert_ids)} alerts with key {group_key}")
            return group
            
        except Exception as e:
            logger.error(f"Error grouping alerts: {e}", exc_info=True)
            raise
    
    async def store_alert_group(self, group: AlertGroup) -> str:
        """Store alert group in database"""
        try:
            group_db = AlertGroupDB(
                id=str(uuid.uuid4()),
                group_key=group.group_key,
                group_type=group.group_type,
                alert_ids=group.alert_ids,
                alert_count=group.alert_count,
                is_active=group.is_active,
                first_alert_at=group.first_alert_at,
                last_alert_at=group.last_alert_at,
                resolved_at=group.resolved_at if hasattr(group, 'resolved_at') else None,
                resolved_by=group.resolved_by if hasattr(group, 'resolved_by') else None
            )
            
            async with get_session() as session:
                session.add(group_db)
                await session.commit()
                return group_db.id
                
        except Exception as e:
            logger.error(f"Error storing alert group: {e}")
            raise
    
    async def create_alert_escalation(
        self,
        escalation_name: str,
        trigger_condition: EscalationTriggerCondition,
        trigger_value: Dict[str, Any],
        escalation_actions: List[str],
        severity_filter: Optional[List[str]] = None,
        component_filter: Optional[List[str]] = None,
        notification_channels: Optional[List[str]] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> AlertEscalation:
        """Create an alert escalation rule"""
        try:
            escalation = AlertEscalation(
                escalation_name=escalation_name,
                description=description,
                trigger_condition=trigger_condition,
                trigger_value=trigger_value,
                escalation_actions=escalation_actions,
                notification_channels=notification_channels or [],
                severity_filter=severity_filter or [],
                component_filter=component_filter or [],
                is_active=True,
                created_by=created_by
            )
            
            # Store escalation
            stored_id = await self.store_alert_escalation(escalation)
            escalation.id = stored_id
            
            logger.info(f"Created alert escalation {escalation_name}")
            return escalation
            
        except Exception as e:
            logger.error(f"Error creating alert escalation: {e}", exc_info=True)
            raise
    
    async def store_alert_escalation(self, escalation: AlertEscalation) -> str:
        """Store alert escalation in database"""
        try:
            escalation_db = AlertEscalationDB(
                id=str(uuid.uuid4()),
                escalation_name=escalation.escalation_name,
                description=escalation.description,
                trigger_condition=escalation.trigger_condition.value,
                trigger_value=escalation.trigger_value,
                escalation_actions=escalation.escalation_actions,
                notification_channels=escalation.notification_channels,
                severity_filter=escalation.severity_filter,
                component_filter=escalation.component_filter,
                is_active=escalation.is_active,
                escalation_count=escalation.escalation_count if hasattr(escalation, 'escalation_count') else 0,
                last_escalated_at=escalation.last_escalated_at if hasattr(escalation, 'last_escalated_at') else None,
                created_by=escalation.created_by
            )
            
            async with get_session() as session:
                session.add(escalation_db)
                await session.commit()
                return escalation_db.id
                
        except Exception as e:
            logger.error(f"Error storing alert escalation: {e}")
            raise
    
    async def check_and_escalate_alerts(self) -> List[str]:
        """Check for alerts that need escalation"""
        try:
            escalated = []
            
            # Get active, unacknowledged alerts
            async with get_session() as session:
                stmt = select(AlertDB).where(
                    AlertDB.is_active == True,
                    AlertDB.is_acknowledged == False
                )
                result = await session.execute(stmt)
                alerts = result.scalars().all()
            
            # Get active escalation rules
            async with get_session() as session:
                stmt = select(AlertEscalationDB).where(
                    AlertEscalationDB.is_active == True
                )
                result = await session.execute(stmt)
                escalations = result.scalars().all()
            
            for alert in alerts:
                for escalation_db in escalations:
                    # Check if escalation applies to this alert
                    if escalation_db.severity_filter and alert.severity not in escalation_db.severity_filter:
                        continue
                    if escalation_db.component_filter and alert.component not in escalation_db.component_filter:
                        continue
                    
                    # Check trigger condition
                    should_escalate = False
                    condition = EscalationTriggerCondition(escalation_db.trigger_condition)
                    
                    if condition == EscalationTriggerCondition.UNACKNOWLEDGED_DURATION:
                        duration_seconds = escalation_db.trigger_value.get("duration_seconds", 0)
                        if alert.acknowledged_at is None:
                            time_unacknowledged = (datetime.utcnow() - alert.triggered_at).total_seconds()
                            should_escalate = time_unacknowledged >= duration_seconds
                    
                    elif condition == EscalationTriggerCondition.SEVERITY:
                        required_severity = escalation_db.trigger_value.get("severity")
                        if required_severity and alert.severity == required_severity:
                            should_escalate = True
                    
                    elif condition == EscalationTriggerCondition.COUNT:
                        min_count = escalation_db.trigger_value.get("min_count", 0)
                        # Count similar alerts (would need to define similarity)
                        should_escalate = True  # Simplified
                    
                    if should_escalate:
                        # Note: send_alert_notification will be accessed via service composition
                        # For now, we'll log the escalation
                        logger.info(f"Alert {alert.id} should be escalated via {escalation_db.escalation_name}")
                        
                        # Update escalation stats
                        escalation_db.escalation_count += 1
                        escalation_db.last_escalated_at = datetime.utcnow()
                        async with get_session() as session:
                            session.add(escalation_db)
                            await session.commit()
                        
                        escalated.append(alert.id)
            
            return escalated
            
        except Exception as e:
            logger.error(f"Error checking and escalating alerts: {e}", exc_info=True)
            return []

