"""
Alert management routes
Provides endpoints for alerts, alert rules, and notifications
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
import logging

from app.schemas.monitoring import Alert, AlertRule, NotificationChannel
from app.services.monitoring_service import monitoring_service
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    active_only: bool = Query(True),
    severity: Optional[str] = Query(None),
    component: Optional[str] = Query(None),
    limit: int = Query(50)
):
    """Get system alerts"""
    try:
        async with get_session() as session:
            from app.models.monitoring import AlertDB
            from sqlalchemy import desc
            from sqlmodel import select

            stmt = select(AlertDB)

            if active_only:
                stmt = stmt.where(AlertDB.is_active == True)
            if severity:
                stmt = stmt.where(AlertDB.severity == severity)
            if component:
                stmt = stmt.where(AlertDB.component == component)

            stmt = stmt.order_by(desc(AlertDB.triggered_at)).limit(limit)
            result = await session.execute(stmt)
            alerts_db = result.scalars().all()

            return [
                Alert(
                    id=alert.id,
                    severity=alert.severity,
                    component=alert.component,
                    title=alert.title,
                    description=alert.description,
                    triggered_at=alert.triggered_at,
                    resolved_at=alert.resolved_at,
                    acknowledged_at=alert.acknowledged_at,
                    acknowledged_by=alert.acknowledged_by,
                    metric_value=alert.metric_value,
                    threshold_value=alert.threshold_value,
                    affected_models=alert.affected_models or [],
                    is_active=alert.is_active,
                    is_acknowledged=alert.is_acknowledged
                )
                for alert in alerts_db
            ]
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts", response_model=Dict[str, str], status_code=201)
async def create_alert(
    severity: str = Query(..., description="Alert severity"),
    component: str = Query(..., description="System component"),
    title: str = Query(..., description="Alert title"),
    description: str = Query(..., description="Alert description"),
    metric_value: Optional[float] = Query(None, description="Metric value"),
    threshold_value: Optional[float] = Query(None, description="Threshold value"),
    affected_models: Optional[List[str]] = Query(None, description="Affected model IDs")
):
    """Create a new alert"""
    try:
        alert_id = await monitoring_service.create_alert(
            severity=severity,
            component=component,
            title=title,
            description=description,
            metric_value=metric_value,
            threshold_value=threshold_value,
            affected_models=affected_models or []
        )
        return {"id": alert_id, "message": "Alert created successfully"}
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve", response_model=Dict[str, bool])
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    try:
        success = await monitoring_service.resolve_alert(alert_id=alert_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge", response_model=Dict[str, bool])
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: Optional[str] = Query(None, description="User acknowledging the alert")
):
    """Acknowledge an alert"""
    try:
        success = await monitoring_service.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=acknowledged_by
        )
        return {"success": success}
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/check", response_model=List[Alert])
async def check_and_create_alerts():
    """Check system metrics and create alerts if thresholds are exceeded"""
    try:
        alerts = await monitoring_service.check_and_create_alerts()
        return alerts
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alert-rules", response_model=Dict[str, str], status_code=201)
async def create_alert_rule(rule: AlertRule):
    """Create a new alert rule"""
    try:
        rule_id = await monitoring_service.create_alert_rule(
            rule_name=rule.rule_name,
            description=rule.description,
            metric_name=rule.metric_name,
            condition=rule.condition,
            threshold_value=rule.threshold_value,
            threshold_min=rule.threshold_min,
            threshold_max=rule.threshold_max,
            severity=rule.severity,
            component=rule.component,
            model_id=rule.model_id,
            deployment_id=rule.deployment_id,
            min_interval_seconds=rule.min_interval_seconds,
            max_alerts_per_hour=rule.max_alerts_per_hour,
            cooldown_period_seconds=rule.cooldown_period_seconds,
            is_active=rule.is_active,
            created_by=rule.created_by
        )
        return {"id": rule_id, "message": "Alert rule created successfully"}
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notifications/channels", response_model=Dict[str, str], status_code=201)
async def create_notification_channel(channel: NotificationChannel):
    """Create a new notification channel"""
    try:
        channel_id = await monitoring_service.create_notification_channel(
            channel_name=channel.channel_name,
            channel_type=channel.channel_type,
            description=channel.description,
            config=channel.config,
            email_recipients=channel.email_recipients,
            slack_webhook_url=channel.slack_webhook_url,
            webhook_url=channel.webhook_url
        )
        return {"id": channel_id, "message": "Notification channel created successfully"}
    except Exception as e:
        logger.error(f"Error creating notification channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notifications/send", response_model=Dict[str, bool])
async def send_alert_notification(
    alert_id: str = Query(..., description="Alert ID"),
    channel_id: Optional[str] = Query(None, description="Notification channel ID")
):
    """Send alert notification manually"""
    try:
        success = await monitoring_service.send_alert_notification(
            alert_id=alert_id,
            channel_id=channel_id
        )
        return {"success": success}
    except Exception as e:
        logger.error(f"Error sending alert notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/group", response_model=Dict[str, str], status_code=201)
async def group_alerts(
    alert_ids: List[str] = Body(..., description="List of alert IDs to group"),
    group_name: Optional[str] = Body(None, description="Group name")
):
    """Group alerts together"""
    try:
        group_id = await monitoring_service.group_alerts(
            alert_ids=alert_ids,
            group_name=group_name
        )
        return {"id": group_id, "message": "Alerts grouped successfully"}
    except Exception as e:
        logger.error(f"Error grouping alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/escalations", response_model=Dict[str, str], status_code=201)
async def create_alert_escalation(
    escalation_name: str = Body(..., description="Escalation name"),
    trigger_condition: str = Body(..., description="Trigger condition"),
    trigger_value: Dict[str, Any] = Body(..., description="Trigger value"),
    escalation_actions: List[str] = Body(..., description="Escalation actions"),
    severity_filter: Optional[List[str]] = Body(None, description="Severity filter"),
    component_filter: Optional[List[str]] = Body(None, description="Component filter"),
    notification_channels: Optional[List[str]] = Body(None, description="Notification channels"),
    description: Optional[str] = Body(None, description="Description"),
    created_by: Optional[str] = Body(None, description="Created by")
):
    """Create alert escalation"""
    try:
        from app.schemas.monitoring import AlertEscalation, EscalationTriggerCondition
        # Convert string to enum
        trigger_condition_enum = EscalationTriggerCondition(trigger_condition)
        escalation = await monitoring_service.create_alert_escalation(
            escalation_name=escalation_name,
            trigger_condition=trigger_condition_enum,
            trigger_value=trigger_value,
            escalation_actions=escalation_actions,
            severity_filter=severity_filter,
            component_filter=component_filter,
            notification_channels=notification_channels,
            description=description,
            created_by=created_by
        )
        return {"id": escalation.id, "message": "Alert escalation created successfully"}
    except Exception as e:
        logger.error(f"Error creating alert escalation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/escalate", response_model=Dict[str, List[str]])
async def check_and_escalate_alerts():
    """Check and escalate alerts based on escalation rules"""
    try:
        escalated_ids = await monitoring_service.check_and_escalate_alerts()
        return {"escalated_alert_ids": escalated_ids}
    except Exception as e:
        logger.error(f"Error checking and escalating alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

