"""
Alert rules service
Handles alert rule creation, storage, and evaluation
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from app.database import get_session
from app.models.monitoring import AlertRuleDB
from app.schemas.monitoring import AlertRule, AlertCondition, AlertSeverity, SystemComponent
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class AlertRulesService(BaseMonitoringService):
    """Service for alert rules"""
    
    async def create_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        condition: AlertCondition,
        severity: AlertSeverity,
        component: SystemComponent,
        threshold_value: Optional[float] = None,
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        min_interval_seconds: Optional[int] = None,
        max_alerts_per_hour: Optional[int] = None,
        cooldown_period_seconds: Optional[int] = None,
        model_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> AlertRule:
        """Create a custom alert rule"""
        try:
            rule = AlertRule(
                rule_name=rule_name,
                description=description,
                metric_name=metric_name,
                condition=condition,
                threshold_value=threshold_value,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                severity=severity,
                component=component,
                min_interval_seconds=min_interval_seconds,
                max_alerts_per_hour=max_alerts_per_hour,
                cooldown_period_seconds=cooldown_period_seconds,
                model_id=model_id,
                deployment_id=deployment_id,
                is_active=True,
                created_by=created_by
            )
            
            # Store rule
            stored_id = await self.store_alert_rule(rule)
            rule.id = stored_id
            
            logger.info(f"Created alert rule {rule_name} for {metric_name}")
            return rule
            
        except Exception as e:
            logger.error(f"Error creating alert rule: {e}", exc_info=True)
            raise
    
    async def store_alert_rule(self, rule: AlertRule) -> str:
        """Store alert rule in database"""
        try:
            rule_db = AlertRuleDB(
                id=str(uuid.uuid4()),
                rule_name=rule.rule_name,
                description=rule.description,
                metric_name=rule.metric_name,
                condition=rule.condition.value,
                threshold_value=rule.threshold_value,
                threshold_min=rule.threshold_min,
                threshold_max=rule.threshold_max,
                severity=rule.severity.value,
                component=rule.component.value,
                min_interval_seconds=rule.min_interval_seconds,
                max_alerts_per_hour=rule.max_alerts_per_hour,
                cooldown_period_seconds=rule.cooldown_period_seconds,
                model_id=rule.model_id,
                deployment_id=rule.deployment_id,
                is_active=rule.is_active,
                last_triggered_at=rule.last_triggered_at,
                trigger_count=rule.trigger_count,
                created_by=rule.created_by
            )
            
            async with get_session() as session:
                session.add(rule_db)
                await session.commit()
                return rule_db.id
                
        except Exception as e:
            logger.error(f"Error storing alert rule: {e}")
            raise
    
    async def evaluate_alert_rule(
        self,
        rule_id: str,
        metric_value: float,
        model_id: Optional[str] = None
    ) -> bool:
        """Evaluate an alert rule against a metric value"""
        try:
            async with get_session() as session:
                rule_db = await session.get(AlertRuleDB, rule_id)
                if not rule_db or not rule_db.is_active:
                    return False
                
                # Check scope
                if rule_db.model_id and rule_db.model_id != model_id:
                    return False
                
                # Check frequency controls
                if rule_db.min_interval_seconds and rule_db.last_triggered_at:
                    time_since = (datetime.utcnow() - rule_db.last_triggered_at).total_seconds()
                    if time_since < rule_db.min_interval_seconds:
                        return False
                
                if rule_db.max_alerts_per_hour:
                    # Count alerts in last hour
                    hour_ago = datetime.utcnow() - timedelta(hours=1)
                    if rule_db.last_triggered_at and rule_db.last_triggered_at >= hour_ago:
                        if rule_db.trigger_count >= rule_db.max_alerts_per_hour:
                            return False
                
                # Evaluate condition
                condition_met = False
                condition = AlertCondition(rule_db.condition)
                
                if condition == AlertCondition.GT:
                    condition_met = metric_value > (rule_db.threshold_value or 0)
                elif condition == AlertCondition.LT:
                    condition_met = metric_value < (rule_db.threshold_value or 0)
                elif condition == AlertCondition.EQ:
                    condition_met = abs(metric_value - (rule_db.threshold_value or 0)) < 0.0001
                elif condition == AlertCondition.GTE:
                    condition_met = metric_value >= (rule_db.threshold_value or 0)
                elif condition == AlertCondition.LTE:
                    condition_met = metric_value <= (rule_db.threshold_value or 0)
                elif condition == AlertCondition.BETWEEN:
                    if rule_db.threshold_min is not None and rule_db.threshold_max is not None:
                        condition_met = rule_db.threshold_min <= metric_value <= rule_db.threshold_max
                elif condition == AlertCondition.NOT_BETWEEN:
                    if rule_db.threshold_min is not None and rule_db.threshold_max is not None:
                        condition_met = not (rule_db.threshold_min <= metric_value <= rule_db.threshold_max)
                
                if condition_met:
                    # Update rule stats
                    rule_db.trigger_count += 1
                    rule_db.last_triggered_at = datetime.utcnow()
                    await session.commit()
                
                return condition_met
                
        except Exception as e:
            logger.error(f"Error evaluating alert rule: {e}")
            return False

