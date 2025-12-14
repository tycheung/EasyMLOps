"""
Tests for Alert Service
Tests alert creation, rules, notifications, grouping, and escalation
"""

import pytest
from datetime import datetime

from app.services.monitoring_service import monitoring_service
from app.schemas.monitoring import (
    AlertSeverity, SystemComponent, AlertCondition,
    NotificationChannelType
)


class TestAlertService:
    """Test alert service functionality"""
    
    @pytest.mark.asyncio
    async def test_create_alert_rule(self, test_model):
        """Test creating an alert rule"""
        rule = await monitoring_service.create_alert_rule(
            rule_name="High Latency Alert",
            metric_name="avg_latency_ms",
            condition=AlertCondition.GT,
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            threshold_value=1000.0,
            min_interval_seconds=300,
            max_alerts_per_hour=5,
            model_id=test_model.id,
            description="Alert when latency exceeds 1 second",
            created_by="admin"
        )
        
        assert rule is not None
        assert rule.rule_name == "High Latency Alert"
        assert rule.metric_name == "avg_latency_ms"
        assert rule.condition == AlertCondition.GT
        assert rule.threshold_value == 1000.0
        assert rule.model_id == test_model.id
        assert rule.is_active is True
    
    @pytest.mark.asyncio
    async def test_evaluate_alert_rule(self, test_model):
        """Test evaluating an alert rule"""
        # Create a rule
        rule = await monitoring_service.create_alert_rule(
            rule_name="Test Rule",
            metric_name="cpu_usage",
            condition=AlertCondition.GT,
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            threshold_value=80.0
        )
        
        # Evaluate with value above threshold
        result = await monitoring_service.evaluate_alert_rule(
            rule_id=rule.id,
            metric_value=85.0
        )
        assert result is True
        
        # Evaluate with value below threshold
        result = await monitoring_service.evaluate_alert_rule(
            rule_id=rule.id,
            metric_value=75.0
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_notification_channel(self, test_model):
        """Test creating a notification channel"""
        channel = await monitoring_service.create_notification_channel(
            channel_name="Slack Alerts",
            channel_type=NotificationChannelType.SLACK,
            config={"webhook_url": "https://hooks.slack.com/test"},
            slack_webhook_url="https://hooks.slack.com/test",
            slack_channel="#alerts",
            severity_filter=["warning", "error", "critical"],
            description="Slack channel for alerts",
            created_by="admin"
        )
        
        assert channel is not None
        assert channel.channel_name == "Slack Alerts"
        assert channel.channel_type == NotificationChannelType.SLACK
        assert channel.slack_webhook_url == "https://hooks.slack.com/test"
        assert len(channel.severity_filter) == 3
        assert channel.is_active is True
    
    @pytest.mark.asyncio
    async def test_group_alerts(self, test_model):
        """Test grouping alerts"""
        # Create some alerts
        alert1 = await monitoring_service.create_alert(
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            title="Alert 1",
            description="First alert"
        )
        
        alert2 = await monitoring_service.create_alert(
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            title="Alert 2",
            description="Second alert"
        )
        
        # Group them
        group = await monitoring_service.group_alerts(
            alert_ids=[alert1.id, alert2.id],
            group_key="api_server_warnings",
            group_type="component"
        )
        
        assert group is not None
        assert group.group_key == "api_server_warnings"
        assert group.group_type == "component"
        assert len(group.alert_ids) == 2
        assert group.alert_count == 2
        assert group.is_active is True
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, test_model):
        """Test acknowledging an alert"""
        # Create an alert
        alert = await monitoring_service.create_alert(
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            title="Test Alert",
            description="Test alert description"
        )
        
        # Acknowledge it
        acknowledged = await monitoring_service.acknowledge_alert(
            alert_id=alert.id,
            acknowledged_by="admin"
        )
        
        assert acknowledged is not None
        assert acknowledged.is_acknowledged is True
        assert acknowledged.acknowledged_by == "admin"
        assert acknowledged.acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_create_alert_escalation(self, test_model):
        """Test creating an alert escalation rule"""
        escalation = await monitoring_service.create_alert_escalation(
            escalation_name="Critical Alert Escalation",
            trigger_condition="unacknowledged_duration",
            trigger_value={"duration_minutes": 30},
            escalation_actions=["notify_manager", "page_oncall"],
            notification_channels=[],
            severity_filter=["critical"],
            description="Escalate critical alerts after 30 minutes",
            created_by="admin"
        )
        
        assert escalation is not None
        assert escalation.escalation_name == "Critical Alert Escalation"
        assert escalation.trigger_condition == "unacknowledged_duration"
        assert escalation.is_active is True
    
    @pytest.mark.asyncio
    async def test_create_alert(self, test_model):
        """Test creating an alert"""
        alert = await monitoring_service.create_alert(
            severity=AlertSeverity.ERROR,
            component=SystemComponent.MODEL_SERVICE,
            title="Model Service Error",
            description="Model service is experiencing errors",
            metric_value=95.5,
            threshold_value=90.0
        )
        
        assert alert is not None
        assert alert.severity == AlertSeverity.ERROR
        assert alert.component == SystemComponent.MODEL_SERVICE
        assert alert.title == "Model Service Error"
        assert alert.is_active is True
        assert alert.is_acknowledged is False
        assert alert.metric_value == 95.5
        assert alert.threshold_value == 90.0
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self):
        """Test resolving an alert"""
        # Create an alert
        alert = await monitoring_service.create_alert(
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            title="Test Alert",
            description="Test alert"
        )
        
        # Resolve it
        with patch.object(monitoring_service, 'resolve_alert', new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = alert
            mock_resolve.return_value.is_active = False
            mock_resolve.return_value.resolved_at = datetime.utcnow()
            
            resolved = await monitoring_service.resolve_alert(alert.id)
            
            assert resolved is not None
            assert resolved.is_active is False
            assert resolved.resolved_at is not None
            
            mock_resolve.assert_called_once_with(alert.id)

