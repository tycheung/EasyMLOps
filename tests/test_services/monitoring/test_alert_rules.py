"""
Comprehensive tests for Alert Rules Service
Tests alert rule creation, storage, and evaluation
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import uuid

from app.services.monitoring.alert_rules import AlertRulesService
from app.models.monitoring import AlertRuleDB
from app.schemas.monitoring import AlertRule, AlertCondition, AlertSeverity, SystemComponent


@pytest.fixture
def alert_rules_service():
    """Create alert rules service instance"""
    return AlertRulesService()


@pytest.fixture
def sample_alert_rule(test_model, test_session):
    """Create a sample alert rule in database"""
    rule_db = AlertRuleDB(
        id=str(uuid.uuid4()),
        rule_name="test_rule",
        metric_name="error_rate",
        condition="gt",
        threshold_value=5.0,
        severity="error",
        component="model_service",
        model_id=test_model.id,
        is_active=True,
        created_at=datetime.utcnow()
    )
    test_session.add(rule_db)
    test_session.commit()
    test_session.refresh(rule_db)
    return rule_db


class TestAlertRulesService:
    """Test alert rules service methods"""
    
    @pytest.mark.asyncio
    async def test_create_alert_rule_success(self, alert_rules_service, test_model):
        """Test creating alert rule successfully"""
        rule = await alert_rules_service.create_alert_rule(
            rule_name="error_rate_high",
            metric_name="error_rate",
            condition=AlertCondition.GT,
            severity=AlertSeverity.ERROR,
            component=SystemComponent.MODEL_SERVICE,
            threshold_value=5.0,
            model_id=test_model.id,
            description="Alert when error rate exceeds 5%",
            created_by="admin"
        )
        
        assert rule is not None
        assert rule.rule_name == "error_rate_high"
        assert rule.metric_name == "error_rate"
        assert rule.condition == AlertCondition.GT
        assert rule.threshold_value == 5.0
        assert rule.is_active is True
        assert rule.id is not None
    
    @pytest.mark.asyncio
    async def test_create_alert_rule_with_range(self, alert_rules_service, test_model):
        """Test creating alert rule with threshold range"""
        rule = await alert_rules_service.create_alert_rule(
            rule_name="latency_range",
            metric_name="latency_ms",
            condition=AlertCondition.BETWEEN,
            severity=AlertSeverity.WARNING,
            component=SystemComponent.API_SERVER,
            threshold_min=100.0,
            threshold_max=500.0,
            model_id=test_model.id
        )
        
        assert rule.threshold_min == 100.0
        assert rule.threshold_max == 500.0
        assert rule.condition == AlertCondition.BETWEEN
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_store_alert_rule(self, mock_get_session, alert_rules_service):
        """Test storing alert rule"""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        rule = AlertRule(
            rule_name="test_rule",
            metric_name="error_rate",
            condition=AlertCondition.GT,
            severity=AlertSeverity.ERROR,
            component=SystemComponent.MODEL_SERVICE,
            threshold_value=5.0,
            is_active=True
        )
        
        rule_id = await alert_rules_service.store_alert_rule(rule)
        
        assert rule_id is not None
        assert isinstance(rule_id, str)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_gt_condition_met(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with GT condition met"""
        sample_alert_rule.condition = "gt"
        sample_alert_rule.threshold_value = 5.0
        sample_alert_rule.last_triggered_at = None
        sample_alert_rule.trigger_count = 0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=10.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
        assert sample_alert_rule.trigger_count == 1
        assert sample_alert_rule.last_triggered_at is not None
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_gt_condition_not_met(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with GT condition not met"""
        sample_alert_rule.condition = "gt"
        sample_alert_rule.threshold_value = 5.0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=3.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_lt_condition(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with LT condition"""
        sample_alert_rule.condition = "lt"
        sample_alert_rule.threshold_value = 5.0
        sample_alert_rule.last_triggered_at = None
        sample_alert_rule.trigger_count = 0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=3.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_between_condition(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with BETWEEN condition"""
        sample_alert_rule.condition = "between"
        sample_alert_rule.threshold_min = 5.0
        sample_alert_rule.threshold_max = 10.0
        sample_alert_rule.last_triggered_at = None
        sample_alert_rule.trigger_count = 0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Value within range
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=7.5,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
        
        # Value outside range
        sample_alert_rule.last_triggered_at = None
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=15.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_not_between_condition(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with NOT_BETWEEN condition"""
        sample_alert_rule.condition = "not_between"
        sample_alert_rule.threshold_min = 5.0
        sample_alert_rule.threshold_max = 10.0
        sample_alert_rule.last_triggered_at = None
        sample_alert_rule.trigger_count = 0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Value outside range (should trigger)
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=15.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_eq_condition(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with EQ condition"""
        sample_alert_rule.condition = "eq"
        sample_alert_rule.threshold_value = 5.0
        sample_alert_rule.last_triggered_at = None
        sample_alert_rule.trigger_count = 0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Exact match
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=5.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
        
        # Close but not exact (within tolerance)
        sample_alert_rule.last_triggered_at = None
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=5.00005,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_inactive(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating inactive alert rule"""
        sample_alert_rule.is_active = False
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=10.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_not_found(self, mock_get_session, alert_rules_service):
        """Test evaluating non-existent alert rule"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id="nonexistent",
            metric_value=10.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_model_id_mismatch(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with model ID mismatch"""
        sample_alert_rule.model_id = "model_123"
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=10.0,
            model_id="different_model"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_min_interval(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with min interval restriction"""
        sample_alert_rule.condition = "gt"
        sample_alert_rule.threshold_value = 5.0
        sample_alert_rule.min_interval_seconds = 60
        sample_alert_rule.last_triggered_at = datetime.utcnow() - timedelta(seconds=30)  # 30 seconds ago
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=10.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_max_alerts_per_hour(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with max alerts per hour restriction"""
        sample_alert_rule.condition = "gt"
        sample_alert_rule.threshold_value = 5.0
        sample_alert_rule.max_alerts_per_hour = 5
        sample_alert_rule.trigger_count = 5
        sample_alert_rule.last_triggered_at = datetime.utcnow() - timedelta(minutes=30)  # Within last hour
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=10.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_gte_condition(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with GTE condition"""
        sample_alert_rule.condition = "gte"
        sample_alert_rule.threshold_value = 5.0
        sample_alert_rule.last_triggered_at = None
        sample_alert_rule.trigger_count = 0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Equal to threshold
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=5.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
        
        # Greater than threshold
        sample_alert_rule.last_triggered_at = None
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=6.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_rules.get_session')
    async def test_evaluate_alert_rule_lte_condition(self, mock_get_session, alert_rules_service, sample_alert_rule):
        """Test evaluating alert rule with LTE condition"""
        sample_alert_rule.condition = "lte"
        sample_alert_rule.threshold_value = 5.0
        sample_alert_rule.last_triggered_at = None
        sample_alert_rule.trigger_count = 0
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=sample_alert_rule)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Equal to threshold
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=5.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True
        
        # Less than threshold
        sample_alert_rule.last_triggered_at = None
        result = await alert_rules_service.evaluate_alert_rule(
            rule_id=sample_alert_rule.id,
            metric_value=4.0,
            model_id=sample_alert_rule.model_id
        )
        
        assert result is True

