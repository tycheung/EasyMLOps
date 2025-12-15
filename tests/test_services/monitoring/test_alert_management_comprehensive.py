"""
Comprehensive tests for Alert Management Service
Tests alert grouping, escalation, and management functionality
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import uuid

from app.services.monitoring.alert_management import AlertManagementService
from app.models.monitoring import AlertDB, AlertGroupDB, AlertEscalationDB
from app.schemas.monitoring import AlertGroup, AlertEscalation, AlertSeverity, EscalationTriggerCondition


@pytest.fixture
def alert_management_service():
    """Create alert management service instance"""
    return AlertManagementService()


@pytest.fixture
def sample_alert(test_model, test_session):
    """Create a sample alert in database"""
    import uuid
    alert_db = AlertDB(
        id=str(uuid.uuid4()),
        severity="high",
        component="model_service",
        title="Performance Degradation",
        description="Test alert",
        triggered_at=datetime.utcnow(),
        affected_models=[test_model.id]
    )
    test_session.add(alert_db)
    test_session.commit()
    test_session.refresh(alert_db)
    return alert_db


class TestAlertManagementService:
    """Test alert management service methods"""
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_group_alerts_success(self, mock_get_session, alert_management_service, sample_alert):
        """Test grouping alerts successfully"""
        import uuid
        alert2_db = AlertDB(
            id=str(uuid.uuid4()),
            severity="high",
            component="model_service",
            title="Performance Degradation",
            description="Another alert",
            triggered_at=datetime.utcnow(),
            affected_models=sample_alert.affected_models if sample_alert.affected_models else []
        )
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[sample_alert, alert2_db])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        with patch.object(alert_management_service, 'store_alert_group', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = "group_123"
            
            group = await alert_management_service.group_alerts(
                alert_ids=[sample_alert.id, alert2_db.id],
                group_key="performance_issues",
                group_type="custom"
            )
            
            assert group is not None
            assert group.group_key == "performance_issues"
            assert len(group.alert_ids) == 2
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_group_alerts_no_valid_alerts(self, mock_get_session, alert_management_service):
        """Test grouping alerts with no valid alerts"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        with pytest.raises(ValueError, match="No valid alerts"):
            await alert_management_service.group_alerts(
                alert_ids=["nonexistent"],
                group_key="test_group"
            )
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_store_alert_group(self, mock_get_session, alert_management_service):
        """Test storing alert group"""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        group = AlertGroup(
            group_key="test_group",
            group_type="custom",
            alert_ids=["alert_1", "alert_2"],
            alert_count=2,
            first_alert_at=datetime.utcnow(),
            last_alert_at=datetime.utcnow(),
            is_active=True
        )
        
        group_id = await alert_management_service.store_alert_group(group)
        
        assert group_id is not None
        assert isinstance(group_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_create_alert_escalation(self, mock_get_session, alert_management_service):
        """Test creating alert escalation"""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        escalation = AlertEscalation(
            alert_id="alert_123",
            escalation_level=1,
            trigger_condition=EscalationTriggerCondition.TIME_BASED,
            trigger_value=30,  # minutes
            target_recipients=["admin@example.com"],
            is_active=True
        )
        
        escalation_id = await alert_management_service.create_alert_escalation(escalation)
        
        assert escalation_id is not None
        assert isinstance(escalation_id, str)
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_get_alert_groups(self, mock_get_session, alert_management_service):
        """Test getting alert groups"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        groups = await alert_management_service.get_alert_groups(is_active=True)
        
        assert isinstance(groups, list)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_get_alert_escalations(self, mock_get_session, alert_management_service):
        """Test getting alert escalations"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        escalations = await alert_management_service.get_alert_escalations(alert_id="alert_123")
        
        assert isinstance(escalations, list)
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_resolve_alert_group(self, mock_get_session, alert_management_service):
        """Test resolving an alert group"""
        mock_session = MagicMock()
        mock_group = MagicMock()
        mock_group.is_active = True
        mock_session.get = AsyncMock(return_value=mock_group)
        mock_session.commit = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_management_service.resolve_alert_group(
            group_id="group_123",
            resolved_by="admin"
        )
        
        assert result is True
        assert mock_group.is_active is False
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.monitoring.alert_management.get_session')
    async def test_resolve_alert_group_not_found(self, mock_get_session, alert_management_service):
        """Test resolving non-existent alert group"""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        result = await alert_management_service.resolve_alert_group("nonexistent", "admin")
        
        assert result is False

