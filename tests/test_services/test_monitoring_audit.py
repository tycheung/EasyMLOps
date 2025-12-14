"""
Tests for Audit Service
Tests audit logging functionality
"""

import pytest
from datetime import datetime

from app.services.monitoring_service import monitoring_service
from app.models.monitoring import AuditLogDB
from app.database import get_session
from sqlalchemy import select


class TestAuditService:
    """Test audit service functionality"""
    
    @pytest.mark.asyncio
    async def test_log_audit_event_basic(self):
        """Test basic audit event logging"""
        audit_id = await monitoring_service.log_audit_event(
            action="create",
            resource_type="model",
            resource_id="test_model_123",
            user_id="user_456"
        )
        
        assert audit_id is not None
        assert isinstance(audit_id, str)
        assert len(audit_id) == 36  # UUID format
    
    @pytest.mark.asyncio
    async def test_log_audit_event_full(self):
        """Test audit event logging with all fields"""
        audit_id = await monitoring_service.log_audit_event(
            action="update",
            resource_type="deployment",
            resource_id="deploy_789",
            user_id="user_456",
            session_id="session_123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            old_values={"status": "inactive"},
            new_values={"status": "active"},
            success=True,
            error_message=None,
            additional_data={"source": "api"}
        )
        
        assert audit_id is not None
        
        # Verify the audit log was stored
        async with get_session() as session:
            audit_log = await session.get(AuditLogDB, audit_id)
            assert audit_log is not None
            assert audit_log.action == "update"
            assert audit_log.resource_type == "deployment"
            assert audit_log.resource_id == "deploy_789"
            assert audit_log.user_id == "user_456"
            assert audit_log.success is True
            assert audit_log.old_values == {"status": "inactive"}
            assert audit_log.new_values == {"status": "active"}
            assert audit_log.additional_data == {"source": "api"}
    
    @pytest.mark.asyncio
    async def test_log_audit_event_failure(self):
        """Test audit event logging for failed operations"""
        audit_id = await monitoring_service.log_audit_event(
            action="delete",
            resource_type="model",
            resource_id="model_999",
            user_id="user_456",
            success=False,
            error_message="Permission denied"
        )
        
        assert audit_id is not None
        
        async with get_session() as session:
            audit_log = await session.get(AuditLogDB, audit_id)
            assert audit_log is not None
            assert audit_log.success is False
            assert audit_log.error_message == "Permission denied"
    
    @pytest.mark.asyncio
    async def test_log_audit_event_minimal(self):
        """Test audit event logging with minimal required fields"""
        audit_id = await monitoring_service.log_audit_event(
            action="view",
            resource_type="dashboard"
        )
        
        assert audit_id is not None
        
        async with get_session() as session:
            audit_log = await session.get(AuditLogDB, audit_id)
            assert audit_log is not None
            assert audit_log.action == "view"
            assert audit_log.resource_type == "dashboard"
            assert audit_log.resource_id is None
            assert audit_log.user_id is None
    
    @pytest.mark.asyncio
    async def test_log_audit_event_with_additional_data(self):
        """Test audit event logging with additional data"""
        audit_id = await monitoring_service.log_audit_event(
            action="predict",
            resource_type="model",
            resource_id="model_123",
            additional_data={
                "input_size": 100,
                "processing_time": 0.5,
                "model_version": "v1.2.3"
            }
        )
        
        assert audit_id is not None
        
        async with get_session() as session:
            audit_log = await session.get(AuditLogDB, audit_id)
            assert audit_log is not None
            assert audit_log.additional_data is not None
            assert audit_log.additional_data["input_size"] == 100
            assert audit_log.additional_data["processing_time"] == 0.5
            assert audit_log.additional_data["model_version"] == "v1.2.3"

