"""
Tests for Enterprise AI Governance Service
Tests data lineage, governance workflows, compliance records, and data retention policies
"""

import pytest
from datetime import datetime

from app.services.monitoring_service import monitoring_service
from app.schemas.monitoring import (
    LineageType, RelationshipType, WorkflowType, WorkflowStatus,
    ComplianceType, ComplianceRecordType, ComplianceRecordStatus
)


class TestGovernanceService:
    """Test governance service functionality"""
    
    @pytest.mark.asyncio
    async def test_create_data_lineage(self, test_model):
        """Test creating data lineage"""
        lineage = await monitoring_service.create_data_lineage(
            lineage_type=LineageType.PREDICTION,
            source_id="request_123",
            source_type="request",
            target_id="prediction_456",
            target_type="prediction_log",
            relationship_type=RelationshipType.PREDICTED_BY,
            model_id=test_model.id
        )
        
        assert lineage is not None
        assert lineage.lineage_type == LineageType.PREDICTION
        assert lineage.source_id == "request_123"
        assert lineage.target_id == "prediction_456"
        assert lineage.relationship_type == RelationshipType.PREDICTED_BY
        assert lineage.model_id == test_model.id
    
    @pytest.mark.asyncio
    async def test_create_governance_workflow(self, test_model):
        """Test creating a governance workflow"""
        workflow = await monitoring_service.create_governance_workflow(
            workflow_type=WorkflowType.MODEL_APPROVAL,
            resource_type="model",
            resource_id=test_model.id,
            requested_by="test_user",
            request_data={"reason": "New model version"},
            comments="Please approve this model"
        )
        
        assert workflow is not None
        assert workflow.workflow_type == WorkflowType.MODEL_APPROVAL
        assert workflow.resource_type == "model"
        assert workflow.resource_id == test_model.id
        assert workflow.workflow_status == WorkflowStatus.PENDING
        assert workflow.requested_by == "test_user"
        assert workflow.comments == "Please approve this model"
    
    @pytest.mark.asyncio
    async def test_create_compliance_record(self, test_model):
        """Test creating a compliance record"""
        record = await monitoring_service.create_compliance_record(
            compliance_type=ComplianceType.GDPR,
            record_type=ComplianceRecordType.DATA_DELETION,
            subject_id="user_123",
            subject_type="user",
            requested_by="user_123",
            data_scope={"model_id": test_model.id}
        )
        
        assert record is not None
        assert record.compliance_type == ComplianceType.GDPR
        assert record.record_type == ComplianceRecordType.DATA_DELETION
        assert record.subject_id == "user_123"
        assert record.status == ComplianceRecordStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_create_data_retention_policy(self, test_model):
        """Test creating a data retention policy"""
        policy = await monitoring_service.create_data_retention_policy(
            policy_name="Prediction Log Retention",
            resource_type="prediction_log",
            retention_period_days=90,
            action_on_expiry="delete",
            model_id=test_model.id,
            policy_description="Retain prediction logs for 90 days",
            created_by="admin"
        )
        
        assert policy is not None
        assert policy.policy_name == "Prediction Log Retention"
        assert policy.resource_type == "prediction_log"
        assert policy.retention_period_days == 90
        assert policy.action_on_expiry == "delete"
        assert policy.model_id == test_model.id
        assert policy.is_active is True

