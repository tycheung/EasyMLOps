"""
Governance routes
Provides endpoints for data lineage, workflows, and compliance records
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
import logging

from app.schemas.monitoring import DataLineage, GovernanceWorkflow, ComplianceRecord
from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.post("/governance/lineage", response_model=Dict[str, str], status_code=201)
async def create_data_lineage(lineage: DataLineage):
    """Create data lineage record"""
    try:
        lineage_id = await monitoring_service.create_data_lineage(
            entity_type=lineage.lineage_type.value if hasattr(lineage.lineage_type, 'value') else str(lineage.lineage_type),
            entity_id=lineage.source_id,
            relationships={
                "source": {"id": lineage.source_id, "type": lineage.source_type},
                "target": {"id": lineage.target_id, "type": lineage.target_type},
                "relationship": lineage.relationship_type.value if hasattr(lineage.relationship_type, 'value') else str(lineage.relationship_type)
            },
            metadata={
                "source_metadata": lineage.source_metadata,
                "target_metadata": lineage.target_metadata,
                "relationship_metadata": lineage.relationship_metadata
            }
        )
        return {"id": lineage_id, "message": "Data lineage created successfully"}
    except Exception as e:
        logger.error(f"Error creating data lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/workflows", response_model=Dict[str, str], status_code=201)
async def create_governance_workflow(workflow: GovernanceWorkflow):
    """Create governance workflow"""
    try:
        workflow_id = await monitoring_service.create_governance_workflow(
            workflow_type=workflow.workflow_type.value if hasattr(workflow.workflow_type, 'value') else str(workflow.workflow_type),
            workflow_name=f"{workflow.workflow_type.value if hasattr(workflow.workflow_type, 'value') else str(workflow.workflow_type)}_workflow",
            description=f"Workflow for {workflow.resource_type}:{workflow.resource_id}",
            steps={"steps": []},
            config={
                "resource_type": workflow.resource_type,
                "resource_id": workflow.resource_id,
                "request_data": workflow.request_data,
                "policy_checks": workflow.policy_checks
            }
        )
        return {"id": workflow_id, "message": "Governance workflow created successfully"}
    except Exception as e:
        logger.error(f"Error creating governance workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/compliance", response_model=Dict[str, str], status_code=201)
async def create_compliance_record(record: ComplianceRecord):
    """Create compliance record"""
    try:
        record_id = await monitoring_service.create_compliance_record(
            compliance_type=record.compliance_type.value if hasattr(record.compliance_type, 'value') else str(record.compliance_type),
            record_type=record.record_type.value if hasattr(record.record_type, 'value') else str(record.record_type),
            entity_id=record.subject_id or "unknown",
            entity_type=record.subject_type or "unknown",
            description=record.description,
            metadata={
                "request_id": record.request_id,
                "requested_by": record.requested_by,
                "data_scope": record.data_scope,
                "additional_data": record.additional_data
            }
        )
        return {"id": record_id, "message": "Compliance record created successfully"}
    except Exception as e:
        logger.error(f"Error creating compliance record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/retention-policies", response_model=Dict[str, str], status_code=201)
async def create_data_retention_policy(policy: Dict[str, Any]):
    """Create data retention policy"""
    try:
        from app.schemas.monitoring import DataRetentionPolicy
        policy_obj = DataRetentionPolicy(**policy)
        policy_id = await monitoring_service.create_data_retention_policy(
            policy_name=policy_obj.policy_name,
            policy_description=policy_obj.policy_description,
            resource_type=policy_obj.resource_type,
            model_id=policy_obj.model_id,
            deployment_id=policy_obj.deployment_id,
            retention_period_days=policy_obj.retention_period_days,
            retention_condition=policy_obj.retention_condition,
            action_on_expiry=policy_obj.action_on_expiry,
            archive_location=policy_obj.archive_location,
            created_by=policy_obj.created_by
        )
        return {"id": policy_id, "message": "Data retention policy created successfully"}
    except Exception as e:
        logger.error(f"Error creating data retention policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

