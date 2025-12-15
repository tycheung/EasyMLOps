"""
Governance routes
Provides endpoints for data lineage, workflows, and compliance records
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Body, Query
import logging

from app.schemas.monitoring import DataLineage, GovernanceWorkflow, ComplianceRecord
from app.services.monitoring_service import monitoring_service
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.get("/governance/lineage", response_model=List[Dict[str, Any]])
async def list_data_lineage(
    lineage_type: Optional[str] = Query(None, description="Filter by lineage type"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, description="Maximum number of lineage records to return")
):
    """List all data lineage records"""
    try:
        async with get_session() as session:
            from app.models.monitoring import DataLineageDB
            from sqlalchemy import select, desc
            
            stmt = select(DataLineageDB)
            if lineage_type:
                stmt = stmt.where(DataLineageDB.lineage_type == lineage_type)
            if model_id:
                stmt = stmt.where(DataLineageDB.model_id == model_id)
            
            stmt = stmt.order_by(desc(DataLineageDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            lineage_db = result.scalars().all()
            
            return [
                {
                    "id": lineage.id,
                    "lineage_type": lineage.lineage_type,
                    "source_id": lineage.source_id,
                    "source_type": lineage.source_type,
                    "source_metadata": lineage.source_metadata,
                    "target_id": lineage.target_id,
                    "target_type": lineage.target_type,
                    "target_metadata": lineage.target_metadata,
                    "relationship_type": lineage.relationship_type,
                    "relationship_metadata": lineage.relationship_metadata,
                    "model_id": lineage.model_id,
                    "deployment_id": lineage.deployment_id,
                    "created_at": lineage.created_at.isoformat() if lineage.created_at else None
                }
                for lineage in lineage_db
            ]
    except Exception as e:
        logger.error(f"Error listing data lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/governance/workflows", response_model=List[Dict[str, Any]])
async def list_governance_workflows(
    workflow_type: Optional[str] = Query(None, description="Filter by workflow type"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    limit: int = Query(100, description="Maximum number of workflows to return")
):
    """List all governance workflows"""
    try:
        async with get_session() as session:
            from app.models.monitoring import GovernanceWorkflowDB
            from sqlalchemy import select, desc
            
            stmt = select(GovernanceWorkflowDB)
            if workflow_type:
                stmt = stmt.where(GovernanceWorkflowDB.workflow_type == workflow_type)
            if resource_type:
                stmt = stmt.where(GovernanceWorkflowDB.resource_type == resource_type)
            
            stmt = stmt.order_by(desc(GovernanceWorkflowDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            workflows_db = result.scalars().all()
            
            return [
                {
                    "id": workflow.id,
                    "workflow_type": workflow.workflow_type,
                    "workflow_status": workflow.workflow_status,
                    "resource_type": workflow.resource_type,
                    "resource_id": workflow.resource_id,
                    "requested_by": workflow.requested_by,
                    "requested_at": workflow.requested_at.isoformat() if workflow.requested_at else None,
                    "approved_by": workflow.approved_by,
                    "approved_at": workflow.approved_at.isoformat() if workflow.approved_at else None,
                    "created_at": workflow.created_at.isoformat() if workflow.created_at else None
                }
                for workflow in workflows_db
            ]
    except Exception as e:
        logger.error(f"Error listing governance workflows: {e}")
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


@router.get("/governance/compliance", response_model=List[Dict[str, Any]])
async def list_compliance_records(
    compliance_type: Optional[str] = Query(None, description="Filter by compliance type"),
    record_type: Optional[str] = Query(None, description="Filter by record type"),
    limit: int = Query(100, description="Maximum number of records to return")
):
    """List all compliance records"""
    try:
        async with get_session() as session:
            from app.models.monitoring import ComplianceRecordDB
            from sqlalchemy import select, desc
            
            stmt = select(ComplianceRecordDB)
            if compliance_type:
                stmt = stmt.where(ComplianceRecordDB.compliance_type == compliance_type)
            if record_type:
                stmt = stmt.where(ComplianceRecordDB.record_type == record_type)
            
            stmt = stmt.order_by(desc(ComplianceRecordDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            records_db = result.scalars().all()
            
            return [
                {
                    "id": record.id,
                    "compliance_type": record.compliance_type,
                    "record_type": record.record_type,
                    "subject_id": record.subject_id,
                    "subject_type": record.subject_type,
                    "request_id": record.request_id,
                    "requested_by": record.requested_by,
                    "status": record.status,
                    "created_at": record.created_at.isoformat() if record.created_at else None
                }
                for record in records_db
            ]
    except Exception as e:
        logger.error(f"Error listing compliance records: {e}")
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


@router.get("/governance/retention-policies", response_model=List[Dict[str, Any]])
async def list_retention_policies(
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, description="Maximum number of policies to return")
):
    """List all data retention policies"""
    try:
        async with get_session() as session:
            from app.models.monitoring import DataRetentionPolicyDB
            from sqlalchemy import select, desc
            
            stmt = select(DataRetentionPolicyDB)
            if resource_type:
                stmt = stmt.where(DataRetentionPolicyDB.resource_type == resource_type)
            if model_id:
                stmt = stmt.where(DataRetentionPolicyDB.model_id == model_id)
            
            stmt = stmt.order_by(desc(DataRetentionPolicyDB.created_at)).limit(limit)
            result = await session.execute(stmt)
            policies_db = result.scalars().all()
            
            return [
                {
                    "id": policy.id,
                    "policy_name": policy.policy_name,
                    "policy_description": policy.policy_description,
                    "resource_type": policy.resource_type,
                    "model_id": policy.model_id,
                    "deployment_id": policy.deployment_id,
                    "retention_period_days": policy.retention_period_days,
                    "retention_condition": policy.retention_condition,
                    "action_on_expiry": policy.action_on_expiry,
                    "archive_location": policy.archive_location,
                    "created_at": policy.created_at.isoformat() if policy.created_at else None,
                    "created_by": policy.created_by
                }
                for policy in policies_db
            ]
    except Exception as e:
        logger.error(f"Error listing retention policies: {e}")
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

