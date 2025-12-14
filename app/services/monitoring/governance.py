"""
Governance service
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import select, and_

from app.database import get_session
from app.models.monitoring import (
    DataLineageDB, GovernanceWorkflowDB, ComplianceRecordDB, DataRetentionPolicyDB
)
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class GovernanceService(BaseMonitoringService):
    """Service for governance"""
    
    async def create_data_lineage(
        self,
        entity_type: str,
        entity_id: str,
        relationships: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Create data lineage record"""
        try:
            lineage_id = str(uuid.uuid4())
            source = relationships.get("source", {})
            target = relationships.get("target", {})
            relationship = relationships.get("relationship", "unknown")
            
            lineage_db = DataLineageDB(
                id=lineage_id,
                lineage_type=entity_type,
                source_id=source.get("id", entity_id),
                source_type=source.get("type", "unknown"),
                source_metadata=metadata.get("source_metadata", {}),
                target_id=target.get("id", "unknown"),
                target_type=target.get("type", "unknown"),
                target_metadata=metadata.get("target_metadata", {}),
                relationship_type=relationship,
                relationship_metadata=metadata.get("relationship_metadata", {}),
                model_id=metadata.get("model_id"),
                deployment_id=metadata.get("deployment_id"),
                prediction_log_id=metadata.get("prediction_log_id")
            )
            
            async with get_session() as session:
                session.add(lineage_db)
                await session.commit()
                logger.info(f"Created data lineage {lineage_id} for {entity_type}:{entity_id}")
                return lineage_id
        except Exception as e:
            logger.error(f"Error creating data lineage: {e}", exc_info=True)
            raise
    
    async def create_governance_workflow(
        self,
        workflow_type: str,
        workflow_name: str,
        description: Optional[str],
        steps: Dict[str, Any],
        config: Dict[str, Any]
    ) -> str:
        """Create governance workflow"""
        try:
            workflow_id = str(uuid.uuid4())
            workflow_db = GovernanceWorkflowDB(
                id=workflow_id,
                workflow_type=workflow_type,
                workflow_status="pending",
                resource_type=config.get("resource_type", "unknown"),
                resource_id=config.get("resource_id", "unknown"),
                requested_by=config.get("requested_by"),
                request_data=config.get("request_data", {}),
                policy_checks=config.get("policy_checks", {}),
                comments=description,
                additional_data=config.get("additional_data", {})
            )
            
            async with get_session() as session:
                session.add(workflow_db)
                await session.commit()
                logger.info(f"Created governance workflow {workflow_id}: {workflow_name}")
                return workflow_id
        except Exception as e:
            logger.error(f"Error creating governance workflow: {e}", exc_info=True)
            raise
    
    async def create_compliance_record(
        self,
        compliance_type: str,
        record_type: str,
        entity_id: str,
        entity_type: str,
        description: Optional[str],
        metadata: Dict[str, Any]
    ) -> str:
        """Create compliance record"""
        try:
            record_id = str(uuid.uuid4())
            record_db = ComplianceRecordDB(
                id=record_id,
                compliance_type=compliance_type,
                record_type=record_type,
                subject_id=entity_id,
                subject_type=entity_type,
                request_id=metadata.get("request_id"),
                requested_by=metadata.get("requested_by"),
                data_scope=metadata.get("data_scope", {}),
                additional_data=metadata.get("additional_data", {}),
                status="pending"
            )
            
            async with get_session() as session:
                session.add(record_db)
                await session.commit()
                logger.info(f"Created compliance record {record_id} ({record_type}) for {entity_type}:{entity_id}")
                return record_id
        except Exception as e:
            logger.error(f"Error creating compliance record: {e}", exc_info=True)
            raise
    
    async def create_data_retention_policy(
        self,
        policy_name: str,
        policy_description: Optional[str],
        resource_type: str,
        model_id: Optional[str],
        deployment_id: Optional[str],
        retention_period_days: int,
        retention_condition: Dict[str, Any],
        action_on_expiry: str,
        archive_location: Optional[str],
        created_by: Optional[str]
    ) -> str:
        """Create data retention policy"""
        try:
            policy_id = str(uuid.uuid4())
            policy_db = DataRetentionPolicyDB(
                id=policy_id,
                policy_name=policy_name,
                policy_description=policy_description,
                resource_type=resource_type,
                model_id=model_id,
                deployment_id=deployment_id,
                retention_period_days=retention_period_days,
                retention_condition=retention_condition,
                action_on_expiry=action_on_expiry,
                archive_location=archive_location,
                is_active=True,
                created_by=created_by
            )
            
            async with get_session() as session:
                session.add(policy_db)
                await session.commit()
                logger.info(f"Created data retention policy {policy_id}: {policy_name}")
                return policy_id
        except Exception as e:
            logger.error(f"Error creating data retention policy: {e}", exc_info=True)
            raise
