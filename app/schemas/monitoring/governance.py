"""
Enterprise AI governance schemas
Contains schemas for data lineage, governance workflows, compliance, and data retention
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class LineageType(str, Enum):
    INPUT = "input"
    MODEL_VERSION = "model_version"
    PREDICTION = "prediction"
    DATA_SOURCE = "data_source"


class RelationshipType(str, Enum):
    USED_BY = "used_by"
    DERIVED_FROM = "derived_from"
    TRAINED_ON = "trained_on"
    PREDICTED_BY = "predicted_by"


class DataLineage(BaseModel):
    """Schema for data lineage tracking"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique lineage ID")
    
    # Lineage type
    lineage_type: LineageType = Field(..., description="Type of lineage")
    
    # Source information
    source_id: str = Field(..., description="ID of source entity")
    source_type: str = Field(..., description="Type of source")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source metadata")
    
    # Target information
    target_id: str = Field(..., description="ID of target entity")
    target_type: str = Field(..., description="Type of target")
    target_metadata: Dict[str, Any] = Field(default_factory=dict, description="Target metadata")
    
    # Relationship information
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    relationship_metadata: Dict[str, Any] = Field(default_factory=dict, description="Relationship metadata")
    
    # Context
    model_id: Optional[str] = Field(None, description="Related model ID")
    deployment_id: Optional[str] = Field(None, description="Related deployment ID")
    prediction_log_id: Optional[str] = Field(None, description="Related prediction log ID")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class WorkflowType(str, Enum):
    MODEL_APPROVAL = "model_approval"
    DEPLOYMENT_APPROVAL = "deployment_approval"
    CHANGE_MANAGEMENT = "change_management"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class GovernanceWorkflow(BaseModel):
    """Schema for governance workflows"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique workflow ID")
    
    # Workflow information
    workflow_type: WorkflowType = Field(..., description="Type of workflow")
    workflow_status: WorkflowStatus = Field(WorkflowStatus.PENDING, description="Workflow status")
    
    # Resource information
    resource_type: str = Field(..., description="Type of resource")
    resource_id: str = Field(..., description="Resource ID")
    
    # Approval information
    requested_by: Optional[str] = Field(None, description="User who requested")
    requested_at: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    approved_by: Optional[str] = Field(None, description="User who approved")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    rejected_by: Optional[str] = Field(None, description="User who rejected")
    rejected_at: Optional[datetime] = Field(None, description="Rejection timestamp")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason")
    
    # Workflow data
    request_data: Dict[str, Any] = Field(default_factory=dict, description="Request data")
    approval_data: Dict[str, Any] = Field(default_factory=dict, description="Approval data")
    policy_checks: Dict[str, Any] = Field(default_factory=dict, description="Policy check results")
    
    # Metadata
    comments: Optional[str] = Field(None, description="Comments")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class ComplianceType(str, Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    CUSTOM = "custom"


class ComplianceRecordType(str, Enum):
    DATA_ACCESS = "data_access"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    AUDIT_EXPORT = "audit_export"


class ComplianceRecordStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ComplianceRecord(BaseModel):
    """Schema for compliance records"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique record ID")
    
    # Compliance information
    compliance_type: ComplianceType = Field(..., description="Type of compliance")
    record_type: ComplianceRecordType = Field(..., description="Type of record")
    
    # Subject information
    subject_id: Optional[str] = Field(None, description="Subject identifier")
    subject_type: Optional[str] = Field(None, description="Type of subject")
    
    # Request information
    request_id: Optional[str] = Field(None, description="External request ID")
    requested_by: Optional[str] = Field(None, description="User who requested")
    requested_at: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    
    # Processing information
    status: ComplianceRecordStatus = Field(ComplianceRecordStatus.PENDING, description="Processing status")
    processed_by: Optional[str] = Field(None, description="User who processed")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Data information
    data_scope: Dict[str, Any] = Field(default_factory=dict, description="Data scope")
    data_retention_policy: Optional[str] = Field(None, description="Applied retention policy")
    deletion_scope: Dict[str, Any] = Field(default_factory=dict, description="Deletion scope")
    
    # Results
    records_affected: Optional[int] = Field(None, description="Number of records affected", ge=0)
    export_location: Optional[str] = Field(None, description="Export location")
    verification_hash: Optional[str] = Field(None, description="Verification hash")
    
    # Metadata
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
    error_message: Optional[str] = Field(None, description="Error message")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class DataRetentionPolicy(BaseModel):
    """Schema for data retention policies"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique policy ID")
    
    # Policy information
    policy_name: str = Field(..., description="Policy name")
    policy_description: Optional[str] = Field(None, description="Policy description")
    
    # Scope
    resource_type: str = Field(..., description="Type of resource")
    model_id: Optional[str] = Field(None, description="Model ID if applicable")
    deployment_id: Optional[str] = Field(None, description="Deployment ID if applicable")
    
    # Retention rules
    retention_period_days: int = Field(..., description="Retention period in days", ge=1)
    retention_condition: Dict[str, Any] = Field(default_factory=dict, description="Additional conditions")
    
    # Action on expiry
    action_on_expiry: str = Field("delete", description="Action on expiry: delete, archive, anonymize")
    archive_location: Optional[str] = Field(None, description="Archive location")
    
    # Status
    is_active: bool = Field(True, description="Whether policy is active")
    last_applied_at: Optional[datetime] = Field(None, description="Last application timestamp")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="User who created policy")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

