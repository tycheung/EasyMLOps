"""
Audit log routes
Provides endpoints for retrieving audit logs
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.monitoring import AuditLog
from app.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


@router.get("/audit", response_model=List[AuditLog])
async def get_audit_logs(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    action: Optional[str] = Query(None, description="Filter by action"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get audit logs"""
    try:
        async with get_session() as session:
            from app.models.monitoring import AuditLogDB
            from sqlalchemy import desc
            from sqlmodel import select
            
            stmt = select(AuditLogDB)
            
            if entity_type:
                stmt = stmt.where(AuditLogDB.resource_type == entity_type)
            if entity_id:
                stmt = stmt.where(AuditLogDB.resource_id == entity_id)
            if action:
                stmt = stmt.where(AuditLogDB.action == action)
            if start_time:
                stmt = stmt.where(AuditLogDB.timestamp >= start_time)
            if end_time:
                stmt = stmt.where(AuditLogDB.timestamp <= end_time)
            
            stmt = stmt.order_by(desc(AuditLogDB.timestamp)).limit(limit)
            result = await session.execute(stmt)
            logs_db = result.scalars().all()
            
            return [
                AuditLog(
                    id=log.id,
                    resource_type=log.resource_type,
                    resource_id=log.resource_id,
                    action=log.action,
                    user_id=log.user_id,
                    session_id=log.session_id,
                    timestamp=log.timestamp,
                    ip_address=log.ip_address,
                    user_agent=log.user_agent,
                    old_values=log.old_values,
                    new_values=log.new_values,
                    success=log.success,
                    error_message=log.error_message,
                    additional_data=log.additional_data or {}
                )
                for log in logs_db
            ]
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

