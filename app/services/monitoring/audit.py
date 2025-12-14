"""
Audit service
Handles audit logging
"""

import logging
import uuid
from typing import Any, Dict, Optional

from app.database import get_session
from app.models.monitoring import AuditLogDB
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class AuditService(BaseMonitoringService):
    """Service for audit logging"""
    
    async def log_audit_event(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log audit event for compliance and tracking"""
        try:
            audit_log = AuditLogDB(
                id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                old_values=old_values,
                new_values=new_values,
                success=success,
                error_message=error_message,
                additional_data=additional_data or {}
            )
            
            async with get_session() as session:
                session.add(audit_log)
                await session.commit()
                
                logger.info(f"Logged audit event: {action} on {resource_type} {resource_id}")
                return audit_log.id
                
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            raise

