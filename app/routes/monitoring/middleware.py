"""
Monitoring middleware utilities
Provides helper functions for logging predictions from other routes
"""

from typing import Dict, Any, Optional
from fastapi import Request
import logging

from app.services.monitoring_service import monitoring_service

logger = logging.getLogger(__name__)


async def log_prediction_middleware(
    request: Request,
    model_id: str,
    input_data: Dict[str, Any],
    output_data: Any,
    latency_ms: float,
    success: bool = True,
    error_message: Optional[str] = None
):
    """Middleware function to log predictions from other routes"""
    try:
        await monitoring_service.log_prediction(
            model_id=model_id,
            deployment_id=None,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            api_endpoint=str(request.url),
            success=success,
            error_message=error_message,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None
        )
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

