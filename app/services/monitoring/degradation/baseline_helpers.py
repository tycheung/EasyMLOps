"""
Baseline performance helpers
Handles retrieving baseline performance metrics for comparison
"""

import logging
from typing import Any, Dict, Optional

from sqlalchemy import select, desc

from app.database import get_session
from app.models.monitoring import ModelPerformanceHistoryDB

logger = logging.getLogger(__name__)


async def get_baseline_performance(
    model_id: str,
    deployment_id: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """Get baseline performance metrics for comparison"""
    try:
        async with get_session() as session:
            stmt = select(ModelPerformanceHistoryDB).where(
                ModelPerformanceHistoryDB.model_id == model_id,
                ModelPerformanceHistoryDB.performance_degraded == False
            )
            if deployment_id:
                stmt = stmt.where(ModelPerformanceHistoryDB.deployment_id == deployment_id)
            
            stmt = stmt.order_by(desc(ModelPerformanceHistoryDB.timestamp)).limit(1)
            result = await session.execute(stmt)
            baseline = result.scalar_one_or_none()
            
            if baseline:
                return {
                    "accuracy": baseline.accuracy,
                    "f1_score": baseline.f1_score,
                    "mae": baseline.mae,
                    "r2_score": baseline.r2_score
                }
            return None
    except Exception as e:
        logger.error(f"Error getting baseline performance: {e}")
        return None

