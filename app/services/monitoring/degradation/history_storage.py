"""
Performance history storage
Handles storing performance history in the database
"""

import logging
import uuid
from typing import Any

from app.database import get_session
from app.models.monitoring import ModelPerformanceHistoryDB
from app.schemas.monitoring import ModelPerformanceHistory

logger = logging.getLogger(__name__)


async def store_performance_history(perf_history: ModelPerformanceHistory) -> str:
    """Store performance history in database"""
    try:
        perf_db = ModelPerformanceHistoryDB(
            id=str(uuid.uuid4()),
            model_id=perf_history.model_id,
            deployment_id=perf_history.deployment_id,
            time_window_start=perf_history.time_window_start,
            time_window_end=perf_history.time_window_end,
            model_type=perf_history.model_type,
            accuracy=perf_history.accuracy,
            precision=perf_history.precision,
            recall=perf_history.recall,
            f1_score=perf_history.f1_score,
            auc_roc=perf_history.auc_roc,
            confusion_matrix=perf_history.confusion_matrix,
            mae=perf_history.mae,
            mse=perf_history.mse,
            rmse=perf_history.rmse,
            r2_score=perf_history.r2_score,
            total_samples=perf_history.total_samples,
            samples_with_ground_truth=perf_history.samples_with_ground_truth,
            baseline_accuracy=perf_history.baseline_accuracy,
            baseline_f1=perf_history.baseline_f1,
            baseline_mae=perf_history.baseline_mae,
            baseline_r2=perf_history.baseline_r2,
            accuracy_delta=perf_history.accuracy_delta,
            f1_delta=perf_history.f1_delta,
            mae_delta=perf_history.mae_delta,
            r2_delta=perf_history.r2_delta,
            performance_degraded=perf_history.performance_degraded,
            degradation_severity=perf_history.degradation_severity,
            degradation_threshold=perf_history.degradation_threshold,
            p_value=perf_history.p_value if hasattr(perf_history, 'p_value') else None,
            per_class_metrics=perf_history.per_class_metrics if hasattr(perf_history, 'per_class_metrics') else None,
            residual_stats=perf_history.residual_stats if hasattr(perf_history, 'residual_stats') else None
        )
        
        async with get_session() as session:
            session.add(perf_db)
            await session.commit()
            logger.info(f"Stored performance history for model {perf_history.model_id}")
            return perf_db.id
            
    except Exception as e:
        logger.error(f"Error storing performance history: {e}")
        raise

