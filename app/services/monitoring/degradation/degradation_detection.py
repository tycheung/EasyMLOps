"""
Performance degradation detection
Handles detecting performance degradation by comparing windows
"""

import logging
from datetime import datetime
from typing import Optional

from app.schemas.monitoring import ModelPerformanceHistory

logger = logging.getLogger(__name__)


async def detect_performance_degradation(
    model_id: str,
    baseline_window_start: datetime,
    baseline_window_end: datetime,
    current_window_start: datetime,
    current_window_end: datetime,
    deployment_id: Optional[str] = None,
    degradation_threshold: float = 0.05,
    calculate_classification_metrics=None,
    calculate_regression_metrics=None,
    store_performance_history=None
) -> ModelPerformanceHistory:
    """Detect performance degradation by comparing baseline and current windows"""
    try:
        # Calculate metrics for baseline window
        baseline_metrics = await calculate_classification_metrics(
            model_id=model_id,
            start_time=baseline_window_start,
            end_time=baseline_window_end,
            deployment_id=deployment_id
        ) if calculate_classification_metrics else None
        
        # Calculate metrics for current window
        current_metrics = await calculate_classification_metrics(
            model_id=model_id,
            start_time=current_window_start,
            end_time=current_window_end,
            deployment_id=deployment_id
        ) if calculate_classification_metrics else None
        
        # If baseline has no accuracy, try regression metrics
        if baseline_metrics and baseline_metrics.accuracy is None and baseline_metrics.mae is not None:
            baseline_regression = await calculate_regression_metrics(
                model_id=model_id,
                start_time=baseline_window_start,
                end_time=baseline_window_end,
                deployment_id=deployment_id
            ) if calculate_regression_metrics else None
            
            current_regression = await calculate_regression_metrics(
                model_id=model_id,
                start_time=current_window_start,
                end_time=current_window_end,
                deployment_id=deployment_id
            ) if calculate_regression_metrics else None
            
            if baseline_regression and current_regression:
                # Use regression metrics
                current_metrics.mae = current_regression.mae
                current_metrics.mae_delta = (current_regression.mae - baseline_regression.mae) if baseline_regression.mae and current_regression.mae else None
                current_metrics.performance_degraded = (current_metrics.mae_delta is not None and 
                                                          current_metrics.mae_delta > degradation_threshold)
        else:
            # Use classification metrics
            if baseline_metrics and current_metrics and baseline_metrics.accuracy and current_metrics.accuracy:
                accuracy_degradation = baseline_metrics.accuracy - current_metrics.accuracy
                current_metrics.accuracy_delta = -accuracy_degradation  # Negative means degradation
                current_metrics.performance_degraded = accuracy_degradation > degradation_threshold
                
                if current_metrics.performance_degraded:
                    if accuracy_degradation > 0.20:
                        current_metrics.degradation_severity = "critical"
                    elif accuracy_degradation > 0.10:
                        current_metrics.degradation_severity = "high"
                    elif accuracy_degradation > 0.05:
                        current_metrics.degradation_severity = "medium"
                    else:
                        current_metrics.degradation_severity = "low"
        
        if current_metrics:
            current_metrics.degradation_threshold = degradation_threshold
            if baseline_metrics:
                current_metrics.baseline_accuracy = baseline_metrics.accuracy
                current_metrics.baseline_f1 = baseline_metrics.f1_score
                current_metrics.baseline_mae = baseline_metrics.mae
                current_metrics.baseline_r2 = baseline_metrics.r2_score
            
            # Store the degradation detection result
            if store_performance_history:
                await store_performance_history(current_metrics)
            
            return current_metrics
        
        # Fallback if no metrics calculated
        return ModelPerformanceHistory(
            model_id=model_id,
            deployment_id=deployment_id,
            time_window_start=current_window_start,
            time_window_end=current_window_end,
            model_type="unknown",
            total_samples=0,
            samples_with_ground_truth=0,
            performance_degraded=False
        )
        
    except Exception as e:
        logger.error(f"Error detecting performance degradation: {e}", exc_info=True)
        raise

