"""
Regression metrics calculation
Handles calculation of regression performance metrics
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# Optional import for sklearn
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    def _sklearn_not_available(*args, **kwargs):
        raise ImportError("scikit-learn is not available. Please install it to use regression metrics.")
    mean_absolute_error = mean_squared_error = r2_score = _sklearn_not_available
    logging.warning("scikit-learn not available - regression metrics will be limited")

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import ModelPerformanceHistory

logger = logging.getLogger(__name__)


async def calculate_regression_metrics(
    model_id: str,
    start_time: datetime,
    end_time: datetime,
    deployment_id: Optional[str] = None,
    extract_prediction_value=None,
    get_baseline_performance=None,
    store_performance_history=None,
    create_performance_degradation_alert=None
) -> ModelPerformanceHistory:
    """Calculate regression metrics (MAE, MSE, RMSE, R²)"""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, skipping regression metrics calculation.")
        return ModelPerformanceHistory(
            model_id=model_id,
            deployment_id=deployment_id,
            time_window_start=start_time,
            time_window_end=end_time,
            model_type="regression",
            total_samples=0,
            samples_with_ground_truth=0,
            performance_degraded=False
        )
    
    try:
        async with get_session() as session:
            # Get prediction logs with ground truth
            stmt = select(PredictionLogDB).where(
                PredictionLogDB.model_id == model_id,
                PredictionLogDB.timestamp >= start_time,
                PredictionLogDB.timestamp <= end_time,
                PredictionLogDB.success == True,
                PredictionLogDB.ground_truth.isnot(None)
            )
            if deployment_id:
                stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
            
            result = await session.execute(stmt)
            logs = result.scalars().all()
            
            if len(logs) < 10:
                return ModelPerformanceHistory(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    model_type="regression",
                    total_samples=len(logs),
                    samples_with_ground_truth=len(logs),
                    performance_degraded=False
                )
            
            # Extract predictions and ground truth
            predictions = []
            ground_truths = []
            
            for log in logs:
                pred_value = extract_prediction_value(log.output_data) if extract_prediction_value else None
                gt_value = extract_prediction_value(log.ground_truth) if extract_prediction_value else None
                
                if pred_value is not None and gt_value is not None:
                    predictions.append(float(pred_value))
                    ground_truths.append(float(gt_value))
            
            if len(predictions) < 10:
                return ModelPerformanceHistory(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    model_type="regression",
                    total_samples=len(logs),
                    samples_with_ground_truth=len(predictions),
                    performance_degraded=False
                )
            
            y_true = np.array(ground_truths)
            y_pred = np.array(predictions)
            
            # Calculate regression metrics
            mae = float(mean_absolute_error(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            
            try:
                r2 = float(r2_score(y_true, y_pred))
            except:
                r2 = None
            
            # Calculate residual statistics
            residuals = y_true - y_pred
            residual_stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "median": float(np.median(residuals))
            }
            
            # Get baseline for comparison
            baseline_metrics = await get_baseline_performance(model_id, deployment_id) if get_baseline_performance else None
            
            # Calculate deltas
            mae_delta = None
            r2_delta = None
            if baseline_metrics:
                if baseline_metrics.get("mae") is not None:
                    mae_delta = mae - baseline_metrics["mae"]
                if baseline_metrics.get("r2_score") is not None and r2 is not None:
                    r2_delta = r2 - baseline_metrics["r2_score"]
            
            # Detect degradation
            performance_degraded = False
            degradation_severity = None
            degradation_threshold = 0.10
            
            if baseline_metrics:
                if baseline_metrics.get("mae") is not None:
                    mae_increase_pct = ((mae - baseline_metrics["mae"]) / baseline_metrics["mae"]) * 100
                    if mae_increase_pct > 20:
                        performance_degraded = True
                        if mae_increase_pct > 50:
                            degradation_severity = "critical"
                        elif mae_increase_pct > 30:
                            degradation_severity = "high"
                        elif mae_increase_pct > 20:
                            degradation_severity = "medium"
                        else:
                            degradation_severity = "low"
                
                if baseline_metrics.get("r2_score") is not None and r2 is not None:
                    r2_decrease = baseline_metrics["r2_score"] - r2
                    if r2_decrease > 0.10:
                        performance_degraded = True
                        if not degradation_severity:
                            if r2_decrease > 0.30:
                                degradation_severity = "critical"
                            elif r2_decrease > 0.20:
                                degradation_severity = "high"
                            elif r2_decrease > 0.10:
                                degradation_severity = "medium"
                            else:
                                degradation_severity = "low"
            
            perf_history = ModelPerformanceHistory(
                model_id=model_id,
                deployment_id=deployment_id,
                time_window_start=start_time,
                time_window_end=end_time,
                model_type="regression",
                mae=mae,
                mse=mse,
                rmse=rmse,
                r2_score=r2,
                total_samples=len(logs),
                samples_with_ground_truth=len(predictions),
                baseline_mae=baseline_metrics.get("mae") if baseline_metrics else None,
                baseline_r2=baseline_metrics.get("r2_score") if baseline_metrics else None,
                mae_delta=mae_delta,
                r2_delta=r2_delta,
                performance_degraded=performance_degraded,
                degradation_severity=degradation_severity,
                degradation_threshold=degradation_threshold,
                residual_stats=residual_stats
            )
            
            # Store performance history
            if store_performance_history:
                await store_performance_history(perf_history)
            
            # Create alert if degraded
            if performance_degraded and create_performance_degradation_alert:
                await create_performance_degradation_alert(perf_history)
            
            return perf_history
            
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}", exc_info=True)
        raise

