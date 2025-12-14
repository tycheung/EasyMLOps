"""
Classification metrics calculation
Handles calculation of classification performance metrics
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# Optional import for sklearn
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    def _sklearn_not_available(*args, **kwargs):
        raise ImportError("scikit-learn is not available. Please install it to use classification metrics.")
    accuracy_score = precision_score = recall_score = f1_score = _sklearn_not_available
    confusion_matrix = roc_auc_score = _sklearn_not_available
    logging.warning("scikit-learn not available - classification metrics will be limited")

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import ModelPerformanceHistory
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


async def calculate_classification_metrics(
    model_id: str,
    start_time: datetime,
    end_time: datetime,
    deployment_id: Optional[str] = None,
    extract_prediction_value=None,
    get_baseline_performance=None,
    store_performance_history=None,
    create_performance_degradation_alert=None
) -> ModelPerformanceHistory:
    """Calculate classification metrics (accuracy, precision, recall, F1, AUC-ROC)"""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, skipping classification metrics calculation.")
        return ModelPerformanceHistory(
            model_id=model_id,
            deployment_id=deployment_id,
            time_window_start=start_time,
            time_window_end=end_time,
            model_type="classification",
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
                    model_type="classification",
                    total_samples=len(logs),
                    samples_with_ground_truth=len(logs),
                    performance_degraded=False
                )
            
            # Extract predictions and ground truth
            predictions = []
            ground_truths = []
            prediction_probs = []
            
            for log in logs:
                pred_value = extract_prediction_value(log.output_data) if extract_prediction_value else None
                gt_value = extract_prediction_value(log.ground_truth) if extract_prediction_value else None
                
                if pred_value is not None and gt_value is not None:
                    if isinstance(pred_value, float):
                        pred_value = int(round(pred_value))
                    if isinstance(gt_value, float):
                        gt_value = int(round(gt_value))
                    
                    predictions.append(pred_value)
                    ground_truths.append(gt_value)
                    
                    # Try to extract probability/confidence for AUC-ROC
                    if isinstance(log.output_data, dict):
                        prob = log.output_data.get("probability") or log.output_data.get("confidence")
                        if prob is not None:
                            prediction_probs.append(float(prob))
            
            if len(predictions) < 10:
                return ModelPerformanceHistory(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    model_type="classification",
                    total_samples=len(logs),
                    samples_with_ground_truth=len(predictions),
                    performance_degraded=False
                )
            
            # Calculate metrics
            y_true = np.array(ground_truths)
            y_pred = np.array(predictions)
            
            accuracy = float(accuracy_score(y_true, y_pred))
            
            try:
                precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            except:
                precision = recall = f1 = 0.0
            
            try:
                cm = confusion_matrix(y_true, y_pred)
                cm_list = cm.tolist()
            except:
                cm_list = None
            
            # Calculate AUC-ROC (binary classification only)
            auc_roc = None
            if len(np.unique(y_true)) == 2 and len(prediction_probs) == len(predictions):
                try:
                    auc_roc = float(roc_auc_score(y_true, prediction_probs))
                except:
                    pass
            
            # Calculate per-class metrics
            per_class_metrics = {}
            unique_classes = np.unique(y_true)
            for cls in unique_classes:
                cls_mask = (y_true == cls)
                cls_pred = y_pred[cls_mask]
                cls_true = y_true[cls_mask]
                if len(cls_true) > 0:
                    per_class_metrics[str(int(cls))] = {
                        "precision": float(precision_score(cls_true, cls_pred, average='binary', zero_division=0)) if len(np.unique(cls_true)) > 1 else 0.0,
                        "recall": float(recall_score(cls_true, cls_pred, average='binary', zero_division=0)) if len(np.unique(cls_true)) > 1 else 0.0,
                        "f1": float(f1_score(cls_true, cls_pred, average='binary', zero_division=0)) if len(np.unique(cls_true)) > 1 else 0.0,
                        "support": int(len(cls_true))
                    }
            
            # Get baseline for comparison
            baseline_metrics = await get_baseline_performance(model_id, deployment_id) if get_baseline_performance else None
            
            # Calculate deltas
            accuracy_delta = None
            f1_delta = None
            if baseline_metrics:
                if baseline_metrics.get("accuracy") is not None:
                    accuracy_delta = accuracy - baseline_metrics["accuracy"]
                if baseline_metrics.get("f1_score") is not None:
                    f1_delta = f1 - baseline_metrics["f1_score"]
            
            # Detect degradation
            performance_degraded = False
            degradation_severity = None
            degradation_threshold = 0.05
            
            if baseline_metrics:
                if baseline_metrics.get("accuracy") is not None:
                    accuracy_degradation = baseline_metrics["accuracy"] - accuracy
                    if accuracy_degradation > degradation_threshold:
                        performance_degraded = True
                        if accuracy_degradation > 0.20:
                            degradation_severity = "critical"
                        elif accuracy_degradation > 0.10:
                            degradation_severity = "high"
                        elif accuracy_degradation > 0.05:
                            degradation_severity = "medium"
                        else:
                            degradation_severity = "low"
            
            perf_history = ModelPerformanceHistory(
                model_id=model_id,
                deployment_id=deployment_id,
                time_window_start=start_time,
                time_window_end=end_time,
                model_type="classification",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                confusion_matrix=cm_list,
                total_samples=len(logs),
                samples_with_ground_truth=len(predictions),
                baseline_accuracy=baseline_metrics.get("accuracy") if baseline_metrics else None,
                baseline_f1=baseline_metrics.get("f1_score") if baseline_metrics else None,
                accuracy_delta=accuracy_delta,
                f1_delta=f1_delta,
                performance_degraded=performance_degraded,
                degradation_severity=degradation_severity,
                degradation_threshold=degradation_threshold,
                per_class_metrics=per_class_metrics if per_class_metrics else None
            )
            
            # Store performance history
            if store_performance_history:
                await store_performance_history(perf_history)
            
            # Create alert if degraded
            if performance_degraded and create_performance_degradation_alert:
                await create_performance_degradation_alert(perf_history)
            
            return perf_history
            
    except Exception as e:
        logger.error(f"Error calculating classification metrics: {e}", exc_info=True)
        raise

