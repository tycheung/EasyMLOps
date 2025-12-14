"""
Metrics calculation service
Handles calculation of performance and confidence metrics
"""

import logging
import statistics
import numpy as np
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.database import get_session
from app.models.monitoring import PredictionLogDB
from app.schemas.monitoring import ModelPerformanceMetrics, ModelConfidenceMetrics
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class MetricsCalculationService(BaseMonitoringService):
    """Service for calculating performance and confidence metrics"""
    
    def __init__(self, metrics_storage_service=None):
        """Initialize with optional storage service"""
        self.metrics_storage = metrics_storage_service
    
    async def get_model_performance_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None
    ) -> ModelPerformanceMetrics:
        """Calculate aggregated performance metrics for a model"""
        try:
            async with get_session() as session:
                stmt = select(PredictionLogDB).where(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time
                )
                if deployment_id:
                    stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                if not logs:
                    return ModelPerformanceMetrics(
                        model_id=model_id,
                        deployment_id=deployment_id,
                        time_window_start=start_time,
                        time_window_end=end_time,
                        total_requests=0,
                        successful_requests=0,
                        failed_requests=0,
                        requests_per_minute=0.0,
                        requests_per_second=0.0,
                        requests_per_hour=0.0,
                        avg_concurrent_requests=None,
                        avg_queue_depth=None,
                        avg_latency_ms=0.0,
                        p50_latency_ms=0.0,
                        p95_latency_ms=0.0,
                        p99_latency_ms=0.0,
                        p99_9_latency_ms=None,
                        min_latency_ms=None,
                        max_latency_ms=0.0,
                        std_dev_latency_ms=None,
                        latency_distribution=None,
                        success_rate=0.0,
                        error_rate=0.0,
                        avg_ttfb_ms=None,
                        avg_inference_time_ms=None,
                        avg_total_time_ms=0.0,
                        batch_metrics=None
                    )
                
                # Calculate metrics
                total_requests = len(logs)
                successful_requests = sum(1 for log in logs if log.success)
                failed_requests = total_requests - successful_requests
                
                # Calculate time-based metrics
                time_diff_seconds = (end_time - start_time).total_seconds()
                time_diff_minutes = time_diff_seconds / 60
                time_diff_hours = time_diff_seconds / 3600
                
                requests_per_minute = total_requests / time_diff_minutes if time_diff_minutes > 0 else 0
                requests_per_second = total_requests / time_diff_seconds if time_diff_seconds > 0 else 0
                requests_per_hour = total_requests / time_diff_hours if time_diff_hours > 0 else 0
                
                # Calculate latency metrics
                latencies = [log.latency_ms for log in logs]
                latencies.sort()
                
                if not latencies:
                    avg_latency_ms = 0.0
                    p50_latency_ms = 0.0
                    p95_latency_ms = 0.0
                    p99_latency_ms = 0.0
                    p99_9_latency_ms = None
                    min_latency_ms = None
                    max_latency_ms = 0.0
                    std_dev_latency_ms = None
                    latency_distribution = None
                else:
                    avg_latency_ms = statistics.mean(latencies)
                    p50_latency_ms = statistics.median(latencies)
                    p95_latency_ms = latencies[int(0.95 * len(latencies))]
                    p99_latency_ms = latencies[int(0.99 * len(latencies))]
                    p99_9_idx = int(0.999 * len(latencies))
                    p99_9_latency_ms = latencies[p99_9_idx] if p99_9_idx < len(latencies) else latencies[-1]
                    min_latency_ms = min(latencies)
                    max_latency_ms = max(latencies)
                    
                    if len(latencies) > 1:
                        std_dev_latency_ms = statistics.stdev(latencies)
                    else:
                        std_dev_latency_ms = 0.0
                    
                    # Create latency distribution histogram
                    num_bins = 10
                    if len(latencies) > 0:
                        min_val = min_latency_ms
                        max_val = max_latency_ms
                        bin_width = (max_val - min_val) / num_bins if max_val > min_val else 1.0
                        bins = [min_val + i * bin_width for i in range(num_bins + 1)]
                        histogram = [0] * num_bins
                        for latency in latencies:
                            bin_idx = min(int((latency - min_val) / bin_width) if bin_width > 0 else 0, num_bins - 1)
                            histogram[bin_idx] += 1
                        
                        latency_distribution = {
                            "bins": bins,
                            "counts": histogram,
                            "bin_width": bin_width,
                            "total_samples": len(latencies)
                        }
                    else:
                        latency_distribution = None
                
                # Calculate success/error rates
                success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
                error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
                
                # Calculate advanced timing metrics
                avg_ttfb_ms = None
                avg_inference_time_ms = None
                avg_total_time_ms = avg_latency_ms
                
                ttfb_values = [log.ttfb_ms for log in logs if log.ttfb_ms is not None]
                if ttfb_values:
                    avg_ttfb_ms = statistics.mean(ttfb_values)
                
                inference_times = [log.inference_time_ms for log in logs if log.inference_time_ms is not None]
                if inference_times:
                    avg_inference_time_ms = statistics.mean(inference_times)
                
                # Estimate concurrent requests using Little's Law
                avg_concurrent_requests = None
                if requests_per_second > 0 and avg_latency_ms > 0:
                    avg_latency_seconds = avg_latency_ms / 1000.0
                    avg_concurrent_requests = requests_per_second * avg_latency_seconds
                
                avg_queue_depth = None
                if avg_concurrent_requests is not None and avg_concurrent_requests > 1:
                    avg_queue_depth = max(0, avg_concurrent_requests - 1)
                
                # Calculate batch processing metrics
                batch_logs = [log for log in logs if log.is_batch]
                batch_metrics = None
                if batch_logs:
                    batch_sizes = [log.batch_size for log in batch_logs if log.batch_size is not None]
                    batch_latencies = [log.latency_ms for log in batch_logs]
                    
                    if batch_sizes:
                        batch_metrics = {
                            "total_batch_requests": len(batch_logs),
                            "avg_batch_size": statistics.mean(batch_sizes),
                            "min_batch_size": min(batch_sizes),
                            "max_batch_size": max(batch_sizes),
                            "avg_batch_latency_ms": statistics.mean(batch_latencies) if batch_latencies else None,
                            "items_per_second": sum(batch_sizes) / time_diff_seconds if time_diff_seconds > 0 else 0
                        }
                
                return ModelPerformanceMetrics(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    total_requests=total_requests,
                    successful_requests=successful_requests,
                    failed_requests=failed_requests,
                    requests_per_minute=requests_per_minute,
                    requests_per_second=requests_per_second,
                    requests_per_hour=requests_per_hour,
                    avg_concurrent_requests=avg_concurrent_requests,
                    avg_queue_depth=avg_queue_depth,
                    avg_latency_ms=avg_latency_ms,
                    p50_latency_ms=p50_latency_ms,
                    p95_latency_ms=p95_latency_ms,
                    p99_latency_ms=p99_latency_ms,
                    p99_9_latency_ms=p99_9_latency_ms,
                    min_latency_ms=min_latency_ms,
                    max_latency_ms=max_latency_ms if max_latency_ms is not None else (max(latencies) if latencies else 0.0),
                    std_dev_latency_ms=std_dev_latency_ms,
                    latency_distribution=latency_distribution,
                    success_rate=success_rate,
                    error_rate=error_rate,
                    avg_ttfb_ms=avg_ttfb_ms,
                    avg_inference_time_ms=avg_inference_time_ms,
                    avg_total_time_ms=avg_total_time_ms,
                    batch_metrics=batch_metrics
                )
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    async def calculate_confidence_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None,
        low_confidence_threshold: float = 0.5,
        high_uncertainty_threshold: Optional[float] = None
    ) -> ModelConfidenceMetrics:
        """Calculate confidence and uncertainty metrics for a time window"""
        try:
            async with get_session() as session:
                stmt = select(PredictionLogDB).where(
                    PredictionLogDB.model_id == model_id,
                    PredictionLogDB.timestamp >= start_time,
                    PredictionLogDB.timestamp <= end_time,
                    PredictionLogDB.success == True,
                    PredictionLogDB.confidence_score.isnot(None)
                )
                if deployment_id:
                    stmt = stmt.where(PredictionLogDB.deployment_id == deployment_id)
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                if len(logs) < 10:
                    return ModelConfidenceMetrics(
                        model_id=model_id,
                        deployment_id=deployment_id,
                        time_window_start=start_time,
                        time_window_end=end_time,
                        total_samples=len(logs),
                        samples_with_confidence=len(logs),
                        low_confidence_count=0
                    )
                
                confidence_scores = [log.confidence_score for log in logs if log.confidence_score is not None]
                
                if len(confidence_scores) < 10:
                    return ModelConfidenceMetrics(
                        model_id=model_id,
                        deployment_id=deployment_id,
                        time_window_start=start_time,
                        time_window_end=end_time,
                        total_samples=len(logs),
                        samples_with_confidence=len(confidence_scores),
                        low_confidence_count=0
                    )
                
                conf_array = np.array(confidence_scores)
                
                # Calculate basic statistics
                avg_confidence = float(np.mean(conf_array))
                min_confidence = float(np.min(conf_array))
                max_confidence = float(np.max(conf_array))
                median_confidence = float(np.median(conf_array))
                std_dev_confidence = float(np.std(conf_array))
                
                # Create confidence distribution histogram
                hist, bins = np.histogram(conf_array, bins=10, range=(0.0, 1.0))
                confidence_distribution = {
                    "bins": bins.tolist(),
                    "counts": hist.tolist()
                }
                
                # Count low confidence predictions
                low_confidence_count = int(np.sum(conf_array < low_confidence_threshold))
                low_confidence_percentage = (low_confidence_count / len(conf_array)) * 100.0
                
                # Extract uncertainty scores
                uncertainty_scores = [log.uncertainty_score for log in logs if log.uncertainty_score is not None]
                avg_uncertainty = None
                high_uncertainty_count = 0
                uncertainty_distribution = None
                
                if len(uncertainty_scores) > 0:
                    unc_array = np.array(uncertainty_scores)
                    avg_uncertainty = float(np.mean(unc_array))
                    
                    if high_uncertainty_threshold is not None:
                        high_uncertainty_count = int(np.sum(unc_array > high_uncertainty_threshold))
                    
                    hist_unc, bins_unc = np.histogram(unc_array, bins=10)
                    uncertainty_distribution = {
                        "bins": bins_unc.tolist(),
                        "counts": hist_unc.tolist()
                    }
                
                # Calculate calibration metrics (if ground truth available)
                calibration_error = None
                brier_score = None
                confidence_accuracy_correlation = None
                samples_with_ground_truth = 0
                
                logs_with_gt = [log for log in logs if log.ground_truth is not None]
                if len(logs_with_gt) >= 10:
                    samples_with_ground_truth = len(logs_with_gt)
                    
                    predictions = []
                    ground_truths = []
                    confidences = []
                    
                    for log in logs_with_gt:
                        pred_value = self._extract_prediction_value(log.output_data)
                        gt_value = self._extract_prediction_value(log.ground_truth)
                        conf = log.confidence_score
                        
                        if pred_value is not None and gt_value is not None and conf is not None:
                            if isinstance(pred_value, float):
                                pred_value = int(round(pred_value))
                            if isinstance(gt_value, float):
                                gt_value = int(round(gt_value))
                            
                            predictions.append(pred_value)
                            ground_truths.append(gt_value)
                            confidences.append(conf)
                    
                    if len(predictions) >= 10:
                        y_true = np.array(ground_truths)
                        y_pred = np.array(predictions)
                        confs = np.array(confidences)
                        
                        # Calculate Expected Calibration Error (ECE)
                        try:
                            n_bins = 10
                            bin_boundaries = np.linspace(0, 1, n_bins + 1)
                            bin_lowers = bin_boundaries[:-1]
                            bin_uppers = bin_boundaries[1:]
                            
                            ece = 0.0
                            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                                in_bin = (confs > bin_lower) & (confs <= bin_upper)
                                prop_in_bin = in_bin.mean()
                                
                                if prop_in_bin > 0:
                                    accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                                    avg_confidence_in_bin = confs[in_bin].mean()
                                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                            
                            calibration_error = float(ece)
                        except:
                            pass
                        
                        # Calculate Brier score
                        try:
                            from sklearn.metrics import brier_score_loss
                            brier = brier_score_loss(y_true, confs)
                            brier_score = float(brier)
                        except:
                            pass
                        
                        # Calculate confidence-accuracy correlation
                        try:
                            correct = (y_true == y_pred).astype(float)
                            correlation = np.corrcoef(confs, correct)[0, 1]
                            if not np.isnan(correlation):
                                confidence_accuracy_correlation = float(correlation)
                        except:
                            pass
                
                # Calculate prediction interval metrics
                avg_interval_width = None
                coverage_rate = None
                
                logs_with_intervals = [log for log in logs if log.prediction_interval_lower is not None and log.prediction_interval_upper is not None]
                if len(logs_with_intervals) > 0:
                    interval_widths = [
                        log.prediction_interval_upper - log.prediction_interval_lower
                        for log in logs_with_intervals
                    ]
                    avg_interval_width = float(np.mean(interval_widths))
                    
                    logs_with_interval_gt = [log for log in logs_with_intervals if log.ground_truth is not None]
                    if len(logs_with_interval_gt) > 0:
                        covered = 0
                        for log in logs_with_interval_gt:
                            gt_value = self._extract_prediction_value(log.ground_truth)
                            if gt_value is not None:
                                if log.prediction_interval_lower <= gt_value <= log.prediction_interval_upper:
                                    covered += 1
                        coverage_rate = (covered / len(logs_with_interval_gt)) * 100.0
                
                # Calculate per-class confidence
                per_class_confidence = None
                if logs and logs[0].confidence_scores is not None:
                    class_confs = {}
                    for log in logs:
                        if log.confidence_scores:
                            for class_name, conf in log.confidence_scores.items():
                                if class_name not in class_confs:
                                    class_confs[class_name] = []
                                class_confs[class_name].append(conf)
                    
                    if class_confs:
                        per_class_confidence = {
                            class_name: {
                                "avg": float(np.mean(confs)),
                                "min": float(np.min(confs)),
                                "max": float(np.max(confs)),
                                "std": float(np.std(confs)),
                                "count": len(confs)
                            }
                            for class_name, confs in class_confs.items()
                        }
                
                confidence_metrics = ModelConfidenceMetrics(
                    model_id=model_id,
                    deployment_id=deployment_id,
                    time_window_start=start_time,
                    time_window_end=end_time,
                    avg_confidence=avg_confidence,
                    min_confidence=min_confidence,
                    max_confidence=max_confidence,
                    median_confidence=median_confidence,
                    std_dev_confidence=std_dev_confidence,
                    confidence_distribution=confidence_distribution,
                    low_confidence_count=low_confidence_count,
                    low_confidence_threshold=low_confidence_threshold,
                    low_confidence_percentage=low_confidence_percentage,
                    calibration_error=calibration_error,
                    brier_score=brier_score,
                    confidence_accuracy_correlation=confidence_accuracy_correlation,
                    avg_uncertainty=avg_uncertainty,
                    high_uncertainty_count=high_uncertainty_count,
                    high_uncertainty_threshold=high_uncertainty_threshold,
                    uncertainty_distribution=uncertainty_distribution,
                    avg_interval_width=avg_interval_width,
                    coverage_rate=coverage_rate,
                    total_samples=len(logs),
                    samples_with_confidence=len(confidence_scores),
                    samples_with_ground_truth=samples_with_ground_truth,
                    per_class_confidence=per_class_confidence
                )
                
                # Store metrics if storage service is available
                if self.metrics_storage:
                    await self.metrics_storage.store_confidence_metrics(confidence_metrics)
                
                return confidence_metrics
                
        except Exception as e:
            logger.error(f"Error calculating confidence metrics: {e}", exc_info=True)
            raise

