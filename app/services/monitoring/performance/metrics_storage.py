"""
Metrics storage service
Handles storing performance and confidence metrics in the database
"""

import logging
import uuid

from app.database import get_session
from app.models.monitoring import ModelPerformanceMetricsDB, ModelConfidenceMetricsDB
from app.schemas.monitoring import ModelPerformanceMetrics, ModelConfidenceMetrics
from app.services.monitoring.base import BaseMonitoringService

logger = logging.getLogger(__name__)


class MetricsStorageService(BaseMonitoringService):
    """Service for storing metrics in the database"""
    
    async def store_performance_metrics(self, metrics: ModelPerformanceMetrics) -> str:
        """Store aggregated performance metrics in database"""
        try:
            metrics_db = ModelPerformanceMetricsDB(
                id=str(uuid.uuid4()),
                model_id=metrics.model_id,
                deployment_id=metrics.deployment_id,
                time_window_start=metrics.time_window_start,
                time_window_end=metrics.time_window_end,
                total_requests=metrics.total_requests,
                successful_requests=metrics.successful_requests,
                failed_requests=metrics.failed_requests,
                requests_per_minute=metrics.requests_per_minute,
                avg_latency_ms=metrics.avg_latency_ms,
                p50_latency_ms=metrics.p50_latency_ms,
                p95_latency_ms=metrics.p95_latency_ms,
                p99_latency_ms=metrics.p99_latency_ms,
                p99_9_latency_ms=metrics.p99_9_latency_ms,
                min_latency_ms=metrics.min_latency_ms,
                std_dev_latency_ms=metrics.std_dev_latency_ms,
                latency_distribution=metrics.latency_distribution,
                success_rate=metrics.success_rate,
                error_rate=metrics.error_rate,
                requests_per_second=metrics.requests_per_second,
                requests_per_hour=metrics.requests_per_hour,
                avg_concurrent_requests=metrics.avg_concurrent_requests,
                avg_queue_depth=metrics.avg_queue_depth,
                avg_ttfb_ms=metrics.avg_ttfb_ms,
                avg_inference_time_ms=metrics.avg_inference_time_ms,
                avg_total_time_ms=metrics.avg_total_time_ms,
                batch_metrics=metrics.batch_metrics
            )
            
            async with get_session() as session:
                session.add(metrics_db)
                await session.commit()
                logger.info(f"Stored performance metrics for model {metrics.model_id}")
                return metrics_db.id
                
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
            raise
    
    async def store_confidence_metrics(self, metrics: ModelConfidenceMetrics) -> str:
        """Store confidence metrics in database"""
        try:
            metrics_db = ModelConfidenceMetricsDB(
                id=str(uuid.uuid4()),
                model_id=metrics.model_id,
                deployment_id=metrics.deployment_id,
                time_window_start=metrics.time_window_start,
                time_window_end=metrics.time_window_end,
                avg_confidence=metrics.avg_confidence,
                min_confidence=metrics.min_confidence,
                max_confidence=metrics.max_confidence,
                median_confidence=metrics.median_confidence,
                std_dev_confidence=metrics.std_dev_confidence,
                confidence_distribution=metrics.confidence_distribution,
                low_confidence_count=metrics.low_confidence_count,
                low_confidence_threshold=metrics.low_confidence_threshold,
                low_confidence_percentage=metrics.low_confidence_percentage,
                calibration_error=metrics.calibration_error,
                brier_score=metrics.brier_score,
                confidence_accuracy_correlation=metrics.confidence_accuracy_correlation,
                avg_uncertainty=metrics.avg_uncertainty,
                high_uncertainty_count=metrics.high_uncertainty_count,
                high_uncertainty_threshold=metrics.high_uncertainty_threshold,
                uncertainty_distribution=metrics.uncertainty_distribution,
                avg_interval_width=metrics.avg_interval_width,
                coverage_rate=metrics.coverage_rate,
                total_samples=metrics.total_samples,
                samples_with_confidence=metrics.samples_with_confidence,
                samples_with_ground_truth=metrics.samples_with_ground_truth,
                per_class_confidence=metrics.per_class_confidence,
                confidence_trend=metrics.confidence_trend if hasattr(metrics, 'confidence_trend') else None
            )
            
            async with get_session() as session:
                session.add(metrics_db)
                await session.commit()
                logger.info(f"Stored confidence metrics for model {metrics.model_id}")
                return metrics_db.id
                
        except Exception as e:
            logger.error(f"Error storing confidence metrics: {e}")
            raise

