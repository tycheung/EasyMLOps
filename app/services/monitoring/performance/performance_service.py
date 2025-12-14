"""
Performance monitoring service - Facade pattern
Delegates to domain-specific performance monitoring services
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from app.schemas.monitoring import ModelPerformanceMetrics, ModelConfidenceMetrics
from app.services.monitoring.performance.prediction_logging import PredictionLoggingService
from app.services.monitoring.performance.metrics_calculation import MetricsCalculationService
from app.services.monitoring.performance.metrics_storage import MetricsStorageService
from app.services.monitoring.performance.aggregation import MetricsAggregationService

logger = logging.getLogger(__name__)


class PerformanceMonitoringService:
    """Service for performance monitoring and metrics - Facade pattern"""
    
    def __init__(self):
        """Initialize all sub-services"""
        self.storage = MetricsStorageService()
        self.calculation = MetricsCalculationService(self.storage)
        self.logging = PredictionLoggingService()
        self.aggregation = MetricsAggregationService(self.calculation)
    
    # Prediction logging - delegate to PredictionLoggingService
    async def log_prediction(
        self,
        model_id: str,
        deployment_id: Optional[str],
        input_data: Dict[str, Any],
        output_data: Any,
        latency_ms: float,
        api_endpoint: str,
        success: bool = True,
        error_message: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        inference_time_ms: Optional[float] = None,
        ttfb_ms: Optional[float] = None,
        batch_size: Optional[int] = None,
        is_batch: bool = False
    ) -> str:
        """Log individual prediction for performance monitoring"""
        return await self.logging.log_prediction(
            model_id, deployment_id, input_data, output_data, latency_ms,
            api_endpoint, success, error_message, user_agent, ip_address,
            inference_time_ms, ttfb_ms, batch_size, is_batch
        )
    
    async def get_prediction_logs(self, model_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction logs"""
        return await self.logging.get_prediction_logs(model_id, limit)
    
    # Metrics calculation - delegate to MetricsCalculationService
    async def get_model_performance_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None
    ) -> ModelPerformanceMetrics:
        """Calculate aggregated performance metrics for a model"""
        return await self.calculation.get_model_performance_metrics(
            model_id, start_time, end_time, deployment_id
        )
    
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
        return await self.calculation.calculate_confidence_metrics(
            model_id, start_time, end_time, deployment_id,
            low_confidence_threshold, high_uncertainty_threshold
        )
    
    # Metrics storage - delegate to MetricsStorageService
    async def store_performance_metrics(self, metrics: ModelPerformanceMetrics) -> str:
        """Store aggregated performance metrics in database"""
        return await self.storage.store_performance_metrics(metrics)
    
    async def store_confidence_metrics(self, metrics: ModelConfidenceMetrics) -> str:
        """Store confidence metrics in database"""
        return await self.storage.store_confidence_metrics(metrics)
    
    # Aggregation - delegate to MetricsAggregationService
    async def get_aggregated_metrics(self, model_id: Optional[str] = None, time_range: str = "24h") -> Dict[str, Any]:
        """Get aggregated metrics"""
        return await self.aggregation.get_aggregated_metrics(model_id, time_range)
    
    async def get_deployment_summary(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment summary"""
        return await self.aggregation.get_deployment_summary(deployment_id)
    
    # Helper methods - delegate to base service methods
    def _extract_confidence_score(self, *args, **kwargs):
        """Extract confidence score"""
        return self.logging._extract_confidence_score(*args, **kwargs)
    
    def _extract_confidence_scores(self, *args, **kwargs):
        """Extract confidence scores"""
        return self.logging._extract_confidence_scores(*args, **kwargs)
    
    def _extract_uncertainty(self, *args, **kwargs):
        """Extract uncertainty"""
        return self.logging._extract_uncertainty(*args, **kwargs)
    
    def _extract_prediction_interval(self, *args, **kwargs):
        """Extract prediction interval"""
        return self.logging._extract_prediction_interval(*args, **kwargs)
    
    def _extract_prediction_value(self, *args, **kwargs):
        """Extract prediction value"""
        return self.logging._extract_prediction_value(*args, **kwargs)

