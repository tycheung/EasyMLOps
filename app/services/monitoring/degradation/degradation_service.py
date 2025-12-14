"""
Performance degradation service facade
Combines all degradation detection functionality
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.monitoring import ModelPerformanceHistory
from app.services.monitoring.base import BaseMonitoringService

from app.services.monitoring.degradation.prediction_logging import log_prediction_with_ground_truth
from app.services.monitoring.degradation.classification_metrics import calculate_classification_metrics
from app.services.monitoring.degradation.regression_metrics import calculate_regression_metrics
from app.services.monitoring.degradation.degradation_detection import detect_performance_degradation
from app.services.monitoring.degradation.baseline_helpers import get_baseline_performance
from app.services.monitoring.degradation.history_storage import store_performance_history
from app.services.monitoring.degradation.alert_creation import create_performance_degradation_alert

logger = logging.getLogger(__name__)


class PerformanceDegradationService(BaseMonitoringService):
    """Service for performance degradation"""
    
    async def log_prediction_with_ground_truth(
        self,
        model_id: str,
        deployment_id: Optional[str],
        input_data: Dict[str, Any],
        output_data: Any,
        ground_truth: Any,
        latency_ms: float,
        api_endpoint: str,
        success: bool = True,
        error_message: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Log prediction with ground truth for performance evaluation"""
        return await log_prediction_with_ground_truth(
            model_id=model_id,
            deployment_id=deployment_id,
            input_data=input_data,
            output_data=output_data,
            ground_truth=ground_truth,
            latency_ms=latency_ms,
            api_endpoint=api_endpoint,
            success=success,
            error_message=error_message,
            user_agent=user_agent,
            ip_address=ip_address
        )
    
    async def calculate_classification_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None
    ) -> ModelPerformanceHistory:
        """Calculate classification metrics (accuracy, precision, recall, F1, AUC-ROC)"""
        return await calculate_classification_metrics(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id,
            extract_prediction_value=self._extract_prediction_value,
            get_baseline_performance=self._get_baseline_performance,
            store_performance_history=self.store_performance_history,
            create_performance_degradation_alert=self._create_performance_degradation_alert
        )
    
    async def calculate_regression_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        deployment_id: Optional[str] = None
    ) -> ModelPerformanceHistory:
        """Calculate regression metrics (MAE, MSE, RMSE, R²)"""
        return await calculate_regression_metrics(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            deployment_id=deployment_id,
            extract_prediction_value=self._extract_prediction_value,
            get_baseline_performance=self._get_baseline_performance,
            store_performance_history=self.store_performance_history,
            create_performance_degradation_alert=self._create_performance_degradation_alert
        )
    
    async def detect_performance_degradation(
        self,
        model_id: str,
        baseline_window_start: datetime,
        baseline_window_end: datetime,
        current_window_start: datetime,
        current_window_end: datetime,
        deployment_id: Optional[str] = None,
        degradation_threshold: float = 0.05
    ) -> ModelPerformanceHistory:
        """Detect performance degradation by comparing baseline and current windows"""
        return await detect_performance_degradation(
            model_id=model_id,
            baseline_window_start=baseline_window_start,
            baseline_window_end=baseline_window_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
            deployment_id=deployment_id,
            degradation_threshold=degradation_threshold,
            calculate_classification_metrics=self.calculate_classification_metrics,
            calculate_regression_metrics=self.calculate_regression_metrics,
            store_performance_history=self.store_performance_history
        )
    
    async def _get_baseline_performance(
        self,
        model_id: str,
        deployment_id: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """Get baseline performance metrics for comparison"""
        return await get_baseline_performance(model_id=model_id, deployment_id=deployment_id)
    
    async def store_performance_history(self, perf_history: ModelPerformanceHistory) -> str:
        """Store performance history in database"""
        return await store_performance_history(perf_history)
    
    async def _create_performance_degradation_alert(self, perf_history: ModelPerformanceHistory) -> Optional[str]:
        """Create an alert for performance degradation"""
        return await create_performance_degradation_alert(perf_history)

