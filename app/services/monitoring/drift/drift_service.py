"""
Drift detection service - Facade pattern
Delegates to domain-specific drift detection services
"""

from datetime import datetime
from typing import Optional
import logging

from app.schemas.monitoring import ModelDriftDetection
from app.services.monitoring.drift.feature_drift import FeatureDriftService
from app.services.monitoring.drift.data_drift import DataDriftService
from app.services.monitoring.drift.prediction_drift import PredictionDriftService
from app.services.monitoring.drift.drift_storage import DriftStorageService

logger = logging.getLogger(__name__)


class DriftDetectionService:
    """Service for drift detection - Facade pattern"""
    
    def __init__(self):
        """Initialize all sub-services"""
        self.storage = DriftStorageService()
        self.feature = FeatureDriftService(self.storage)
        self.data = DataDriftService(self.storage)
        self.prediction = PredictionDriftService(self.storage)
    
    # Feature drift - delegate to FeatureDriftService
    async def detect_feature_drift(
        self,
        model_id: str,
        baseline_window_start: datetime,
        baseline_window_end: datetime,
        current_window_start: datetime,
        current_window_end: datetime,
        deployment_id: Optional[str] = None,
        drift_threshold: float = 0.2,
        ks_p_value_threshold: float = 0.05
    ) -> ModelDriftDetection:
        """Detect feature drift using KS test and PSI"""
        return await self.feature.detect_feature_drift(
            model_id, baseline_window_start, baseline_window_end,
            current_window_start, current_window_end, deployment_id,
            drift_threshold, ks_p_value_threshold
        )
    
    # Data drift - delegate to DataDriftService
    async def detect_data_drift(
        self,
        model_id: str,
        baseline_window_start: datetime,
        baseline_window_end: datetime,
        current_window_start: datetime,
        current_window_end: datetime,
        deployment_id: Optional[str] = None
    ) -> ModelDriftDetection:
        """Detect data drift including schema changes and data quality issues"""
        return await self.data.detect_data_drift(
            model_id, baseline_window_start, baseline_window_end,
            current_window_start, current_window_end, deployment_id
        )
    
    # Prediction drift - delegate to PredictionDriftService
    async def detect_prediction_drift(
        self,
        model_id: str,
        baseline_window_start: datetime,
        baseline_window_end: datetime,
        current_window_start: datetime,
        current_window_end: datetime,
        deployment_id: Optional[str] = None,
        drift_threshold: float = 0.2
    ) -> ModelDriftDetection:
        """Detect prediction drift by comparing prediction distributions"""
        return await self.prediction.detect_prediction_drift(
            model_id, baseline_window_start, baseline_window_end,
            current_window_start, current_window_end, deployment_id, drift_threshold
        )
    
    # Storage - delegate to DriftStorageService
    async def store_drift_detection(self, drift_result: ModelDriftDetection) -> str:
        """Store drift detection result in database"""
        return await self.storage.store_drift_detection(drift_result)
    
    # Helper methods - delegate to base service methods
    def _calculate_psi(self, *args, **kwargs):
        """Calculate PSI"""
        return self.feature._calculate_psi(*args, **kwargs)
    
    def _ks_test(self, *args, **kwargs):
        """KS test"""
        return self.feature._ks_test(*args, **kwargs)
    
    def _extract_prediction_value(self, *args, **kwargs):
        """Extract prediction value"""
        return self.prediction._extract_prediction_value(*args, **kwargs)

