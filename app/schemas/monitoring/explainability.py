"""
Model explainability schemas
Contains schemas for model explanations and feature importance
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ExplanationType(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION = "permutation"
    BUILTIN = "builtin"


class ImportanceType(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    AGGREGATED = "aggregated"


class ModelExplanation(BaseModel):
    """Schema for model explanations"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique explanation ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    prediction_log_id: Optional[str] = Field(None, description="Prediction log ID")
    
    # Explanation metadata
    explanation_type: ExplanationType = Field(..., description="Type of explanation")
    explanation_method: Optional[str] = Field(None, description="Specific explanation method used")
    
    # Input and output
    input_data: Dict[str, Any] = Field(..., description="Original input data")
    prediction: Optional[Any] = Field(None, description="Model prediction")
    
    # Explanation data
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    shap_values: Optional[Union[List[float], Dict[str, float]]] = Field(None, description="SHAP values")
    shap_base_value: Optional[float] = Field(None, description="SHAP base value")
    lime_explanation: Optional[Dict[str, Any]] = Field(None, description="LIME explanation data")
    explanation_text: Optional[str] = Field(None, description="Human-readable explanation")
    
    # Explanation metadata
    explanation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    computation_time_ms: Optional[float] = Field(None, description="Computation time in milliseconds", ge=0)
    
    # Caching
    is_cached: bool = Field(False, description="Whether explanation is cached")
    cache_key: Optional[str] = Field(None, description="Cache key")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class FeatureImportance(BaseModel):
    """Schema for feature importance tracking"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique importance ID")
    model_id: str = Field(..., description="Model ID")
    deployment_id: Optional[str] = Field(None, description="Deployment ID")
    model_version: Optional[str] = Field(None, description="Model version")
    
    # Importance type
    importance_type: ImportanceType = Field(..., description="Type of importance")
    calculation_method: str = Field(..., description="Method used to calculate importance")
    
    # Feature importance data
    feature_importance_scores: Dict[str, float] = Field(..., description="Feature importance scores")
    feature_names: Optional[List[str]] = Field(None, description="Feature names in order")
    total_features: int = Field(..., description="Total number of features", ge=1)
    
    # Aggregation metadata
    sample_size: Optional[int] = Field(None, description="Number of samples used", ge=1)
    time_window_start: Optional[datetime] = Field(None, description="Time window start")
    time_window_end: Optional[datetime] = Field(None, description="Time window end")
    
    # Comparison metadata
    baseline_importance_id: Optional[str] = Field(None, description="Baseline importance ID for comparison")
    importance_delta: Optional[Dict[str, float]] = Field(None, description="Change in importance vs baseline")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

