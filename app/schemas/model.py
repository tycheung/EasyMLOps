"""
Pydantic schemas for EasyMLOps platform
Defines data validation schemas for models, deployments, and API requests/responses
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class ModelFramework(str, Enum):
    """Supported ML frameworks"""
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    H2O = "h2o"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Model types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    OTHER = "other"


class DataType(str, Enum):
    """Supported input data types"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    DATETIME = "datetime"


class ModelStatus(str, Enum):
    """Model deployment status"""
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


class DeploymentStatus(str, Enum):
    """Deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    RUNNING = "running"  # Alias for ACTIVE for backward compatibility
    STOPPED = "stopped"
    STOPPING = "stopping"
    FAILED = "failed"


# Input/Output Schema Definition
class FieldSchema(BaseModel):
    """Schema for individual input/output fields"""
    name: str = Field(..., description="Field name")
    data_type: DataType = Field(..., description="Data type of the field")
    required: bool = Field(True, description="Whether field is required")
    description: Optional[str] = Field(None, description="Field description")
    default_value: Optional[Any] = Field(None, description="Default value if not required")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value for numeric types")
    min_length: Optional[int] = Field(None, description="Minimum length for string/array types")
    max_length: Optional[int] = Field(None, description="Maximum length for string/array types")
    allowed_values: Optional[List[Any]] = Field(None, description="List of allowed values")
    pattern: Optional[str] = Field(None, description="Regex pattern for string validation")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('Field name must be alphanumeric with underscores')
        return v


class InputSchema(BaseModel):
    """Schema for model inputs"""
    fields: List[FieldSchema] = Field(..., description="List of input fields")
    batch_input: bool = Field(False, description="Whether model accepts batch inputs")
    
    @validator('fields')
    def validate_fields(cls, v):
        if not v:
            raise ValueError('At least one input field is required')
        
        # Check for duplicate field names
        names = [field.name for field in v]
        if len(names) != len(set(names)):
            raise ValueError('Field names must be unique')
        
        return v


class OutputSchema(BaseModel):
    """Schema for model outputs"""
    fields: List[FieldSchema] = Field(..., description="List of output fields")
    
    @validator('fields')
    def validate_fields(cls, v):
        if not v:
            raise ValueError('At least one output field is required')
        return v


# Model Upload and Creation
class ModelUpload(BaseModel):
    """Schema for model upload request"""
    name: str = Field(..., description="Model name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Model description", max_length=500)
    framework: ModelFramework = Field(..., description="ML framework used")
    model_type: ModelType = Field(..., description="Type of ML model")
    version: str = Field("1.0.0", description="Model version")
    tags: Optional[List[str]] = Field([], description="Model tags for categorization")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').replace(' ', '').isalnum():
            raise ValueError('Model name must be alphanumeric with spaces, underscores, or hyphens')
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        if v and len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
        return v or []


class ModelMetadata(BaseModel):
    """Complete model metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique model ID")
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    framework: ModelFramework = Field(..., description="ML framework")
    model_type: ModelType = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    status: ModelStatus = Field(ModelStatus.UPLOADED, description="Model status")
    tags: List[str] = Field([], description="Model tags")
    
    # File information
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="File content hash")
    file_path: Optional[str] = Field(None, description="Storage path")
    
    # Schema information
    input_schema: Optional[InputSchema] = Field(None, description="Input schema")
    output_schema: Optional[OutputSchema] = Field(None, description="Output schema")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")
    
    # Performance metrics (to be populated after deployment)
    prediction_count: int = Field(0, description="Total number of predictions made")
    avg_response_time: Optional[float] = Field(None, description="Average response time in seconds")
    last_prediction_at: Optional[datetime] = Field(None, description="Last prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "House Price Predictor",
                "description": "Predicts house prices based on features",
                "framework": "sklearn",
                "model_type": "regression",
                "version": "1.0.0",
                "tags": ["real-estate", "regression", "production"],
                "file_name": "house_price_model.pkl",
                "file_size": 1024000,
                "file_hash": "abc123def456"
            }
        }


class ModelSchemaUpdate(BaseModel):
    """Schema for updating model input/output schemas"""
    input_schema: Optional[InputSchema] = Field(None, description="Updated input schema")
    output_schema: Optional[OutputSchema] = Field(None, description="Updated output schema")


class ModelUpdate(BaseModel):
    """Schema for updating model metadata"""
    name: Optional[str] = Field(None, description="Updated model name")
    description: Optional[str] = Field(None, description="Updated description")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    status: Optional[ModelStatus] = Field(None, description="Updated status")


# Model Response Schemas
class ModelListResponse(BaseModel):
    """Response schema for model listing"""
    models: List[ModelMetadata] = Field(..., description="List of models")
    total: int = Field(..., description="Total number of models")
    page: int = Field(1, description="Current page")
    page_size: int = Field(10, description="Items per page")


class ModelResponse(BaseModel):
    """Response schema for single model"""
    model: ModelMetadata = Field(..., description="Model metadata")
    message: Optional[str] = Field(None, description="Response message")


class ModelUploadResponse(BaseModel):
    """Response schema for model upload"""
    model_id: str = Field(..., description="Uploaded model ID")
    message: str = Field(..., description="Upload status message")
    validation_status: str = Field(..., description="Validation status")


# Model Validation Schemas
class ModelValidationResult(BaseModel):
    """Result of model validation"""
    is_valid: bool = Field(..., description="Whether model is valid")
    framework_detected: Optional[ModelFramework] = Field(None, description="Detected framework")
    model_type_detected: Optional[ModelType] = Field(None, description="Detected model type")
    errors: List[str] = Field([], description="Validation errors")
    warnings: List[str] = Field([], description="Validation warnings")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata from validation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "framework_detected": "sklearn",
                "model_type_detected": "regression",
                "errors": [],
                "warnings": ["Model size is large, consider optimization"],
                "metadata": {
                    "sklearn_version": "1.3.0",
                    "feature_count": 10,
                    "model_class": "RandomForestRegressor"
                }
            }
        }


# Model Prediction Schemas
class PredictionRequest(BaseModel):
    """Schema for model prediction requests"""
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "square_feet": 2000,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "age": 10
                }
            }
        }


class PredictionResponse(BaseModel):
    """Schema for model prediction responses"""
    prediction: Any = Field(..., description="Model prediction")
    model_id: str = Field(..., description="Model ID used for prediction")
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique prediction ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    response_time: float = Field(..., description="Response time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 450000.0,
                "model_id": "abc123-def456-ghi789",
                "prediction_id": "pred-abc123",
                "timestamp": "2024-01-01T12:00:00Z",
                "response_time": 0.05
            }
        }


# Deployment Schemas
class ModelDeploymentCreate(BaseModel):
    """Schema for creating a new model deployment"""
    model_id: str = Field(..., description="Model ID to deploy")
    name: str = Field(..., description="Deployment name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Deployment description", max_length=500)
    config: Optional[Dict[str, Any]] = Field({}, description="Deployment configuration")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').replace(' ', '').isalnum():
            raise ValueError('Deployment name must be alphanumeric with spaces, underscores, or hyphens')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "abc123-def456-ghi789",
                "name": "Production House Price API",
                "description": "Production deployment for house price prediction",
                "config": {
                    "cpu_limit": "1000m",
                    "memory_limit": "1Gi",
                    "replicas": 2
                }
            }
        }


class ModelDeploymentUpdate(BaseModel):
    """Schema for updating a model deployment"""
    name: Optional[str] = Field(None, description="Updated deployment name")
    description: Optional[str] = Field(None, description="Updated deployment description")
    status: Optional[DeploymentStatus] = Field(None, description="Updated deployment status")
    config: Optional[Dict[str, Any]] = Field(None, description="Updated deployment configuration")


class ModelDeploymentResponse(BaseModel):
    """Schema for model deployment response"""
    id: str = Field(..., description="Deployment ID")
    model_id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Deployment name")
    description: Optional[str] = Field(None, description="Deployment description")
    status: DeploymentStatus = Field(..., description="Deployment status")
    endpoint_url: Optional[str] = Field(None, description="Deployment endpoint URL")
    service_name: Optional[str] = Field(None, description="BentoML service name")
    framework: Optional[str] = Field(None, description="Model framework")
    endpoints: List[str] = Field([], description="Available prediction endpoints")
    config: Dict[str, Any] = Field({}, description="Deployment configuration")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "deploy-abc123",
                "model_id": "abc123-def456-ghi789",
                "name": "Production House Price API",
                "description": "Production deployment for house price prediction",
                "status": "active",
                "endpoint_url": "http://localhost:3000/model_service_abc123",
                "service_name": "model_service_abc123",
                "framework": "sklearn",
                "endpoints": ["predict", "predict_proba"],
                "config": {"replicas": 2},
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z"
            }
        }


# Filter and Search Schemas
class ModelFilter(BaseModel):
    """Schema for filtering models"""
    framework: Optional[ModelFramework] = Field(None, description="Filter by framework")
    model_type: Optional[ModelType] = Field(None, description="Filter by model type")
    status: Optional[ModelStatus] = Field(None, description="Filter by status")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    
    
class ModelSearch(BaseModel):
    """Schema for searching models"""
    query: Optional[str] = Field(None, description="Search query")
    filters: Optional[ModelFilter] = Field(None, description="Additional filters")
    page: int = Field(1, description="Page number", ge=1)
    page_size: int = Field(10, description="Items per page", ge=1, le=100)
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('Sort order must be "asc" or "desc"')
        return v 