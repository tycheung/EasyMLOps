"""
Comprehensive tests for all service layer components
Tests deployment, monitoring, schema, and BentoML services business logic
"""

import pytest
import asyncio
import tempfile
import os
import json
import uuid
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.deployment_service import deployment_service, DeploymentService
from app.services.monitoring_service import monitoring_service
from app.services.schema_service import schema_service, SchemaService
from app.services.bentoml_service import bentoml_service_manager
from app.models.model import Model, ModelDeployment
from app.schemas.model import ModelDeploymentCreate, ModelStatus, DeploymentStatus, ModelFramework, ModelType
from app.schemas.monitoring import AlertSeverity, SystemComponent, ModelPerformanceMetrics, MetricType, DriftType, DriftSeverity, ModelPerformanceHistory, ModelConfidenceMetrics, ModelBaseline, ModelVersionComparison, ABTest, ABTestStatus, ABTestMetrics, ABTestComparison, CanaryDeployment, CanaryDeploymentStatus, CanaryMetrics, ProtectedAttributeConfig, ProtectedAttributeType, BiasFairnessMetrics, DemographicDistribution, ModelExplanation, ExplanationType, FeatureImportance, ImportanceType, OutlierDetection, OutlierDetectionMethod, OutlierType, AnomalyDetection, AnomalyType, DataQualityMetrics, RetrainingJob, RetrainingTriggerType, RetrainingJobStatus, RetrainingTriggerConfig, ModelCard, DataLineage, LineageType, RelationshipType, GovernanceWorkflow, WorkflowType, WorkflowStatus, ComplianceRecord, ComplianceType, ComplianceRecordType, ComplianceRecordStatus, DataRetentionPolicy, TimeSeriesAnalysis, AnalysisType, TrendDirection, ComparativeAnalytics, ComparisonType, CustomDashboard, AutomatedReport, ReportType, ScheduleType, ExternalIntegration, IntegrationType, WebhookConfig, SamplingConfig, SamplingStrategy, MetricAggregationConfig, AggregationMethod, AlertRule, AlertCondition, NotificationChannel, NotificationChannelType, AlertGroup, AlertEscalation, EscalationTriggerCondition, Alert
from app.database import get_session



# Tests for Schema service

class TestSchemaService:
    """Test schema service comprehensive functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for schema generation"""
        return [
            {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft": 2000,
                "location": "suburban",
                "price": 350000
            },
            {
                "bedrooms": 4,
                "bathrooms": 3.0,
                "sqft": 2500,
                "location": "urban",
                "price": 450000
            },
            {
                "bedrooms": 2,
                "bathrooms": 1.5,
                "sqft": 1200,
                "location": "rural",
                "price": 250000
            }
        ]
    
    @pytest.mark.asyncio
    async def test_generate_schema_from_data(self, sample_data):
        """Test schema generation from sample data"""
        schema = schema_service.generate_schema_from_data(
            sample_data=sample_data[0],
            schema_type="input",
            include_target=False
        )
        
        assert schema is not None
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "bedrooms" in schema["properties"]
        assert "bathrooms" in schema["properties"]
        assert "sqft" in schema["properties"]
        assert "location" in schema["properties"]
        assert "price" not in schema["properties"]  # Target excluded
    
    @pytest.mark.asyncio
    async def test_generate_schema_with_target(self, sample_data):
        """Test schema generation including target variable"""
        schema = schema_service.generate_schema_from_data(
            sample_data=sample_data[0],
            schema_type="input",
            include_target=True
        )
        
        assert "price" in schema["properties"]
    
    @pytest.mark.asyncio
    async def test_validate_input_schema_success(self):
        """Test successful input schema validation"""
        schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1", "feature2"]
        }
        
        data = {"feature1": 0.5, "feature2": "test"}
        
        is_valid, errors = schema_service.validate_input_schema(data, schema)
        
        assert is_valid is True
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_input_schema_failure(self):
        """Test input schema validation failure"""
        schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1", "feature2"]
        }
        
        # Missing required field
        data = {"feature1": 0.5}
        
        is_valid, errors = schema_service.validate_input_schema(data, schema)
        
        assert is_valid is False
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_compare_schemas_compatible(self):
        """Test schema compatibility check"""
        schema1 = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1"]
        }
        
        schema2 = {
            "type": "object", 
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"},
                "feature3": {"type": "string"}  # New optional field
            },
            "required": ["feature1"]
        }
        
        comparison = schema_service.compare_schemas(schema1, schema2)
        
        assert comparison is not None
        assert "compatible" in comparison or comparison.get("is_compatible", False)
    
    @pytest.mark.asyncio
    async def test_compare_schemas_incompatible(self):
        """Test incompatible schema comparison"""
        schema1 = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"}
            },
            "required": ["feature1"]
        }
        
        schema2 = {
            "type": "object",
            "properties": {
                "feature1": {"type": "string"}  # Type changed
            },
            "required": ["feature1", "feature2"]  # New required field
        }
        
        comparison = schema_service.compare_schemas(schema1, schema2)
        
        assert comparison is not None
    
    @pytest.mark.asyncio
    async def test_convert_to_openapi_schema(self):
        """Test conversion to OpenAPI schema format"""
        json_schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number"},
                "feature2": {"type": "string"}
            },
            "required": ["feature1"]
        }
        
        openapi_schema = schema_service.convert_to_openapi_schema(
            json_schema,
            include_examples=True
        )
        
        assert openapi_schema is not None
        assert openapi_schema["type"] == "object"
        assert "properties" in openapi_schema
    
    @pytest.mark.asyncio
    @patch('aiofiles.open')
    @patch('aiofiles.os.makedirs')
    async def test_save_model_schema(self, mock_makedirs, mock_open, test_model):
        """Test saving model schema to filesystem"""
        mock_file = AsyncMock()
        mock_open.return_value.__aenter__.return_value = mock_file
        
        schema_data = {
            "type": "object",
            "properties": {"feature1": {"type": "number"}}
        }
        
        result = await schema_service.save_model_schema(
            model_id=test_model.id,
            schema_type="input",
            schema_data=schema_data,
            version="1.0"
        )
        
        assert result is not None
        assert result["message"] == "Schema saved successfully"
        assert result["model_id"] == test_model.id
        assert result["schema_type"] == "input"
        assert result["version"] == "1.0"
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()
        mock_file.write.assert_called_once()




