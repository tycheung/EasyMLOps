"""
Comprehensive tests for schema routes
Tests all schema REST API endpoints including validation, generation, conversion, and management
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status
from fastapi.testclient import TestClient

# Import the get_test_app function instead of the non-existent app
from tests.conftest import get_test_app


@pytest.fixture
def schema_client():
    """Create test client for schema routes"""
    return TestClient(get_test_app())


@pytest.fixture
def sample_input_schema():
    """Sample input schema for testing"""
    return {
        "type": "object",
        "properties": {
            "bedrooms": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of bedrooms"
            },
            "bathrooms": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 10,
                "description": "Number of bathrooms"
            },
            "sqft": {
                "type": "number",
                "minimum": 100,
                "description": "Square footage"
            },
            "location": {
                "type": "string",
                "enum": ["urban", "suburban", "rural"],
                "description": "Property location type"
            }
        },
        "required": ["bedrooms", "bathrooms", "sqft"],
        "additionalProperties": False
    }


@pytest.fixture
def sample_output_schema():
    """Sample output schema for testing"""
    return {
        "type": "object",
        "properties": {
            "predicted_price": {
                "type": "number",
                "minimum": 0,
                "description": "Predicted house price"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Prediction confidence score"
            },
            "price_range": {
                "type": "object",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"}
                },
                "description": "Predicted price range"
            }
        },
        "required": ["predicted_price", "confidence"]
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for schema generation"""
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



# Tests for Schema Management

class TestModelSchemaManagement:
    """Test model schema management endpoints"""
    
    @patch('app.services.schema_service.schema_service.get_model_schemas')
    def test_get_model_schemas_success(self, mock_get_schemas, schema_client, test_model):
        """Test retrieving model schemas"""
        from app.schemas.model import InputSchema, OutputSchema, FieldSchema, DataType
        
        # Mock input schema
        input_schema = InputSchema(fields=[
            FieldSchema(
                name="feature1",
                data_type=DataType.FLOAT,
                required=True,
                description="First feature"
            )
        ])
        
        # Mock output schema
        output_schema = OutputSchema(fields=[
            FieldSchema(
                name="prediction",
                data_type=DataType.FLOAT,
                required=True,
                description="Model prediction"
            )
        ])
        
        mock_get_schemas.return_value = (input_schema, output_schema)
        
        response = schema_client.get(f"/api/v1/schemas/models/{test_model.id}")
        
        assert response.status_code == 200
        result = response.json()
        assert "input_schema" in result
        assert "output_schema" in result
        assert result["model_id"] == test_model.id
        mock_get_schemas.assert_called_once_with(test_model.id)
    
    @patch('app.services.schema_service.schema_service.get_model_schemas')
    def test_get_model_schemas_not_found(self, mock_get_schemas, schema_client):
        """Test retrieving schemas for non-existent model"""
        mock_get_schemas.return_value = (None, None)
        
        response = schema_client.get("/api/v1/schemas/models/nonexistent")
        
        assert response.status_code == 404
    
    @patch('app.services.schema_service.schema_service.save_model_schema')
    def test_save_model_schema_success(self, mock_save, schema_client, test_model, sample_input_schema):
        """Test saving model schema"""
        mock_save.return_value = {
            "id": "schema_123",
            "model_id": test_model.id,
            "schema_type": "input",
            "schema_data": sample_input_schema,
            "version": "1.0"
        }
        
        schema_request = {
            "model_id": test_model.id,
            "schema_type": "input",
            "schema_data": sample_input_schema,
            "version": "1.0",
            "description": "Input schema for house price prediction model"
        }
        
        response = schema_client.post("/api/v1/schemas/models", json=schema_request)
        
        assert response.status_code == 201
        result = response.json()
        assert result["id"] == "schema_123"
        assert result["model_id"] == test_model.id
        assert result["schema_type"] == "input"
        mock_save.assert_called_once()
    
    @patch('app.services.schema_service.schema_service.update_model_schema')
    def test_update_model_schema_success(self, mock_update, schema_client, test_model, sample_input_schema):
        """Test updating model schema"""
        updated_schema = sample_input_schema.copy()
        updated_schema["properties"]["new_field"] = {"type": "string"}
        
        mock_update.return_value = {
            "id": "schema_123",
            "model_id": test_model.id,
            "schema_type": "input",
            "schema_data": updated_schema,
            "version": "1.1"
        }
        
        update_request = {
            "schema_data": updated_schema,
            "version": "1.1",
            "description": "Updated input schema"
        }
        
        response = schema_client.put("/api/v1/schemas/schema_123", json=update_request)
        
        assert response.status_code == 200
        result = response.json()
        assert result["version"] == "1.1"
        assert "new_field" in result["schema_data"]["properties"]
        mock_update.assert_called_once()
    
    @patch('app.services.schema_service.schema_service.delete_model_schema')
    def test_delete_model_schema_success(self, mock_delete, schema_client):
        """Test deleting model schema"""
        mock_delete.return_value = True
        
        response = schema_client.delete("/api/v1/schemas/schema_123")
        
        assert response.status_code == 204
        mock_delete.assert_called_once_with("schema_123")
    
    @patch('app.services.schema_service.schema_service.delete_model_schema')
    def test_delete_model_schema_not_found(self, mock_delete, schema_client):
        """Test deleting non-existent schema"""
        mock_delete.return_value = False
        
        response = schema_client.delete("/api/v1/schemas/nonexistent")
        
        assert response.status_code == 404




class TestSchemaVersioning:
    """Test schema versioning endpoints"""
    
    @patch('app.services.schema_service.schema_service.get_schema_versions')
    def test_get_schema_versions(self, mock_versions, schema_client):
        """Test retrieving schema versions"""
        mock_versions.return_value = [
            {
                "version": "1.0",
                "created_at": "2024-01-01T00:00:00Z",
                "description": "Initial schema",
                "is_active": False
            },
            {
                "version": "1.1",
                "created_at": "2024-01-02T00:00:00Z",
                "description": "Added optional fields",
                "is_active": True
            }
        ]
        
        response = schema_client.get("/api/v1/schemas/schema_123/versions")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2
        assert result[0]["version"] == "1.0"
        assert result[1]["is_active"] is True
    
    @patch('app.services.schema_service.schema_service.create_schema_version')
    def test_create_schema_version(self, mock_create, schema_client, sample_input_schema):
        """Test creating new schema version"""
        mock_create.return_value = {
            "id": "schema_123",
            "model_id": "model_456",
            "schema_type": "input",
            "schema_data": sample_input_schema,
            "version": "2.0"
        }
        
        version_request = {
            "schema_data": sample_input_schema,
            "version": "2.0",
            "description": "Major schema update",
            "migration_notes": "Removed deprecated fields"
        }
        
        response = schema_client.post("/api/v1/schemas/schema_123/versions", json=version_request)
        
        assert response.status_code == 201
        result = response.json()
        assert result["version"] == "2.0"
        mock_create.assert_called_once()




