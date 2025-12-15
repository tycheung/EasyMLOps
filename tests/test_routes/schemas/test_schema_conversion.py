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



# Tests for Schema Conversion

class TestSchemaConversion:
    """Test schema conversion endpoints"""
    
    @patch('app.services.schema_service.schema_service.convert_to_openapi_schema')
    def test_convert_to_openapi_success(self, mock_convert, schema_client, sample_input_schema):
        """Test successful conversion to OpenAPI schema"""
        openapi_schema = {
            "type": "object",
            "properties": sample_input_schema["properties"],
            "required": sample_input_schema["required"],
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft": 2000,
                "location": "suburban"
            }
        }
        mock_convert.return_value = openapi_schema
        
        conversion_request = {
            "schema": sample_input_schema,
            "target_format": "openapi",
            "include_examples": True
        }
        
        response = schema_client.post("/api/v1/schemas/convert", json=conversion_request)
        
        assert response.status_code == 200
        result = response.json()
        assert "converted_schema" in result
        assert "example" in result["converted_schema"]
        mock_convert.assert_called_once()
    
    @patch('app.services.schema_service.schema_service.convert_to_json_schema')
    def test_convert_to_json_schema(self, mock_convert, schema_client, sample_input_schema):
        """Test conversion to JSON Schema format"""
        mock_convert.return_value = sample_input_schema
        
        conversion_request = {
            "schema": sample_input_schema,
            "target_format": "json_schema"
        }
        
        response = schema_client.post("/api/v1/schemas/convert", json=conversion_request)
        
        assert response.status_code == 200
        result = response.json()
        assert "converted_schema" in result
        mock_convert.assert_called_once()
    
    def test_convert_unsupported_format(self, schema_client, sample_input_schema):
        """Test conversion to unsupported format"""
        conversion_request = {
            "schema": sample_input_schema,
            "target_format": "unsupported_format"
        }
        
        response = schema_client.post("/api/v1/schemas/convert", json=conversion_request)
        
        assert response.status_code == 400
        result = response.json()
        assert "unsupported" in result["error"]["message"].lower()
    
    @patch('app.services.schema_service.schema_service.convert_to_openapi_schema')
    def test_convert_schema_service_error(self, mock_convert, schema_client, sample_input_schema):
        """Test schema conversion when service fails"""
        mock_convert.side_effect = Exception("Conversion failed")
        
        conversion_request = {
            "schema": sample_input_schema,
            "target_format": "openapi"
        }
        
        response = schema_client.post("/api/v1/schemas/convert", json=conversion_request)
        
        assert response.status_code == 500




