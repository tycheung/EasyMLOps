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



# Tests for Schema Generation

class TestSchemaGeneration:
    """Test schema generation endpoints"""
    
    @patch('app.services.schema_service.schema_service.generate_schema_from_data')
    def test_generate_schema_from_samples_success(self, mock_generate, schema_client, sample_training_data, sample_input_schema):
        """Test successful schema generation from sample data"""
        mock_generate.return_value = sample_input_schema
        
        generation_request = {
            "sample_data": sample_training_data,
            "schema_type": "input",
            "include_target": False
        }
        
        response = schema_client.post("/api/v1/schemas/generate", json=generation_request)
        
        assert response.status_code == 200
        result = response.json()
        assert "schema" in result
        assert result["schema"]["type"] == "object"
        assert "properties" in result["schema"]
        assert "bedrooms" in result["schema"]["properties"]
        mock_generate.assert_called_once()
    
    @patch('app.services.schema_service.schema_service.generate_schema_from_data')
    def test_generate_output_schema(self, mock_generate, schema_client, sample_output_schema):
        """Test generating output schema"""
        mock_generate.return_value = sample_output_schema
        
        sample_outputs = [
            {"predicted_price": 350000, "confidence": 0.85},
            {"predicted_price": 450000, "confidence": 0.92},
            {"predicted_price": 250000, "confidence": 0.78}
        ]
        
        generation_request = {
            "sample_data": sample_outputs,
            "schema_type": "output"
        }
        
        response = schema_client.post("/api/v1/schemas/generate", json=generation_request)
        
        assert response.status_code == 200
        result = response.json()
        assert "schema" in result
        assert "predicted_price" in result["schema"]["properties"]
        assert "confidence" in result["schema"]["properties"]
    
    @patch('app.services.schema_service.schema_service.generate_schema_from_data')
    def test_generate_schema_with_options(self, mock_generate, schema_client, sample_training_data, sample_input_schema):
        """Test schema generation with custom options"""
        mock_generate.return_value = sample_input_schema
        
        generation_request = {
            "sample_data": sample_training_data,
            "schema_type": "input",
            "include_target": False,
            "options": {
                "infer_enum_threshold": 5,
                "include_descriptions": True,
                "strict_types": True
            }
        }
        
        response = schema_client.post("/api/v1/schemas/generate", json=generation_request)
        
        assert response.status_code == 200
        mock_generate.assert_called_once()
    
    def test_generate_schema_empty_data(self, schema_client):
        """Test schema generation with empty sample data"""
        generation_request = {
            "sample_data": [],
            "schema_type": "input"
        }
        
        response = schema_client.post("/api/v1/schemas/generate", json=generation_request)
        
        assert response.status_code == 400
        result = response.json()
        assert "empty" in result["error"]["message"].lower()
    
    @patch('app.services.schema_service.schema_service.generate_schema_from_data')
    def test_generate_schema_service_error(self, mock_generate, schema_client, sample_training_data):
        """Test schema generation when service fails"""
        mock_generate.side_effect = Exception("Failed to generate schema")
        
        generation_request = {
            "sample_data": sample_training_data,
            "schema_type": "input"
        }
        
        response = schema_client.post("/api/v1/schemas/generate", json=generation_request)
        
        assert response.status_code == 500




