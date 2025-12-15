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



# Tests for Schema Validation

class TestSchemaValidation:
    """Test schema validation endpoints"""
    
    @patch('app.services.schema_service.schema_service.validate_input_schema')
    def test_validate_schema_success(self, mock_validate, schema_client, sample_input_schema):
        """Test successful schema validation"""
        mock_validate.return_value = (True, [])
        
        test_data = {
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft": 2000,
            "location": "suburban"
        }
        
        validation_request = {
            "schema": sample_input_schema,
            "data": test_data
        }
        
        response = schema_client.post("/api/v1/schemas/validate", json=validation_request)
        
        assert response.status_code == 200
        result = response.json()
        assert result["valid"] is True
        assert result["errors"] == []
        mock_validate.assert_called_once()
    
    @patch('app.services.schema_service.schema_service.validate_input_schema')
    def test_validate_schema_failure(self, mock_validate, schema_client, sample_input_schema):
        """Test schema validation failure"""
        mock_validate.return_value = (False, ["'bedrooms' is a required property"])
        
        test_data = {
            "bathrooms": 2.5,
            "sqft": 2000
            # Missing required 'bedrooms' field
        }
        
        validation_request = {
            "schema": sample_input_schema,
            "data": test_data
        }
        
        response = schema_client.post("/api/v1/schemas/validate", json=validation_request)
        
        assert response.status_code == 200
        result = response.json()
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "bedrooms" in result["errors"][0]
    
    @patch('app.services.schema_service.schema_service.validate_input_schema')
    def test_validate_schema_type_error(self, mock_validate, schema_client, sample_input_schema):
        """Test schema validation with type errors"""
        mock_validate.return_value = (False, ["'not_a_number' is not of type 'integer'"])
        
        test_data = {
            "bedrooms": "not_a_number",  # Should be integer
            "bathrooms": 2.5,
            "sqft": 2000
        }
        
        validation_request = {
            "schema": sample_input_schema,
            "data": test_data
        }
        
        response = schema_client.post("/api/v1/schemas/validate", json=validation_request)
        
        assert response.status_code == 200
        result = response.json()
        assert result["valid"] is False
        assert "type" in result["errors"][0].lower()
    
    def test_validate_schema_invalid_request(self, schema_client):
        """Test validation with invalid request data"""
        invalid_request = {
            "schema": "not_a_schema",  # Should be object
            "data": {}
        }
        
        response = schema_client.post("/api/v1/schemas/validate", json=invalid_request)
        
        assert response.status_code == 422
    
    @patch('app.services.schema_service.schema_service.validate_input_schema')
    def test_validate_schema_service_error(self, mock_validate, schema_client, sample_input_schema):
        """Test schema validation when service raises error"""
        mock_validate.side_effect = Exception("Schema service unavailable")
        
        validation_request = {
            "schema": sample_input_schema,
            "data": {"bedrooms": 3, "bathrooms": 2, "sqft": 2000}
        }
        
        response = schema_client.post("/api/v1/schemas/validate", json=validation_request)
        
        assert response.status_code == 500




