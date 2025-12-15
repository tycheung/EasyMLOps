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



# Tests for Schema Comparison

class TestSchemaComparison:
    """Test schema comparison endpoints"""
    
    @patch('app.services.schema_service.schema_service.compare_schemas')
    def test_compare_schemas_compatible(self, mock_compare, schema_client, sample_input_schema):
        """Test comparing compatible schemas"""
        # Slightly modified schema (added optional field)
        modified_schema = sample_input_schema.copy()
        modified_schema["properties"]["garage"] = {
            "type": "boolean",
            "description": "Has garage"
        }
        
        mock_compare.return_value = {
            "compatible": True,
            "compatibility_score": 0.95,
            "differences": [
                {
                    "type": "added_field",
                    "field": "garage",
                    "severity": "minor"
                }
            ],
            "breaking_changes": []
        }
        
        comparison_request = {
            "schema1": sample_input_schema,
            "schema2": modified_schema,
            "strict_comparison": False
        }
        
        response = schema_client.post("/api/v1/schemas/compare", json=comparison_request)
        
        assert response.status_code == 200
        result = response.json()
        assert result["compatible"] is True
        assert result["compatibility_score"] > 0.9
        assert len(result["differences"]) > 0
        assert len(result["breaking_changes"]) == 0
        mock_compare.assert_called_once()
    
    @patch('app.services.schema_service.schema_service.compare_schemas')
    def test_compare_schemas_incompatible(self, mock_compare, schema_client, sample_input_schema):
        """Test comparing incompatible schemas"""
        # Incompatible schema (changed required field type)
        incompatible_schema = sample_input_schema.copy()
        incompatible_schema["properties"]["bedrooms"]["type"] = "string"
        
        mock_compare.return_value = {
            "compatible": False,
            "compatibility_score": 0.3,
            "differences": [
                {
                    "type": "type_change",
                    "field": "bedrooms",
                    "old_type": "integer",
                    "new_type": "string",
                    "severity": "major"
                }
            ],
            "breaking_changes": [
                {
                    "type": "required_field_type_change",
                    "field": "bedrooms",
                    "description": "Type change in required field"
                }
            ]
        }
        
        comparison_request = {
            "schema1": sample_input_schema,
            "schema2": incompatible_schema,
            "strict_comparison": True
        }
        
        response = schema_client.post("/api/v1/schemas/compare", json=comparison_request)
        
        assert response.status_code == 200
        result = response.json()
        assert result["compatible"] is False
        assert result["compatibility_score"] < 0.5
        assert len(result["breaking_changes"]) > 0
    
    @patch('app.services.schema_service.schema_service.compare_schemas')
    def test_compare_identical_schemas(self, mock_compare, schema_client, sample_input_schema):
        """Test comparing identical schemas"""
        mock_compare.return_value = {
            "compatible": True,
            "compatibility_score": 1.0,
            "differences": [],
            "breaking_changes": []
        }
        
        comparison_request = {
            "schema1": sample_input_schema,
            "schema2": sample_input_schema,
            "strict_comparison": True
        }
        
        response = schema_client.post("/api/v1/schemas/compare", json=comparison_request)
        
        assert response.status_code == 200
        result = response.json()
        assert result["compatible"] is True
        assert result["compatibility_score"] == 1.0
        assert len(result["differences"]) == 0




