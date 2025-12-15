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



# Tests for Schema Errors

class TestSchemaErrorHandling:
    """Test error handling in schema routes"""
    
    def test_validate_schema_malformed_json(self, schema_client):
        """Test validation with malformed JSON schema"""
        malformed_request = {
            "schema": "not_valid_json_schema",
            "data": {"test": "data"}
        }
        
        response = schema_client.post("/api/v1/schemas/validate", json=malformed_request)
        
        assert response.status_code == 422
    
    def test_generate_schema_invalid_sample_data(self, schema_client):
        """Test schema generation with invalid sample data"""
        invalid_request = {
            "sample_data": "not_a_list",
            "schema_type": "input"
        }
        
        response = schema_client.post("/api/v1/schemas/generate", json=invalid_request)
        
        assert response.status_code == 422
    
    @patch('app.services.schema_service.schema_service.get_model_schemas')
    def test_database_error_handling(self, mock_get_schemas, schema_client):
        """Test handling database errors"""
        mock_get_schemas.side_effect = Exception("Database connection failed")
        
        response = schema_client.get("/api/v1/schemas/models/test_model")
        
        assert response.status_code == 500
    
    def test_missing_required_fields(self, schema_client):
        """Test handling missing required fields"""
        incomplete_request = {
            "schema": {"type": "object"},
            # Missing 'data' field
        }
        
        response = schema_client.post("/api/v1/schemas/validate", json=incomplete_request)
        
        assert response.status_code == 422




class TestSchemaIntegration:
    """Integration tests for schema workflows"""
    
    @patch('app.services.schema_service.schema_service.generate_schema_from_data')
    @patch('app.services.schema_service.schema_service.validate_input_schema')
    @patch('app.services.schema_service.schema_service.save_model_schema')
    def test_complete_schema_workflow(self, mock_save, mock_validate, mock_generate, 
                                    schema_client, sample_training_data, sample_input_schema, test_model):
        """Test complete schema workflow: generate -> validate -> save"""
        # Mock schema generation
        mock_generate.return_value = sample_input_schema
        
        # Mock validation
        mock_validate.return_value = (True, [])
        
        # Mock saving
        mock_save.return_value = {
            "id": "schema_123",
            "model_id": test_model.id,
            "schema_type": "input",
            "schema_data": sample_input_schema,
            "version": "1.0"
        }
        
        # Generate schema
        generation_request = {
            "sample_data": sample_training_data,
            "schema_type": "input"
        }
        gen_response = schema_client.post("/api/v1/schemas/generate", json=generation_request)
        assert gen_response.status_code == 200
        generated_schema = gen_response.json()["schema"]
        
        # Validate schema
        validation_request = {
            "schema": generated_schema,
            "data": sample_training_data[0]
        }
        val_response = schema_client.post("/api/v1/schemas/validate", json=validation_request)
        assert val_response.status_code == 200
        assert val_response.json()["valid"] is True
        
        # Save schema
        save_request = {
            "model_id": test_model.id,
            "schema_type": "input",
            "schema_data": generated_schema,
            "version": "1.0"
        }
        save_response = schema_client.post("/api/v1/schemas/models", json=save_request)
        assert save_response.status_code == 201
        
        # Verify all services were called
        mock_generate.assert_called_once()
        mock_validate.assert_called_once()
        mock_save.assert_called_once() 

