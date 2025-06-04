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