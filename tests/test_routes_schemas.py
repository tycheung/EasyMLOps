"""
Comprehensive tests for schema routes
Tests all schema REST API endpoints including validation, generation, conversion, and management

This file has been refactored into domain-specific test files in tests/test_routes/schemas/:
- test_schema_validation.py: Schema validation tests
- test_schema_generation.py: Schema generation tests
- test_schema_conversion.py: Schema conversion tests
- test_schema_comparison.py: Schema comparison tests
- test_schema_management.py: Schema management and versioning tests
- test_schema_errors.py: Error handling and integration tests

This file maintains backward compatibility by re-exporting all test classes.
"""

# Re-export all test classes for backward compatibility
from tests.test_routes.schemas.test_schema_validation import TestSchemaValidation
from tests.test_routes.schemas.test_schema_generation import TestSchemaGeneration
from tests.test_routes.schemas.test_schema_conversion import TestSchemaConversion
from tests.test_routes.schemas.test_schema_comparison import TestSchemaComparison
from tests.test_routes.schemas.test_schema_management import (
    TestModelSchemaManagement,
    TestSchemaVersioning,
)
from tests.test_routes.schemas.test_schema_errors import (
    TestSchemaErrorHandling,
    TestSchemaIntegration,
)

__all__ = [
    "TestSchemaValidation",
    "TestSchemaGeneration",
    "TestSchemaConversion",
    "TestSchemaComparison",
    "TestModelSchemaManagement",
    "TestSchemaVersioning",
    "TestSchemaErrorHandling",
    "TestSchemaIntegration",
]

# Keep fixtures for backward compatibility
import pytest
from tests.conftest import get_test_app
from fastapi.testclient import TestClient

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
