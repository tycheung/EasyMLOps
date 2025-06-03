"""
Schema routes for model input/output schema management
Provides REST API endpoints for defining and managing model schemas
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.model import (
    InputSchema,
    OutputSchema,
    ModelSchemaUpdate,
    FieldSchema,
    DataType
)
from app.services.schema_service import schema_service

router = APIRouter()


@router.post("/{model_id}/schemas", status_code=status.HTTP_201_CREATED)
async def create_model_schemas(
    model_id: str, 
    input_schema: InputSchema, 
    output_schema: OutputSchema
):
    """
    Create input and output schemas for a model
    
    Define the expected input fields and output fields for a model.
    This enables automatic validation and API documentation generation.
    
    Example input schema:
    ```
    {
        "fields": [
            {
                "name": "house_size",
                "data_type": "float",
                "required": true,
                "description": "House size in square feet",
                "min_value": 0,
                "max_value": 10000
            },
            {
                "name": "bedrooms",
                "data_type": "integer",
                "required": true,
                "description": "Number of bedrooms",
                "min_value": 1,
                "max_value": 10
            }
        ]
    }
    ```
    """
    success, message = await schema_service.create_model_schemas(
        model_id, input_schema, output_schema
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"message": message, "model_id": model_id}
    )


@router.get("/{model_id}/schemas")
async def get_model_schemas(model_id: str):
    """
    Get input and output schemas for a model
    
    Returns the defined input and output field schemas for the model,
    including validation rules, data types, and descriptions.
    """
    input_schema, output_schema = await schema_service.get_model_schemas(model_id)
    
    if input_schema is None and output_schema is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No schemas found for model {model_id}"
        )
    
    return {
        "model_id": model_id,
        "input_schema": input_schema.dict() if input_schema else None,
        "output_schema": output_schema.dict() if output_schema else None
    }


@router.patch("/{model_id}/schemas")
async def update_model_schemas(model_id: str, schema_update: ModelSchemaUpdate):
    """
    Update model schemas
    
    Update the input and/or output schemas for a model.
    You can update just the input schema, just the output schema, or both.
    """
    success, message = await schema_service.update_model_schemas(model_id, schema_update)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": message, "model_id": model_id}
    )


@router.delete("/{model_id}/schemas")
async def delete_model_schemas(model_id: str):
    """
    Delete all schemas for a model
    
    Removes both input and output schema definitions for the model.
    This will disable automatic validation for prediction requests.
    """
    success, message = await schema_service.delete_model_schemas(model_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": message}
    )


@router.get("/{model_id}/schemas/example")
async def get_model_example_data(model_id: str):
    """
    Get example input data for a model
    
    Returns example data that matches the model's input schema.
    Useful for testing and documentation purposes.
    """
    example_data = await schema_service.get_model_example_data(model_id)
    
    return {
        "model_id": model_id,
        "example_input": example_data,
        "description": "Example data based on the model's input schema"
    }


@router.post("/{model_id}/schemas/validate")
async def validate_data_against_schema(model_id: str, data: Dict[str, Any]):
    """
    Validate data against model schema
    
    Test whether input data conforms to the model's defined input schema.
    Returns validation results and any errors found.
    
    Example request:
    ```
    {
        "house_size": 2000.5,
        "bedrooms": 3,
        "location": "downtown"
    }
    ```
    """
    is_valid, message, validated_data = await schema_service.validate_prediction_data(
        model_id, data
    )
    
    return {
        "model_id": model_id,
        "is_valid": is_valid,
        "message": message,
        "validated_data": validated_data,
        "original_data": data
    }


@router.get("/{model_id}/schemas/openapi")
async def get_model_openapi_schema(model_id: str):
    """
    Get OpenAPI schema for model endpoints
    
    Returns OpenAPI-compatible schema definitions for the model's
    input and output, suitable for API documentation generation.
    """
    input_schema, output_schema = await schema_service.get_model_schemas(model_id)
    
    if input_schema is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No input schema found for model {model_id}"
        )
    
    # Generate OpenAPI schema format
    openapi_input = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for field in input_schema.fields:
        field_schema = {
            "type": _datatype_to_openapi_type(field.data_type),
            "description": field.description or f"{field.name} field"
        }
        
        # Add validation constraints
        if field.min_value is not None:
            field_schema["minimum"] = field.min_value
        if field.max_value is not None:
            field_schema["maximum"] = field.max_value
        if field.min_length is not None:
            field_schema["minLength"] = field.min_length
        if field.max_length is not None:
            field_schema["maxLength"] = field.max_length
        if field.pattern is not None:
            field_schema["pattern"] = field.pattern
        if field.allowed_values is not None:
            field_schema["enum"] = field.allowed_values
        
        openapi_input["properties"][field.name] = field_schema
        
        if field.required:
            openapi_input["required"].append(field.name)
    
    # Generate output schema if available
    openapi_output = None
    if output_schema:
        openapi_output = {
            "type": "object",
            "properties": {}
        }
        
        for field in output_schema.fields:
            openapi_output["properties"][field.name] = {
                "type": _datatype_to_openapi_type(field.data_type),
                "description": field.description or f"{field.name} output"
            }
    
    return {
        "model_id": model_id,
        "openapi_schema": {
            "input": openapi_input,
            "output": openapi_output
        }
    }


def _datatype_to_openapi_type(data_type: DataType) -> str:
    """Convert DataType enum to OpenAPI type string"""
    type_mapping = {
        DataType.INTEGER: "integer",
        DataType.FLOAT: "number",
        DataType.STRING: "string",
        DataType.BOOLEAN: "boolean",
        DataType.ARRAY: "array",
        DataType.OBJECT: "object",
        DataType.DATE: "string",
        DataType.DATETIME: "string"
    }
    return type_mapping.get(data_type, "string")


# Schema template endpoints for easier setup
@router.get("/templates/common")
async def get_common_schema_templates():
    """
    Get common schema templates
    
    Returns pre-defined schema templates for common use cases
    to help users quickly set up schemas for their models.
    """
    templates = {
        "house_price_prediction": {
            "description": "Schema for house price prediction model",
            "input_schema": {
                "fields": [
                    {
                        "name": "square_feet",
                        "data_type": "float",
                        "required": True,
                        "description": "House size in square feet",
                        "min_value": 100,
                        "max_value": 10000
                    },
                    {
                        "name": "bedrooms",
                        "data_type": "integer",
                        "required": True,
                        "description": "Number of bedrooms",
                        "min_value": 1,
                        "max_value": 10
                    },
                    {
                        "name": "bathrooms",
                        "data_type": "float",
                        "required": True,
                        "description": "Number of bathrooms",
                        "min_value": 0.5,
                        "max_value": 10
                    },
                    {
                        "name": "age",
                        "data_type": "integer",
                        "required": True,
                        "description": "House age in years",
                        "min_value": 0,
                        "max_value": 200
                    },
                    {
                        "name": "location",
                        "data_type": "string",
                        "required": False,
                        "description": "House location/neighborhood",
                        "max_length": 100
                    }
                ]
            },
            "output_schema": {
                "fields": [
                    {
                        "name": "predicted_price",
                        "data_type": "float",
                        "description": "Predicted house price in USD"
                    },
                    {
                        "name": "confidence",
                        "data_type": "float",
                        "description": "Prediction confidence score"
                    }
                ]
            }
        },
        "classification": {
            "description": "Schema for binary classification model",
            "input_schema": {
                "fields": [
                    {
                        "name": "feature1",
                        "data_type": "float",
                        "required": True,
                        "description": "First feature"
                    },
                    {
                        "name": "feature2",
                        "data_type": "float",
                        "required": True,
                        "description": "Second feature"
                    }
                ]
            },
            "output_schema": {
                "fields": [
                    {
                        "name": "prediction",
                        "data_type": "integer",
                        "description": "Predicted class (0 or 1)"
                    },
                    {
                        "name": "probability",
                        "data_type": "float",
                        "description": "Prediction probability"
                    }
                ]
            }
        },
        "text_analysis": {
            "description": "Schema for text analysis/sentiment model",
            "input_schema": {
                "fields": [
                    {
                        "name": "text",
                        "data_type": "string",
                        "required": True,
                        "description": "Input text to analyze",
                        "min_length": 1,
                        "max_length": 5000
                    },
                    {
                        "name": "language",
                        "data_type": "string",
                        "required": False,
                        "description": "Text language",
                        "allowed_values": ["en", "es", "fr", "de", "it"],
                        "default_value": "en"
                    }
                ]
            },
            "output_schema": {
                "fields": [
                    {
                        "name": "sentiment",
                        "data_type": "string",
                        "description": "Detected sentiment"
                    },
                    {
                        "name": "confidence",
                        "data_type": "float",
                        "description": "Confidence score"
                    }
                ]
            }
        }
    }
    
    return {
        "templates": templates,
        "description": "Common schema templates for quick model setup"
    } 