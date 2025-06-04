"""
Schema routes for model input/output schema management
Provides REST API endpoints for defining and managing model schemas
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio

from app.schemas.model import (
    InputSchema,
    OutputSchema,
    ModelSchemaUpdate,
    FieldSchema,
    DataType
)
from app.services.schema_service import schema_service

router = APIRouter()


# Request/Response models for schema endpoints
class SchemaValidationRequest(BaseModel):
    schema: Dict[str, Any]
    data: Dict[str, Any]

class SchemaValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []

class SchemaGenerationRequest(BaseModel):
    sample_data: List[Dict[str, Any]]
    schema_type: str = "input"

class SchemaComparisonRequest(BaseModel):
    schema1: Dict[str, Any]
    schema2: Dict[str, Any]

class SchemaVersionRequest(BaseModel):
    schema_data: Dict[str, Any]
    version: str
    description: Optional[str] = None
    migration_notes: Optional[str] = None

class ModelSchemaRequest(BaseModel):
    model_id: str
    schema_type: str
    schema_data: Dict[str, Any]
    version: str = "1.0"


# General schema endpoints (expected by tests)
@router.post("/validate", response_model=SchemaValidationResponse)
async def validate_schema(request: SchemaValidationRequest):
    """Validate data against a JSON schema"""
    try:
        is_valid, errors = await asyncio.to_thread(
            schema_service.validate_input_schema, 
            request.schema, 
            request.data
        )
        return SchemaValidationResponse(valid=is_valid, errors=errors)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schema validation error: {str(e)}"
        )

@router.post("/generate")
async def generate_schema_from_data(request: SchemaGenerationRequest):
    """Generate JSON schema from sample data"""
    try:
        if not request.sample_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sample data cannot be empty"
            )
        
        # Use the first sample to generate schema
        schema = schema_service.generate_schema_from_data(request.sample_data[0])
        
        return {
            "schema": schema,
            "schema_type": request.schema_type,
            "generated_from_samples": len(request.sample_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schema generation error: {str(e)}"
        )

@router.post("/compare")
async def compare_schemas(request: SchemaComparisonRequest):
    """Compare two schemas for compatibility"""
    try:
        result = await asyncio.to_thread(schema_service.compare_schemas, request.schema1, request.schema2)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schema comparison error: {str(e)}"
        )

@router.post("/convert")
async def convert_schema_format(request: Dict[str, Any] = Body(...)):
    """Convert schema between different formats"""
    try:
        schema = request.get("schema")
        target_format = request.get("target_format", "").lower()
        
        if not schema:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Schema is required"
            )
        
        converted_schema: Optional[Dict[str, Any]] = None # Initialize
        if target_format == "openapi":
            # converted = schema_service.convert_to_openapi_schema(schema)
            converted_schema = await asyncio.to_thread(schema_service.convert_to_openapi_schema, schema)
        elif target_format in ["json", "json_schema"]:
            # converted = schema_service.convert_to_json_schema(schema)
            converted_schema = await asyncio.to_thread(schema_service.convert_to_json_schema, schema)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {target_format}"
            )
        
        return {
            "converted_schema": converted_schema,
            "original_format": "json", # Assuming original is always JSON for this endpoint
            "target_format": target_format
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schema conversion error: {str(e)}"
        )

# Schema versioning endpoints
@router.get("/{schema_id}/versions")
async def get_schema_versions(schema_id: str):
    """Get all versions of a schema"""
    try:
        versions = await schema_service.get_schema_versions(schema_id)
        if not versions:
            pass
        return versions
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving schema versions: {str(e)}"
        )

@router.post("/{schema_id}/versions", status_code=status.HTTP_201_CREATED)
async def create_schema_version(schema_id: str, request: SchemaVersionRequest):
    """Create a new version of a schema"""
    try:
        result = await schema_service.create_schema_version(
            schema_id_base=schema_id, 
            schema_data=request.schema_data, 
            version=request.version, 
            description=request.description
        )
        return result
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating schema version: {str(e)}"
        )

# Model schema management endpoints
@router.get("/models/{model_id}")
async def get_model_schemas_general(model_id: str):
    """Get schemas for a model (general endpoint)"""
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model schemas: {str(e)}"
        )

@router.post("/models", status_code=status.HTTP_201_CREATED)
async def save_model_schema_general(request: ModelSchemaRequest):
    """Save a schema for a model (general endpoint)"""
    try:
        result = await schema_service.save_model_schema(
            request.model_id,
            request.schema_type,
            request.schema_data,
            request.version
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving model schema: {str(e)}"
        )

@router.put("/{schema_id}")
async def update_model_schema_general(schema_id: str, request: Dict[str, Any] = Body(...)):
    """Update a model schema (general endpoint)"""
    try:
        # Ensure schema_data is present in the request body
        schema_data = request.get("schema_data")
        if schema_data is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="'schema_data' field is required in the request body."
            )

        result = await schema_service.update_model_schema(
            schema_id,
            schema_data,
            request.get("version")
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating model schema: {str(e)}"
        )

@router.delete("/{schema_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_schema_general(schema_id: str):
    """Delete a model schema (general endpoint)"""
    try:
        success = await schema_service.delete_model_schema(schema_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema {schema_id} not found"
            )
        # Return nothing for 204 No Content
        return
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException as he:
        # Re-raise HTTPExceptions as-is
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model schema: {str(e)}"
        )


# Original model-specific schema endpoints
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
        "input_schema": openapi_input,
        "output_schema": openapi_output
    }


def _datatype_to_openapi_type(data_type: DataType) -> str:
    """Convert DataType enum to OpenAPI type string"""
    mapping = {
        DataType.INTEGER: "integer",
        DataType.FLOAT: "number",
        DataType.STRING: "string",
        DataType.BOOLEAN: "boolean",
        DataType.ARRAY: "array",
        DataType.OBJECT: "object",
        DataType.DATE: "string",
        DataType.DATETIME: "string"
    }
    return mapping.get(data_type, "string")


@router.get("/templates/common")
async def get_common_schema_templates():
    """
    Get common schema templates for typical ML use cases
    
    Returns pre-defined schema templates that can be used as starting points
    for common machine learning scenarios.
    """
    templates = {
        "house_price_prediction": {
            "input": {
                "type": "object",
                "properties": {
                    "bedrooms": {"type": "integer", "minimum": 1, "maximum": 10},
                    "bathrooms": {"type": "number", "minimum": 0.5, "maximum": 10},
                    "sqft": {"type": "number", "minimum": 100},
                    "location": {"type": "string", "enum": ["urban", "suburban", "rural"]}
                },
                "required": ["bedrooms", "bathrooms", "sqft"]
            },
            "output": {
                "type": "object", 
                "properties": {
                    "predicted_price": {"type": "number", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["predicted_price", "confidence"]
            }
        },
        "image_classification": {
            "input": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "format": "base64"},
                    "image_url": {"type": "string", "format": "uri"}
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "predicted_class": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "top_predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "class": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["predicted_class", "confidence"]
            }
        }
    }
    
    return {
        "templates": templates,
        "description": "Common schema templates for ML use cases"
    } 