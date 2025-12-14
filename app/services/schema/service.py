"""
Schema service for dynamic model schema generation and validation
Handles user-defined input/output schemas and generates corresponding validation models

This module has been refactored into submodules:
- app.services.schema.crud: CRUD operations for model schemas
- app.services.schema.validation: Validation and dynamic model generation
- app.services.schema.utils: Schema utility functions
- app.services.schema.comparison: Schema comparison and compatibility
- app.services.schema.versioning: Schema versioning operations

This file maintains backward compatibility by providing a facade that delegates to sub-modules.
"""

from typing import Dict, List, Optional, Tuple, Any, Type
import logging

from pydantic import BaseModel

from app.services.schema.crud import SchemaCRUD
from app.services.schema.validation import SchemaValidation
from app.services.schema.utils import SchemaUtils
from app.services.schema.comparison import SchemaComparison
from app.services.schema.versioning import SchemaVersioning
from app.schemas.model import (
    InputSchema, 
    OutputSchema, 
    ModelSchemaUpdate
)

logger = logging.getLogger(__name__)


class SchemaService:
    """Service for managing model input/output schemas - Facade pattern"""
    
    def __init__(self):
        """Initialize all sub-services"""
        self.crud = SchemaCRUD()
        self.validation = SchemaValidation(self.crud)
        self.utils = SchemaUtils()
        self.comparison = SchemaComparison()
        self.versioning = SchemaVersioning()
    
    # CRUD operations - delegate to SchemaCRUD
    async def create_model_schemas(
        self, 
        model_id: str, 
        input_schema: InputSchema, 
        output_schema: OutputSchema
    ) -> Tuple[bool, str]:
        """Create input and output schemas for a model"""
        return await self.crud.create_model_schemas(model_id, input_schema, output_schema)
    
    async def get_model_schemas(self, model_id: str) -> Tuple[Optional[InputSchema], Optional[OutputSchema]]:
        """Get input and output schemas for a model"""
        return await self.crud.get_model_schemas(model_id)
    
    async def update_model_schemas(
        self, 
        model_id: str, 
        schema_update: ModelSchemaUpdate
    ) -> Tuple[bool, str]:
        """Update model schemas"""
        return await self.crud.update_model_schemas(model_id, schema_update)
    
    async def delete_model_schemas(self, model_id: str) -> Tuple[bool, str]:
        """Delete all schemas for a model"""
        return await self.crud.delete_model_schemas(model_id)
    
    # Validation operations - delegate to SchemaValidation
    async def generate_dynamic_validation_model(self, model_id: str) -> Tuple[Optional[Type[BaseModel]], Optional[Type[BaseModel]]]:
        """Generate dynamic Pydantic models for validation"""
        return await self.validation.generate_dynamic_validation_model(model_id)
    
    async def get_model_example_data(self, model_id: str) -> Dict[str, Any]:
        """Get example input data for a model"""
        return await self.validation.get_model_example_data(model_id)
    
    async def validate_prediction_data(self, model_id: str, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate data against model schema"""
        return await self.validation.validate_prediction_data(model_id, data)
    
    # Utility operations - delegate to SchemaUtils
    def generate_schema_from_data(self, sample_data: Any, schema_type: str = "input", include_target: bool = False) -> Dict[str, Any]:
        """Generate JSON schema from sample data"""
        return self.utils.generate_schema_from_data(sample_data, schema_type, include_target)
    
    def validate_input_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against a schema dictionary"""
        return self.utils.validate_input_schema(data, schema)
    
    def merge_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two schemas"""
        return self.utils.merge_schemas(schema1, schema2)
    
    def convert_to_openapi_schema(self, json_schema: Dict[str, Any], include_examples: bool = False) -> Dict[str, Any]:
        """Convert JSON schema to OpenAPI format"""
        return self.utils.convert_to_openapi_schema(json_schema, include_examples)
    
    def convert_to_json_schema(self, openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAPI schema to JSON schema format"""
        return self.utils.convert_to_json_schema(openapi_schema)
    
    # Comparison operations - delegate to SchemaComparison
    def validate_schema_compatibility(self, old_schema: Dict[str, Any], new_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate compatibility between old and new schemas"""
        return self.comparison.validate_schema_compatibility(old_schema, new_schema)
    
    def compare_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two schemas and return compatibility information"""
        return self.comparison.compare_schemas(schema1, schema2)
    
    # Versioning operations - delegate to SchemaVersioning
    async def save_model_schema(self, model_id: str, schema_type: str, schema_data: Dict[str, Any], version: str = "1.0") -> Dict[str, Any]:
        """Saves a model schema to the filesystem asynchronously."""
        return await self.versioning.save_model_schema(model_id, schema_type, schema_data, version)
    
    async def update_model_schema(self, schema_id: str, schema_data: Dict[str, Any], version: Optional[str] = None) -> Dict[str, Any]:
        """Updates an existing model schema asynchronously."""
        return await self.versioning.update_model_schema(schema_id, schema_data, version)
    
    async def delete_model_schema(self, schema_id: str) -> bool:
        """Deletes a model schema from the filesystem asynchronously."""
        return await self.versioning.delete_model_schema(schema_id)
    
    async def get_schema_versions(self, schema_id: str) -> List[Dict[str, Any]]:
        """Retrieves all available versions of a schema asynchronously."""
        return await self.versioning.get_schema_versions(schema_id)
    
    async def create_schema_version(
        self, 
        schema_id_base: str,
        schema_data: Dict[str, Any], 
        version: str, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new version of a schema asynchronously."""
        return await self.versioning.create_schema_version(schema_id_base, schema_data, version, description)
