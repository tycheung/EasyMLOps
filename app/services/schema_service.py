"""
Schema service for dynamic model schema generation and validation
Handles user-defined input/output schemas and generates corresponding validation models
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Type
import logging
from pydantic import BaseModel, Field, create_model, validator
from pydantic.fields import FieldInfo
import json
import asyncio

from sqlmodel import select
import aiofiles
import aiofiles.os as aios
from pathlib import Path

from app.config import get_settings
from app.database import get_session
from app.models.model import Model, ModelInputSchema, ModelOutputSchema
from app.schemas.model import (
    FieldSchema, 
    InputSchema, 
    OutputSchema, 
    DataType,
    ModelSchemaUpdate
)

settings = get_settings()
logger = logging.getLogger(__name__)


class DynamicSchemaGenerator:
    """Generates dynamic Pydantic models from user-defined schemas"""
    
    @staticmethod
    def create_pydantic_model_from_schema(
        schema_name: str, 
        fields: List[FieldSchema], 
        model_prefix: str = "Dynamic"
    ) -> Type[BaseModel]:
        """Create a Pydantic model from field schema definitions"""
        
        pydantic_fields = {}
        
        for field in fields:
            field_type = DynamicSchemaGenerator._get_python_type(field.data_type)
            field_info = DynamicSchemaGenerator._create_field_info(field)
            
            if field.required:
                pydantic_fields[field.name] = (field_type, field_info)
            else:
                pydantic_fields[field.name] = (Optional[field_type], field_info)
        
        # Create the dynamic model
        model_name = f"{model_prefix}{schema_name.replace(' ', '').replace('-', '').replace('_', '')}"
        
        dynamic_model = create_model(
            model_name,
            **pydantic_fields
        )
        
        return dynamic_model
    
    @staticmethod
    def _get_python_type(data_type: DataType) -> Type:
        """Convert DataType enum to Python type"""
        type_mapping = {
            DataType.INTEGER: int,
            DataType.FLOAT: float,
            DataType.STRING: str,
            DataType.BOOLEAN: bool,
            DataType.ARRAY: List[Any],
            DataType.OBJECT: Dict[str, Any],
            DataType.DATE: str,  # Will be validated as date string
            DataType.DATETIME: str,  # Will be validated as datetime string
        }
        return type_mapping.get(data_type, str)
    
    @staticmethod
    def _create_field_info(field: FieldSchema) -> FieldInfo:
        """Create Pydantic FieldInfo from FieldSchema"""
        field_kwargs = {
            "description": field.description or f"{field.name} field"
        }
        
        # Add default value if not required
        if not field.required and field.default_value is not None:
            field_kwargs["default"] = field.default_value
        elif not field.required:
            field_kwargs["default"] = None
        
        # Add validation constraints
        if field.min_value is not None:
            field_kwargs["ge"] = field.min_value
        
        if field.max_value is not None:
            field_kwargs["le"] = field.max_value
        
        if field.min_length is not None:
            field_kwargs["min_length"] = field.min_length
        
        if field.max_length is not None:
            field_kwargs["max_length"] = field.max_length
        
        if field.pattern is not None:
            field_kwargs["regex"] = field.pattern
        
        return Field(**field_kwargs)
    
    @staticmethod
    def generate_example_data(fields: List[FieldSchema]) -> Dict[str, Any]:
        """Generate example data based on field schemas"""
        example = {}
        
        for field in fields:
            if field.data_type == DataType.INTEGER:
                example[field.name] = field.min_value or 1
            elif field.data_type == DataType.FLOAT:
                example[field.name] = field.min_value or 1.0
            elif field.data_type == DataType.STRING:
                example[field.name] = f"example_{field.name}"
            elif field.data_type == DataType.BOOLEAN:
                example[field.name] = True
            elif field.data_type == DataType.ARRAY:
                example[field.name] = [1, 2, 3]
            elif field.data_type == DataType.OBJECT:
                example[field.name] = {"key": "value"}
            elif field.data_type == DataType.DATE:
                example[field.name] = "2024-01-01"
            elif field.data_type == DataType.DATETIME:
                example[field.name] = "2024-01-01T12:00:00Z"
            else:
                example[field.name] = "example"
        
        return example


class SchemaService:
    """Service for managing model input/output schemas"""
    
    async def create_model_schemas(
        self, 
        model_id: str, 
        input_schema: InputSchema, 
        output_schema: OutputSchema
    ) -> Tuple[bool, str]:
        """Create input and output schemas for a model"""
        try:
            async with get_session() as session:
                # Check if model exists
                model = await session.get(Model, model_id)
                if not model:
                    return False, f"Model {model_id} not found"
                
                # Delete existing schemas
                await self._delete_existing_schemas(session, model_id)
                
                # Create input schema entries
                for field in input_schema.fields:
                    input_schema_entry = ModelInputSchema(
                        id=str(uuid.uuid4()),
                        model_id=model_id,
                        field_name=field.name,
                        data_type=field.data_type,
                        required=field.required,
                        description=field.description,
                        default_value=field.default_value,
                        min_value=field.min_value,
                        max_value=field.max_value,
                        min_length=field.min_length,
                        max_length=field.max_length,
                        allowed_values=field.allowed_values,
                        pattern=field.pattern,
                        created_at=datetime.utcnow()
                    )
                    session.add(input_schema_entry)
                
                # Create output schema entries
                for field in output_schema.fields:
                    output_schema_entry = ModelOutputSchema(
                        id=str(uuid.uuid4()),
                        model_id=model_id,
                        field_name=field.name,
                        data_type=field.data_type,
                        description=field.description,
                        created_at=datetime.utcnow()
                    )
                    session.add(output_schema_entry)
                
                # Update model to indicate schemas are defined
                model.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(f"Successfully created schemas for model {model_id}")
                return True, "Schemas created successfully"
                
        except Exception as e:
            logger.error(f"Error creating schemas for model {model_id}: {e}")
            return False, str(e)
    
    async def get_model_schemas(self, model_id: str) -> Tuple[Optional[InputSchema], Optional[OutputSchema]]:
        """Get input and output schemas for a model"""
        try:
            async with get_session() as session:
                # Get input schema entries
                input_result = await session.execute(
                    select(ModelInputSchema).where(ModelInputSchema.model_id == model_id)
                )
                input_entries = input_result.scalars().all()
                
                # Get output schema entries
                output_result = await session.execute(
                    select(ModelOutputSchema).where(ModelOutputSchema.model_id == model_id)
                )
                output_entries = output_result.scalars().all()
                
                # Convert to schema objects
                input_schema = None
                if input_entries:
                    input_fields = [
                        FieldSchema(
                            name=entry.field_name,
                            data_type=entry.data_type,
                            required=entry.required,
                            description=entry.description,
                            default_value=entry.default_value,
                            min_value=entry.min_value,
                            max_value=entry.max_value,
                            min_length=entry.min_length,
                            max_length=entry.max_length,
                            allowed_values=entry.allowed_values,
                            pattern=entry.pattern
                        )
                        for entry in input_entries
                    ]
                    input_schema = InputSchema(fields=input_fields)
                
                output_schema = None
                if output_entries:
                    output_fields = [
                        FieldSchema(
                            name=entry.field_name,
                            data_type=entry.data_type,
                            required=True,  # Output fields are always present
                            description=entry.description
                        )
                        for entry in output_entries
                    ]
                    output_schema = OutputSchema(fields=output_fields)
                
                return input_schema, output_schema
                
        except Exception as e:
            logger.error(f"Error getting schemas for model {model_id}: {e}")
            return None, None
    
    async def update_model_schemas(
        self, 
        model_id: str, 
        schema_update: ModelSchemaUpdate
    ) -> Tuple[bool, str]:
        """Update model schemas"""
        try:
            async with get_session() as session:
                # Check if model exists
                model = await session.get(Model, model_id)
                if not model:
                    return False, f"Model {model_id} not found"
                
                # Update input schema if provided
                if schema_update.input_schema:
                    # Delete existing input schemas
                    input_result = await session.execute(
                        select(ModelInputSchema).where(ModelInputSchema.model_id == model_id)
                    )
                    for entry in input_result.scalars().all():
                        await session.delete(entry)
                    
                    # Create new input schema entries
                    for field in schema_update.input_schema.fields:
                        input_schema_entry = ModelInputSchema(
                            id=str(uuid.uuid4()),
                            model_id=model_id,
                            field_name=field.name,
                            data_type=field.data_type,
                            required=field.required,
                            description=field.description,
                            default_value=field.default_value,
                            min_value=field.min_value,
                            max_value=field.max_value,
                            min_length=field.min_length,
                            max_length=field.max_length,
                            allowed_values=field.allowed_values,
                            pattern=field.pattern,
                            created_at=datetime.utcnow()
                        )
                        session.add(input_schema_entry)
                
                # Update output schema if provided
                if schema_update.output_schema:
                    # Delete existing output schemas
                    output_result = await session.execute(
                        select(ModelOutputSchema).where(ModelOutputSchema.model_id == model_id)
                    )
                    for entry in output_result.scalars().all():
                        await session.delete(entry)
                    
                    # Create new output schema entries
                    for field in schema_update.output_schema.fields:
                        output_schema_entry = ModelOutputSchema(
                            id=str(uuid.uuid4()),
                            model_id=model_id,
                            field_name=field.name,
                            data_type=field.data_type,
                            description=field.description,
                            created_at=datetime.utcnow()
                        )
                        session.add(output_schema_entry)
                
                # Update model timestamp
                model.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(f"Successfully updated schemas for model {model_id}")
                return True, "Schemas updated successfully"
                
        except Exception as e:
            logger.error(f"Error updating schemas for model {model_id}: {e}")
            return False, str(e)
    
    async def delete_model_schemas(self, model_id: str) -> Tuple[bool, str]:
        """Delete all schemas for a model"""
        try:
            async with get_session() as session:
                await self._delete_existing_schemas(session, model_id)
                await session.commit()
                
                logger.info(f"Successfully deleted schemas for model {model_id}")
                return True, "Schemas deleted successfully"
                
        except Exception as e:
            logger.error(f"Error deleting schemas for model {model_id}: {e}")
            return False, str(e)
    
    async def _delete_existing_schemas(self, session, model_id: str):
        """Delete existing schemas for a model"""
        # Delete input schemas
        input_result = await session.execute(
            select(ModelInputSchema).where(ModelInputSchema.model_id == model_id)
        )
        for entry in input_result.scalars().all():
            await session.delete(entry)
        
        # Delete output schemas
        output_result = await session.execute(
            select(ModelOutputSchema).where(ModelOutputSchema.model_id == model_id)
        )
        for entry in output_result.scalars().all():
            await session.delete(entry)
    
    async def generate_dynamic_validation_model(self, model_id: str) -> Tuple[Optional[Type[BaseModel]], Optional[Type[BaseModel]]]:
        """Generate dynamic Pydantic models for validation"""
        try:
            input_schema, output_schema = await self.get_model_schemas(model_id)
            
            input_model = None
            if input_schema:
                input_model = DynamicSchemaGenerator.create_pydantic_model_from_schema(
                    f"Model{model_id}Input", 
                    input_schema.fields,
                    "Input"
                )
            
            output_model = None
            if output_schema:
                output_model = DynamicSchemaGenerator.create_pydantic_model_from_schema(
                    f"Model{model_id}Output", 
                    output_schema.fields,
                    "Output"
                )
            
            return input_model, output_model
            
        except Exception as e:
            logger.error(f"Error generating dynamic models for {model_id}: {e}")
            return None, None
    
    async def get_model_example_data(self, model_id: str) -> Dict[str, Any]:
        """Get example input data for a model"""
        try:
            input_schema, _ = await self.get_model_schemas(model_id)
            
            if input_schema:
                return DynamicSchemaGenerator.generate_example_data(input_schema.fields)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting example data for model {model_id}: {e}")
            return {}
    
    async def validate_prediction_data(self, model_id: str, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate data against model schema"""
        try:
            input_model, _ = await self.generate_dynamic_validation_model(model_id)
            
            if not input_model:
                return True, "No schema defined, validation skipped", data
            
            # Validate using the dynamic model
            validated_data = input_model(**data)
            return True, "Validation successful", validated_data.dict()
            
        except Exception as e:
            logger.error(f"Validation error for model {model_id}: {e}")
            return False, str(e), None
    
    def generate_schema_from_data(self, sample_data: Any, schema_type: str = "input", include_target: bool = False) -> Dict[str, Any]:
        """Generate JSON schema from sample data"""
        if not sample_data:
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        # Handle list of samples - use the first item
        if isinstance(sample_data, list):
            if not sample_data:
                return {
                    "type": "object", 
                    "properties": {},
                    "required": []
                }
            sample_data = sample_data[0]
        
        # Ensure we have a dictionary to work with
        if not isinstance(sample_data, dict):
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for key, value in sample_data.items():
            # Skip target field if not included
            if not include_target and key in ["price", "target", "label", "y"]:
                continue
                
            if isinstance(value, str):
                schema["properties"][key] = {"type": "string"}
            elif isinstance(value, int):
                schema["properties"][key] = {"type": "integer"}
            elif isinstance(value, float):
                schema["properties"][key] = {"type": "number"}
            elif isinstance(value, bool):
                schema["properties"][key] = {"type": "boolean"}
            elif isinstance(value, list):
                schema["properties"][key] = {"type": "array"}
            elif isinstance(value, dict):
                schema["properties"][key] = {"type": "object"}
            else:
                schema["properties"][key] = {"type": "string"}
            
            # Assume all fields are required for generated schema
            schema["required"].append(key)
        
        return schema
    
    def validate_input_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against a schema dictionary (parameter order swapped to match tests)"""
        errors = []
        
        # Basic validation - this is a simplified implementation
        # In a real scenario, you'd use a proper JSON schema validator
        try:
            required_fields = schema.get('required', [])
            properties = schema.get('properties', {})
            
            # Check required fields
            for field in required_fields:
                if field not in data:
                    errors.append(f"Required field '{field}' is missing")
            
            # Check field types
            for field, value in data.items():
                if field in properties:
                    expected_type = properties[field].get('type')
                    if expected_type:
                        if not self._validate_type(value, expected_type):
                            errors.append(f"Field '{field}' has invalid type")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")
            return False, errors
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    def merge_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two schemas"""
        merged = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Merge properties
        if "properties" in schema1:
            merged["properties"].update(schema1["properties"])
        if "properties" in schema2:
            merged["properties"].update(schema2["properties"])
        
        # Merge required fields
        if "required" in schema1:
            merged["required"].extend(schema1["required"])
        if "required" in schema2:
            merged["required"].extend(schema2["required"])
        
        # Remove duplicates from required
        merged["required"] = list(set(merged["required"]))
        
        return merged
    
    def convert_to_openapi_schema(self, json_schema: Dict[str, Any], include_examples: bool = False) -> Dict[str, Any]:
        """Convert JSON schema to OpenAPI format"""
        # For basic conversion, OpenAPI schema is similar to JSON schema
        # but with some differences in keywords and structure
        openapi_schema = json_schema.copy()
        
        # OpenAPI uses 'example' instead of 'default' in some cases
        if "properties" in openapi_schema:
            for prop_name, prop_schema in openapi_schema["properties"].items():
                if "default" in prop_schema:
                    prop_schema["example"] = prop_schema["default"]
                    
                # Add examples if requested
                if include_examples and "example" not in prop_schema:
                    if prop_schema.get("type") == "string":
                        prop_schema["example"] = f"example_{prop_name}"
                    elif prop_schema.get("type") == "integer":
                        prop_schema["example"] = 1
                    elif prop_schema.get("type") == "number":
                        prop_schema["example"] = 1.0
                    elif prop_schema.get("type") == "boolean":
                        prop_schema["example"] = True
        
        return openapi_schema
    
    def convert_to_json_schema(self, openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAPI schema to JSON schema format"""
        # Reverse conversion from OpenAPI to JSON schema
        json_schema = openapi_schema.copy()
        
        # Convert 'example' back to 'default' if needed
        if "properties" in json_schema:
            for prop_name, prop_schema in json_schema["properties"].items():
                if "example" in prop_schema and "default" not in prop_schema:
                    prop_schema["default"] = prop_schema["example"]
                    del prop_schema["example"]
        
        return json_schema
    
    def validate_schema_compatibility(self, old_schema: Dict[str, Any], new_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate compatibility between old and new schemas"""
        issues = []
        
        old_properties = old_schema.get("properties", {})
        new_properties = new_schema.get("properties", {})
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))
        
        # Check for removed fields
        removed_fields = set(old_properties.keys()) - set(new_properties.keys())
        if removed_fields:
            issues.append(f"Removed fields: {', '.join(removed_fields)}")
        
        # Check for newly required fields
        newly_required = new_required - old_required
        if newly_required:
            issues.append(f"Newly required fields: {', '.join(newly_required)}")
        
        # Check for type changes
        for field in old_properties:
            if field in new_properties:
                old_type = old_properties[field].get("type")
                new_type = new_properties[field].get("type")
                if old_type != new_type:
                    issues.append(f"Type changed for field '{field}': {old_type} -> {new_type}")
        
        return len(issues) == 0, issues
    
    def compare_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two schemas and return compatibility information"""
        try:
            properties1 = schema1.get("properties", {})
            properties2 = schema2.get("properties", {})
            required1 = set(schema1.get("required", []))
            required2 = set(schema2.get("required", []))
            
            differences = []
            breaking_changes = []
            
            # Check for removed fields
            removed_fields = set(properties1.keys()) - set(properties2.keys())
            for field in removed_fields:
                differences.append({
                    "type": "removed_field",
                    "field": field,
                    "severity": "major" if field in required1 else "minor"
                })
                if field in required1:
                    breaking_changes.append({
                        "type": "required_field_removed",
                        "field": field,
                        "description": f"Required field '{field}' was removed"
                    })
            
            # Check for added fields
            added_fields = set(properties2.keys()) - set(properties1.keys())
            for field in added_fields:
                differences.append({
                    "type": "added_field",
                    "field": field,
                    "severity": "major" if field in required2 else "minor"
                })
                if field in required2:
                    breaking_changes.append({
                        "type": "new_required_field",
                        "field": field,
                        "description": f"New required field '{field}' was added"
                    })
            
            # Check for type changes
            common_fields = set(properties1.keys()) & set(properties2.keys())
            for field in common_fields:
                type1 = properties1[field].get("type")
                type2 = properties2[field].get("type")
                if type1 != type2:
                    differences.append({
                        "type": "type_change",
                        "field": field,
                        "old_type": type1,
                        "new_type": type2,
                        "severity": "major" if field in required1 or field in required2 else "minor"
                    })
                    if field in required1 or field in required2:
                        breaking_changes.append({
                            "type": "required_field_type_change",
                            "field": field,
                            "description": f"Type changed for field '{field}': {type1} -> {type2}"
                        })
            
            # Calculate compatibility score
            total_fields = len(set(properties1.keys()) | set(properties2.keys()))
            major_issues = len([d for d in differences if d.get("severity") == "major"])
            
            if total_fields == 0:
                compatibility_score = 1.0
            else:
                compatibility_score = max(0.0, 1.0 - (major_issues / total_fields))
            
            # Determine if schemas are compatible
            is_compatible = len(breaking_changes) == 0
            
            return {
                "compatible": is_compatible,
                "compatibility_score": compatibility_score,
                "differences": differences,
                "breaking_changes": breaking_changes,
                "summary": {
                    "total_differences": len(differences),
                    "breaking_changes_count": len(breaking_changes),
                    "added_fields": len(added_fields),
                    "removed_fields": len(removed_fields),
                    "type_changes": len([d for d in differences if d["type"] == "type_change"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing schemas: {e}")
            return {
                "compatible": False,
                "compatibility_score": 0.0,
                "differences": [],
                "breaking_changes": [],
                "error": str(e)
            }

    async def save_model_schema(self, model_id: str, schema_type: str, schema_data: Dict[str, Any], version: str = "1.0") -> Dict[str, Any]:
        """Saves a model schema to the filesystem asynchronously."""
        try:
            # Ensure model directory exists (can be sync as it's a pre-check)
            model_base_dir = Path(settings.MODELS_DIR) / model_id
            if not await aios.path.exists(model_base_dir):
                 # If model dir itself doesn't exist, this might be an issue depending on app logic
                 # For now, we'll assume model directory is created elsewhere or this implies an error.
                 # If we need to create it: await aios.makedirs(model_base_dir, exist_ok=True)
                 logger.warning(f"Base directory for model {model_id} not found at {model_base_dir}")
                 # Depending on requirements, could raise error or proceed to create

            schema_dir = model_base_dir / "schemas" / schema_type / version
            await aios.makedirs(schema_dir, exist_ok=True)
            
            file_path = schema_dir / "schema.json"
            
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(schema_data, indent=2))
            
            logger.info(f"Saved schema for model {model_id}, type {schema_type}, version {version} to {file_path}")
            return {
                "message": "Schema saved successfully",
                "model_id": model_id,
                "schema_type": schema_type,
                "version": version,
                "path": str(file_path)
            }
        except Exception as e:
            logger.error(f"Error saving schema for model {model_id}: {e}")
            # Consider re-raising or returning a more specific error structure
            raise # Or return a dict indicating failure

    async def update_model_schema(self, schema_id: str, schema_data: Dict[str, Any], version: Optional[str] = None) -> Dict[str, Any]:
        """Updates an existing model schema asynchronously. schema_id is expected to be model_id/schema_type."""
        try:
            parts = schema_id.split('/')
            if len(parts) < 2:
                # This simple splitting might not be robust enough for a real schema_id.
                # A more structured schema_id or separate params for model_id/type might be better.
                raise ValueError("Invalid schema_id format. Expected model_id/schema_type[/version]")
            
            model_id = parts[0]
            schema_type = parts[1]
            # Use provided version or determine the latest/default if not given
            # For this example, if version is not in schema_id and not passed, it will update a 'default' or 'latest' schema.
            # A real implementation would need robust version handling.
            target_version = version or (parts[2] if len(parts) > 2 else "1.0") # Default to 1.0 or use from ID

            schema_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type / target_version
            file_path = schema_dir / "schema.json"

            if not await aios.path.exists(file_path):
                logger.error(f"Schema file not found at {file_path} for update.")
                # Option 1: Create if not exists (like a save/upsert)
                # await aios.makedirs(schema_dir, exist_ok=True)
                # logger.info(f"Schema file not found, creating at {file_path}")
                # Option 2: Raise error
                raise FileNotFoundError(f"Schema file not found at {file_path}. Cannot update.")

            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(schema_data, indent=2))
            
            logger.info(f"Updated schema for {schema_id}, version {target_version} at {file_path}")
            return {
                "message": "Schema updated successfully",
                "schema_id": schema_id, # Original ID passed
                "model_id": model_id,
                "schema_type": schema_type,
                "version": target_version, # The version that was updated
                "path": str(file_path)
            }
        except FileNotFoundError as fnf_e:
            logger.error(f"Error updating schema {schema_id}: {fnf_e}")
            raise # Re-raise specific error to be caught by route
        except Exception as e:
            logger.error(f"Error updating schema {schema_id}: {e}")
            raise # Or return a dict indicating failure

    async def delete_model_schema(self, schema_id: str) -> bool:
        """Deletes a model schema (or a specific version) from the filesystem asynchronously.
           schema_id can be model_id/schema_type to delete all versions for that type,
           or model_id/schema_type/version to delete a specific version.
        """
        try:
            parts = schema_id.split('/')
            if len(parts) < 2:
                raise ValueError("Invalid schema_id format. Expected model_id/schema_type[/version]")

            model_id = parts[0]
            schema_type = parts[1]
            version = parts[2] if len(parts) > 2 else None

            model_schema_base_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type

            if not await aios.path.exists(model_schema_base_dir):
                logger.warning(f"Schema directory not found for {schema_id} at {model_schema_base_dir}. Nothing to delete.")
                return False # Or True if not finding it means it's already effectively deleted

            if version:
                # Delete specific version
                schema_file_path = model_schema_base_dir / version / "schema.json"
                version_dir_path = model_schema_base_dir / version
                if await aios.path.exists(schema_file_path):
                    await aios.remove(schema_file_path)
                    # Try to remove version directory if empty, then schema_type dir if empty, then schemas dir etc.
                    try:
                        await aios.rmdir(version_dir_path) # Fails if not empty
                        logger.info(f"Removed empty version directory: {version_dir_path}")
                    except OSError: # Directory not empty or other issue
                        pass # Ignore, file was deleted which is primary goal
                    logger.info(f"Deleted schema version {schema_id} from {schema_file_path}")
                    return True
                else:
                    logger.warning(f"Schema version file not found for {schema_id} at {schema_file_path}. Nothing to delete.")
                    return False
            else:
                # Delete all versions (the entire schema_type directory)
                import shutil # shutil.rmtree is synchronous
                # For async removal of a directory tree, we might need a helper or iterate
                # For simplicity here, if aiofiles.os doesn't have rmtree, this is complex
                # For now, let's use a blocking call within to_thread for directory removal if no simple async alternative
                # OR, implement recursive async delete
                # This is a placeholder for robust async directory tree removal:
                # await self._recursive_delete(model_schema_base_dir) 
                # If using asyncio.to_thread for shutil.rmtree:
                if await aios.path.exists(model_schema_base_dir):
                    await asyncio.to_thread(shutil.rmtree, model_schema_base_dir)
                    logger.info(f"Deleted all schema versions for {model_id}/{schema_type} from {model_schema_base_dir}")
                    # Potentially try to remove parent dirs if they become empty
                    return True
                else:
                    logger.warning(f"Schema directory not found for {model_id}/{schema_type}. Nothing to delete.")
                    return False

        except ValueError as ve:
            logger.error(f"Invalid schema_id for delete: {ve}")
            return False # Or raise to be caught by route
        except Exception as e:
            logger.error(f"Error deleting schema {schema_id}: {e}")
            return False # Or raise

    async def get_schema_versions(self, schema_id: str) -> List[Dict[str, Any]]:
        """Retrieves all available versions of a schema asynchronously.
           schema_id is expected to be model_id/schema_type.
        """
        versions_info = []
        try:
            parts = schema_id.split('/')
            if len(parts) != 2:
                raise ValueError("Invalid schema_id format. Expected model_id/schema_type")

            model_id = parts[0]
            schema_type = parts[1]

            schema_type_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type

            if not await aios.path.exists(schema_type_dir) or not await aios.path.isdir(schema_type_dir):
                logger.info(f"Schema type directory not found for {schema_id} at {schema_type_dir}")
                return [] # No versions if directory doesn't exist

            # Iterate over version directories
            version_dirs = await aios.listdir(schema_type_dir)
            for version_name in version_dirs:
                version_path = schema_type_dir / version_name
                if await aios.path.isdir(version_path):
                    schema_file_path = version_path / "schema.json"
                    if await aios.path.exists(schema_file_path):
                        try:
                            async with aiofiles.open(schema_file_path, "r") as f:
                                content = await f.read()
                                schema_data = json.loads(content) # json.loads is sync, but on small string okay
                            
                            # Attempt to get file stats for modification time
                            try:
                                stat_result = await aios.stat(schema_file_path)
                                last_modified = datetime.fromtimestamp(stat_result.st_mtime).isoformat()
                            except Exception as stat_err:
                                logger.warning(f"Could not get stat for {schema_file_path}: {stat_err}")
                                last_modified = datetime.utcnow().isoformat() # Fallback

                            versions_info.append({
                                "version": version_name,
                                "schema_id": f"{schema_id}/{version_name}",
                                "model_id": model_id,
                                "schema_type": schema_type,
                                "description": schema_data.get("description", "N/A"), # Assuming schema has description
                                "retrieved_at": datetime.utcnow().isoformat(),
                                "last_modified": last_modified,
                                # "schema_data": schema_data # Optionally include full schema data
                            })
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Error decoding JSON for schema {schema_file_path}: {json_err}")
                        except Exception as e:
                            logger.error(f"Error reading schema version {version_path}: {e}")
            
            versions_info.sort(key=lambda x: x.get("version", "0.0.0"), reverse=True) # Basic sort
            return versions_info
        except ValueError as ve:
            logger.error(f"Invalid schema_id for get_schema_versions: {ve}")
            raise # Propagate to route for 400 error
        except Exception as e:
            logger.error(f"Error retrieving schema versions for {schema_id}: {e}")
            # In a real app, might return empty list or raise specific error
            raise # Propagate to route for 500 error

    async def create_schema_version(
        self, 
        schema_id_base: str, # Expected model_id/schema_type
        schema_data: Dict[str, Any], 
        version: str, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new version of a schema asynchronously."""
        try:
            parts = schema_id_base.split('/')
            if len(parts) != 2:
                raise ValueError("Invalid schema_id_base format. Expected model_id/schema_type")

            model_id = parts[0]
            schema_type = parts[1]

            # Add description to schema_data if provided (common practice)
            if description and "description" not in schema_data:
                schema_data["description"] = description
            elif description:
                # If schema_data already has a description, prefer the one from schema_data
                # or decide on a merging strategy. For now, let's assume schema_data is king.
                pass 

            schema_version_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type / version
            
            if await aios.path.exists(schema_version_dir):
                # Depending on policy, either raise error or allow overwrite (or require a force flag)
                raise FileExistsError(f"Schema version {version} already exists for {schema_id_base}")

            await aios.makedirs(schema_version_dir, exist_ok=True) # exist_ok handles race conditions better
            
            file_path = schema_version_dir / "schema.json"
            
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(schema_data, indent=2))
            
            logger.info(f"Created schema version {version} for {schema_id_base} at {file_path}")
            return {
                "message": "Schema version created successfully",
                "schema_id": f"{schema_id_base}/{version}",
                "model_id": model_id,
                "schema_type": schema_type,
                "version": version,
                "description": description or schema_data.get("description"),
                "path": str(file_path),
                "created_at": datetime.utcnow().isoformat()
            }
        except FileExistsError as fee:
            logger.error(f"Error creating schema version for {schema_id_base}: {fee}")
            raise # Propagate to route for 409 Conflict or similar
        except ValueError as ve:
            logger.error(f"Invalid input for create_schema_version: {ve}")
            raise # Propagate to route for 400 error
        except Exception as e:
            logger.error(f"Error creating schema version for {schema_id_base}: {e}")
            raise # Propagate to route for 500 error


# Global service instance
schema_service = SchemaService()

# Module-level functions for backward compatibility
def generate_schema_from_data(*args, **kwargs):
    """Module-level function for generating schema from data"""
    return schema_service.generate_schema_from_data(*args, **kwargs)

def validate_input_schema(*args, **kwargs):
    """Module-level function for validating input schema"""
    return schema_service.validate_input_schema(*args, **kwargs) 