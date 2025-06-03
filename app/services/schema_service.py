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

from sqlmodel import select

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
                # Get input schema
                input_result = await session.exec(
                    select(ModelInputSchema).where(ModelInputSchema.model_id == model_id)
                )
                input_entries = input_result.all()
                
                # Get output schema
                output_result = await session.exec(
                    select(ModelOutputSchema).where(ModelOutputSchema.model_id == model_id)
                )
                output_entries = output_result.all()
                
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
                    # Delete existing input schema
                    input_result = await session.exec(
                        select(ModelInputSchema).where(ModelInputSchema.model_id == model_id)
                    )
                    for entry in input_result.all():
                        await session.delete(entry)
                    
                    # Create new input schema
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
                    # Delete existing output schema
                    output_result = await session.exec(
                        select(ModelOutputSchema).where(ModelOutputSchema.model_id == model_id)
                    )
                    for entry in output_result.all():
                        await session.delete(entry)
                    
                    # Create new output schema
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
        """Helper method to delete existing schemas"""
        # Delete input schemas
        input_result = await session.exec(
            select(ModelInputSchema).where(ModelInputSchema.model_id == model_id)
        )
        for entry in input_result.all():
            await session.delete(entry)
        
        # Delete output schemas
        output_result = await session.exec(
            select(ModelOutputSchema).where(ModelOutputSchema.model_id == model_id)
        )
        for entry in output_result.all():
            await session.delete(entry)
    
    async def generate_dynamic_validation_model(self, model_id: str) -> Tuple[Optional[Type[BaseModel]], Optional[Type[BaseModel]]]:
        """Generate dynamic Pydantic models for input/output validation"""
        try:
            input_schema, output_schema = await self.get_model_schemas(model_id)
            
            input_model = None
            output_model = None
            
            if input_schema:
                input_model = DynamicSchemaGenerator.create_pydantic_model_from_schema(
                    f"Model{model_id}Input",
                    input_schema.fields,
                    "Input"
                )
            
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
        """Generate example data for a model based on its schema"""
        try:
            input_schema, _ = await self.get_model_schemas(model_id)
            
            if input_schema:
                return DynamicSchemaGenerator.generate_example_data(input_schema.fields)
            else:
                return {"message": "No input schema defined for this model"}
                
        except Exception as e:
            logger.error(f"Error generating example data for {model_id}: {e}")
            return {"error": str(e)}
    
    async def validate_prediction_data(self, model_id: str, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate prediction data against model schema"""
        try:
            input_model, _ = await self.generate_dynamic_validation_model(model_id)
            
            if not input_model:
                # No schema defined, allow any data
                return True, "No validation schema defined", data
            
            # Validate the data
            validated_data = input_model(**data)
            return True, "Validation successful", validated_data.dict()
            
        except Exception as e:
            logger.error(f"Validation error for model {model_id}: {e}")
            return False, str(e), None


# Global schema service instance
schema_service = SchemaService() 