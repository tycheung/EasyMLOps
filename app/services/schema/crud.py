"""
Schema CRUD operations
Handles create, read, update, delete operations for model schemas
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from sqlmodel import select

from app.database import get_session
from app.models.model import Model, ModelInputSchema, ModelOutputSchema
from app.schemas.model import (
    FieldSchema, 
    InputSchema, 
    OutputSchema, 
    ModelSchemaUpdate
)

logger = logging.getLogger(__name__)


class SchemaCRUD:
    """CRUD operations for model schemas"""
    
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

