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



"""
Dynamic Schema Generator
Generates dynamic Pydantic models from user-defined schemas
"""

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
