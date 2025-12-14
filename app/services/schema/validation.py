"""
Schema validation and dynamic model generation
Handles validation of prediction data and generation of dynamic Pydantic models
"""

from typing import Dict, List, Optional, Tuple, Any, Type
import logging

from pydantic import BaseModel

from app.services.schema.generator import DynamicSchemaGenerator
from app.services.schema.crud import SchemaCRUD

logger = logging.getLogger(__name__)


class SchemaValidation:
    """Schema validation and dynamic model generation"""
    
    def __init__(self, crud: SchemaCRUD):
        """Initialize with CRUD service"""
        self.crud = crud
    
    async def generate_dynamic_validation_model(self, model_id: str) -> Tuple[Optional[Type[BaseModel]], Optional[Type[BaseModel]]]:
        """Generate dynamic Pydantic models for validation"""
        try:
            input_schema, output_schema = await self.crud.get_model_schemas(model_id)
            
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
            input_schema, _ = await self.crud.get_model_schemas(model_id)
            
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

