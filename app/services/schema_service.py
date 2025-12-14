"""
Schema service for dynamic model schema generation and validation
Handles user-defined input/output schemas and generates corresponding validation models

This module has been refactored into submodules:
- app.services.schema.generator: DynamicSchemaGenerator class for generating Pydantic models
- app.services.schema.service: SchemaService class for managing model schemas

This file maintains backward compatibility by re-exporting all classes, instances, and functions.
"""

# Re-export for backward compatibility
from app.services.schema.generator import DynamicSchemaGenerator
from app.services.schema.service import SchemaService

# Create global service instance for backward compatibility
schema_service = SchemaService()

# Module-level functions for backward compatibility
def generate_schema_from_data(*args, **kwargs):
    """Module-level function for generating schema from data"""
    return schema_service.generate_schema_from_data(*args, **kwargs)

def validate_input_schema(*args, **kwargs):
    """Module-level function for validating input schema"""
    return schema_service.validate_input_schema(*args, **kwargs) 

__all__ = [
    "DynamicSchemaGenerator",
    "SchemaService",
    "schema_service",
    "generate_schema_from_data",
    "validate_input_schema",
]
