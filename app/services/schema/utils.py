"""
Schema utility functions
Handles schema generation from data, validation, merging, and conversion
"""

from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SchemaUtils:
    """Schema utility functions"""
    
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

