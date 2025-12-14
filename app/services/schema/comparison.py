"""
Schema comparison and compatibility checking
Handles schema comparison and compatibility validation
"""

from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SchemaComparison:
    """Schema comparison and compatibility checking"""
    
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

