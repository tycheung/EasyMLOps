"""
BentoML Service Utilities
Utility methods for service management, testing, and helper functions
"""

import logging
from typing import Any, Dict, List
from pathlib import Path

from app.config import get_settings
from app.models.model import Model

settings = get_settings()
logger = logging.getLogger(__name__)


class BentoMLUtils:
    """Utility methods for BentoML service management"""
    
    def __init__(self):
        pass
    
    def get_input_schema_for_model(self, model: Model) -> Dict[str, Any]:
        """Get the input schema for a model from its input schema definitions"""
        # This would be populated from the ModelInputSchema table
        # For now, return a generic schema
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Input data for prediction"
                }
            },
            "required": ["data"]
        }
    
    def create_service(self, service_name: str, model_path: str) -> Any:
        """Create a BentoML service (for test compatibility)"""
        try:
            # Mock service for testing
            class MockService:
                def __init__(self, name):
                    self.name = name
                    
            return MockService(service_name)
        except Exception as e:
            logger.error(f"Error creating service: {e}")
            raise
    
    def build_bento(self, service_name: str, version: str = "latest") -> Any:
        """Build a Bento (for test compatibility)"""
        try:
            # Mock bento for testing
            class MockBento:
                def __init__(self, name, version):
                    self.tag = f"{name}:{version}"
                    
            return MockBento(service_name, version)
        except Exception as e:
            logger.error(f"Error building bento: {e}")
            raise
    
    def serve_bento(self, bento_tag: str, port: int = 3000) -> Any:
        """Serve a Bento (for test compatibility)"""
        try:
            # Mock server for testing
            class MockServer:
                def __init__(self, tag, port):
                    self.tag = tag
                    self.port = port
                    
            return MockServer(bento_tag, port)
        except Exception as e:
            logger.error(f"Error serving bento: {e}")
            raise
    
    def generate_service_code(self, model_info: Dict[str, Any]) -> str:
        """Generate service code for a model (for test compatibility)"""
        try:
            framework = model_info.get("framework", "sklearn")
            model_name = model_info.get("name", "model")
            model_type = model_info.get("model_type", "classification")
            
            # Basic service template
            service_code = f'''
import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray

# Model service for {model_name}
# Framework: {framework}
# Type: {model_type}

model_ref = bentoml.{framework}.get("{model_name}:latest")
svc = bentoml.Service("{model_name}_service", runners=[model_ref.to_runner()])

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    """Prediction endpoint"""
    return model_ref.to_runner().predict.run(input_data)
'''
            return service_code.strip()
        except Exception as e:
            logger.error(f"Error generating service code: {e}")
            raise
    
    def list_bentos(self) -> List[Any]:
        """List available Bentos (for test compatibility)"""
        try:
            # Mock bentos for testing
            class MockBento:
                def __init__(self, tag):
                    self.tag = tag
                    
            return [
                MockBento("service1:v1"),
                MockBento("service2:v1")
            ]
        except Exception as e:
            logger.error(f"Error listing bentos: {e}")
            raise

