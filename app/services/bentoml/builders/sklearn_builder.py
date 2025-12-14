"""
Sklearn BentoML service builder
Handles creation and code generation for sklearn models
"""

import pickle
import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple
import logging
import aiofiles
import aiofiles.os as aios

from app.models.model import Model
from app.schemas.model import ModelType
from app.services.bentoml.builders.base import (
    BENTOML_AVAILABLE, SKLEARN_AVAILABLE, JOBLIB_AVAILABLE,
    bentoml, settings
)

logger = logging.getLogger(__name__)


class SklearnBuilder:
    """Builder for sklearn BentoML services"""
    
    async def create_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for sklearn models"""
        try:
            if not BENTOML_AVAILABLE:
                return False, "BentoML not available", {}
            if not SKLEARN_AVAILABLE:
                return False, "sklearn not available", {}
            if not JOBLIB_AVAILABLE:
                return False, "joblib not available for loading sklearn model", {}

            # Load the model asynchronously
            loaded_model = None
            try:
                loaded_model = await asyncio.to_thread(joblib.load, model.file_path)
            except Exception as e_joblib:
                logger.warning(f"Failed to load model {model.file_path} with joblib: {e_joblib}, trying pickle.")
                try:
                    async with aiofiles.open(model.file_path, 'rb') as f:
                        model_bytes = await f.read()
                    loaded_model = await asyncio.to_thread(pickle.loads, model_bytes)
                except Exception as e_pickle:
                    logger.error(f"Failed to load model {model.file_path} with pickle after joblib failed: {e_pickle}")
                    return False, f"Failed to load model file: {e_pickle}", {}
        
            if loaded_model is None:
                 return False, f"Model {model.file_path} could not be loaded.", {}

            # Save model to BentoML model store asynchronously
            bento_model_tag_obj = await asyncio.to_thread(
                bentoml.sklearn.save_model,
                name=f"sklearn_model_{model.id.replace('-','_')}",
                model=loaded_model,
                labels={
                    "framework": "sklearn",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            bento_model_tag = str(bento_model_tag_obj)
        
            # Create service definition
            service_code = self.generate_service_code(model, bento_model_tag, config or {})
        
            # Write service file asynchronously
            service_path_obj = Path(settings.BENTOS_DIR) / f"{service_name}.py"
            await aios.makedirs(service_path_obj.parent, exist_ok=True)
        
            async with aiofiles.open(service_path_obj, 'w') as f:
                await f.write(service_code)
        
            service_info = {
                'service_name': service_name,
                'service_path': str(service_path_obj),
                'bento_model_tag': bento_model_tag,
                'framework': 'sklearn',
                'endpoints': ['predict', 'predict_proba'] if model.model_type == ModelType.CLASSIFICATION else ['predict']
            }
        
            return True, "sklearn service created successfully", service_info
        
        except Exception as e:
            logger.error(f"Error creating sklearn service for {model.id}: {e}", exc_info=True)
            return False, str(e), {}
    
    def generate_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for sklearn models"""
    
        service_code = f'''
import bentoml
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pydantic import BaseModel, Field, ValidationError

# Load the model using new BentoML 1.4+ API
model_ref = bentoml.sklearn.get("{bento_model_tag}")

@bentoml.service
class SklearnModelService:
    """Auto-generated BentoML service for sklearn model"""

    def __init__(self):
        self.model = bentoml.sklearn.load_model("{bento_model_tag}")

    async def validate_input_schema(self, input_data: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Validate input data against model schema"""
        try:
            if "data" not in input_data:
                return True, "Direct field input accepted", input_data
            else:
                return True, "Traditional format accepted", input_data["data"]
            
        except Exception as e:
            return False, f"Validation error: {{str(e)}}", {{}}

    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the sklearn model with schema validation"""
        try:
            is_valid, validation_message, validated_data = await self.validate_input_schema(input_data)
        
            if not is_valid:
                return {{
                    "error": f"Input validation failed: {{validation_message}}",
                    "model_id": "{model.id}"
                }}
        
            if isinstance(validated_data, dict) and "data" not in validated_data:
                df = pd.DataFrame([validated_data])
            elif isinstance(validated_data, list):
                if len(validated_data) > 0 and isinstance(validated_data[0], dict):
                    df = pd.DataFrame(validated_data)
                else:
                    df = pd.DataFrame([validated_data])
            elif isinstance(validated_data, dict):
                if "data" in validated_data:
                    data = validated_data["data"]
                    if isinstance(data, list):
                        if len(data) > 0 and isinstance(data[0], dict):
                            df = pd.DataFrame(data)
                        else:
                            df = pd.DataFrame([data])
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                    else:
                        return {{"error": "Invalid data format"}}
                else:
                    df = pd.DataFrame([validated_data])
            else:
                return {{"error": "Invalid input format"}}
        
            predictions = self.model.predict(df)
        
            return {{
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "model_type": "{model.model_type.value if model.model_type else 'unknown'}",
                "validation": {{
                    "performed": True,
                    "message": validation_message
                }}
            }}
        
        except Exception as e:
            return {{"error": str(e)}}

    @bentoml.api
    def predict_batch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make batch predictions with schema validation"""
        try:
            is_valid, validation_message, validated_data = await self.validate_input_schema(input_data)
        
            if not is_valid:
                return {{
                    "error": f"Input validation failed: {{validation_message}}",
                    "model_id": "{model.id}"
                }}
        
            if isinstance(validated_data, list):
                if len(validated_data) > 0 and isinstance(validated_data[0], dict):
                    df = pd.DataFrame(validated_data)
                else:
                    df = pd.DataFrame([validated_data])
            elif isinstance(validated_data, dict) and "data" in validated_data:
                data = validated_data["data"]
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame([data])
            else:
                return {{"error": "Invalid batch input format"}}
        
            predictions = self.model.predict(df)
        
            return {{
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "count": len(predictions)
            }}
        
        except Exception as e:
            return {{"error": str(e)}}

    @bentoml.api
    def predict_proba(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make probability predictions (for classification models)"""
        try:
            if not hasattr(self.model, 'predict_proba'):
                return {{"error": "Model does not support predict_proba"}}
        
            is_valid, validation_message, validated_data = await self.validate_input_schema(input_data)
        
            if not is_valid:
                return {{
                    "error": f"Input validation failed: {{validation_message}}",
                    "model_id": "{model.id}"
                }}
        
            if isinstance(validated_data, dict) and "data" not in validated_data:
                df = pd.DataFrame([validated_data])
            elif isinstance(validated_data, list):
                if len(validated_data) > 0 and isinstance(validated_data[0], dict):
                    df = pd.DataFrame(validated_data)
                else:
                    df = pd.DataFrame([validated_data])
            else:
                df = pd.DataFrame([validated_data])
        
            probabilities = self.model.predict_proba(df)
        
            return {{
                "probabilities": probabilities.tolist() if hasattr(probabilities, 'tolist') else list(probabilities),
                "model_id": "{model.id}",
                "model_name": "{model.name}"
            }}
        
        except Exception as e:
            return {{"error": str(e)}}
'''
    
        return service_code

