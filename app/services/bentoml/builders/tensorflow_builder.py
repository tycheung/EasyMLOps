"""
TensorFlow BentoML service builder
Handles creation and code generation for TensorFlow models
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple
import logging
import aiofiles
import aiofiles.os as aios

from app.models.model import Model
from app.services.bentoml.builders.base import (
    BENTOML_AVAILABLE, TENSORFLOW_AVAILABLE, JOBLIB_AVAILABLE,
    bentoml, tf, joblib, settings
)

logger = logging.getLogger(__name__)


class TensorFlowBuilder:
    """Builder for TensorFlow BentoML services"""
    
    async def create_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for TensorFlow models"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return False, "TensorFlow not available", {}
        
            model_file_path = Path(model.file_path)
            file_ext = model_file_path.suffix.lower()
            loaded_model = None

            # Load the model asynchronously
            if file_ext == '.h5':
                loaded_model = await asyncio.to_thread(tf.keras.models.load_model, model.file_path)
            elif await aios.path.isdir(model.file_path):
                loaded_model = await asyncio.to_thread(tf.saved_model.load, model.file_path)
            elif file_ext in ['.pkl', '.joblib'] and JOBLIB_AVAILABLE:
                loaded_model = await asyncio.to_thread(joblib.load, model.file_path)
            else:
                return False, f"Unsupported TensorFlow model format or missing directory: {model.file_path}", {}

            if loaded_model is None:
                return False, f"TensorFlow model {model.file_path} could not be loaded.", {}

            # Save model to BentoML model store asynchronously
            save_func = bentoml.keras.save_model if hasattr(loaded_model, 'layers') else bentoml.tensorflow.save_model
            bento_model_tag_obj = await asyncio.to_thread(
                save_func,
                name=f"tensorflow_model_{model.id.replace('-','_')}",
                model=loaded_model,
                labels={
                    "framework": "tensorflow",
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
                'framework': 'tensorflow',
                'endpoints': ['predict']
            }
        
            return True, "TensorFlow service created successfully", service_info
        
        except Exception as e:
            logger.error(f"Error creating TensorFlow service for {model.id}: {e}", exc_info=True)
            return False, str(e), {}
    
    def generate_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for TensorFlow models"""
    
        service_code = f'''
import bentoml
import numpy as np
import pandas as pd
from typing import Dict, List, Any

@bentoml.service
class TensorFlowModelService:
    """Auto-generated BentoML service for TensorFlow model"""

    def __init__(self):
        self.model = bentoml.tensorflow.load_model("{bento_model_tag}")

    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the TensorFlow model"""
        try:
            if "data" in input_data:
                data = input_data["data"]
            else:
                data = input_data
        
            if isinstance(data, list):
                input_array = np.array(data)
            elif isinstance(data, dict):
                input_array = np.array(list(data.values()))
            else:
                input_array = np.array(data)
        
            predictions = self.model.predict(input_array)
        
            return {{
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "framework": "tensorflow"
            }}
        
        except Exception as e:
            return {{"error": str(e)}}
'''
    
        return service_code

