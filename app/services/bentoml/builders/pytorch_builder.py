"""
PyTorch BentoML service builder
Handles creation and code generation for PyTorch models
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple
import logging
import aiofiles
import aiofiles.os as aios

from app.models.model import Model
from app.services.bentoml.builders.base import (
    BENTOML_AVAILABLE, PYTORCH_AVAILABLE, JOBLIB_AVAILABLE,
    bentoml, torch, joblib, settings
)

logger = logging.getLogger(__name__)


class PyTorchBuilder:
    """Builder for PyTorch BentoML services"""
    
    async def create_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for PyTorch models"""
        try:
            if not PYTORCH_AVAILABLE:
                return False, "PyTorch not available", {}
        
            model_file_path = Path(model.file_path)
            file_ext = model_file_path.suffix.lower()
            loaded_model = None

            # Load the model asynchronously
            if file_ext in ['.pt', '.pth']:
                loaded_model = await asyncio.to_thread(torch.load, model.file_path, map_location='cpu')
            elif file_ext in ['.pkl', '.joblib'] and JOBLIB_AVAILABLE:
                loaded_model = await asyncio.to_thread(joblib.load, model.file_path)
            else:
                return False, f"Unsupported PyTorch model format: {file_ext}", {}

            if loaded_model is None:
                return False, f"PyTorch model {model.file_path} could not be loaded.", {}

            # Save model to BentoML model store asynchronously
            bento_model_tag_obj = await asyncio.to_thread(
                bentoml.pytorch.save_model,
                name=f"pytorch_model_{model.id.replace('-','_')}",
                model=loaded_model,
                labels={
                    "framework": "pytorch",
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
                'framework': 'pytorch',
                'endpoints': ['predict']
            }
        
            return True, "PyTorch service created successfully", service_info
        
        except Exception as e:
            logger.error(f"Error creating PyTorch service for {model.id}: {e}", exc_info=True)
            return False, str(e), {}
    
    def generate_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for PyTorch models"""
    
        service_code = f'''
import bentoml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

@bentoml.service
class PyTorchModelService:
    """Auto-generated BentoML service for PyTorch model"""

    def __init__(self):
        self.model = bentoml.pytorch.load_model("{bento_model_tag}")

    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the PyTorch model"""
        try:
            if "data" in input_data:
                data = input_data["data"]
            else:
                data = input_data
        
            if isinstance(data, list):
                input_tensor = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, dict):
                input_tensor = torch.tensor(list(data.values()), dtype=torch.float32)
            else:
                input_tensor = torch.tensor(data, dtype=torch.float32)
        
            with torch.no_grad():
                predictions = self.model(input_tensor)
        
            return {{
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "framework": "pytorch"
            }}
        
        except Exception as e:
            return {{"error": str(e)}}
'''
    
        return service_code

