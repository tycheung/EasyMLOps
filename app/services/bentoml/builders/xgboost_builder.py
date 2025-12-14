"""
XGBoost BentoML service builder
Handles creation and code generation for XGBoost models
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
    BENTOML_AVAILABLE, XGBOOST_AVAILABLE, JOBLIB_AVAILABLE,
    bentoml, xgb, joblib, settings
)

logger = logging.getLogger(__name__)


class XGBoostBuilder:
    """Builder for XGBoost BentoML services"""
    
    async def create_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for XGBoost models"""
        try:
            if not XGBOOST_AVAILABLE:
                return False, "XGBoost not available", {}
            if not JOBLIB_AVAILABLE:
                import pickle

            # Load the model asynchronously
            loaded_model = None
            try:
                if JOBLIB_AVAILABLE:
                    loaded_model = await asyncio.to_thread(joblib.load, model.file_path)
                else:
                    raise ImportError("Joblib not available, trying pickle directly.")
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
                return False, f"XGBoost model {model.file_path} could not be loaded.", {}

            # Save model to BentoML model store asynchronously
            bento_model_tag_obj = await asyncio.to_thread(
                bentoml.xgboost.save_model,
                name=f"xgboost_model_{model.id.replace('-','_')}",
                model=loaded_model,
                labels={
                    "framework": "xgboost",
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
                'framework': 'xgboost',
                'endpoints': ['predict', 'predict_proba'] if model.model_type == ModelType.CLASSIFICATION else ['predict']
            }
        
            return True, "XGBoost service created successfully", service_info
        
        except Exception as e:
            logger.error(f"Error creating XGBoost service for {model.id}: {e}", exc_info=True)
            return False, str(e), {}
    
    def generate_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for XGBoost models"""
    
        service_code = f'''
import bentoml
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Any

@bentoml.service  
class XGBoostModelService:
    """Auto-generated BentoML service for XGBoost model"""

    def __init__(self):
        self.model = bentoml.xgboost.load_model("{bento_model_tag}")

    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the XGBoost model"""
        try:
            if "data" in input_data:
                data = input_data["data"]
            else:
                data = input_data
        
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
        
            dmatrix = xgb.DMatrix(df)
            predictions = self.model.predict(dmatrix)
        
            return {{
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "framework": "xgboost"
            }}
        
        except Exception as e:
            return {{"error": str(e)}}
'''
    
        return service_code

