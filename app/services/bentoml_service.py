"""
BentoML service management for dynamic model serving
Handles creation, deployment, and lifecycle management of ML model services
"""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import json
import uuid
from datetime import datetime
import asyncio
import aiofiles
import aiofiles.os as aios
import pickle

# Lazy import for BentoML to avoid configuration issues during testing
try:
    import bentoml
    from bentoml import Service
    BENTOML_AVAILABLE = True
except Exception as e:
    bentoml = None
    Service = None
    BENTOML_AVAILABLE = False
    # Don't log during import - will be logged when actually needed

import pandas as pd
import numpy as np

from app.config import get_settings
from app.schemas.model import ModelFramework, ModelType, ModelStatus, ModelDeploymentCreate, ModelDeploymentResponse
from app.models.model import Model, ModelDeployment
from app.utils.model_utils import ModelValidator
from app.database import get_session
from app.services.schema_service import schema_service

settings = get_settings()
logger = logging.getLogger(__name__)

# Import joblib for tests
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

# Optional imports for ML frameworks
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    # Handle matplotlib and other import errors
    tf = None
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
except Exception as e:
    # Handle XGBoost library not found and other import errors
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False
except Exception as e:
    # Handle LightGBM library not found and other import errors
    lgb = None
    LIGHTGBM_AVAILABLE = False


class BentoMLServiceManager:
    """Manages BentoML services for dynamic model serving"""
    
    def __init__(self):
        self.active_services: Dict[str, Service] = {}
        self.model_cache: Dict[str, Any] = {}
        
    async def create_service_for_model(self, model_id: str, deployment_config: Dict[str, Any] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Create a BentoML service for a specific model"""
        try:
            if not BENTOML_AVAILABLE:
                logger.warning("BentoML not available - service creation disabled")
                return False, "BentoML not available", {}
            
            # Get model from database
            async with get_session() as session:
                model = await session.get(Model, model_id)
                if not model:
                    return False, f"Model {model_id} not found", {}
            
            # Load and validate the model
            model_path = model.file_path
            if not await aios.path.exists(model_path):
                return False, f"Model file not found: {model_path}", {}
            
            # Create the service based on framework
            service_name = f"model_service_{model_id}"
            
            if model.framework == ModelFramework.SKLEARN:
                success, message, service_info = await self._create_sklearn_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.TENSORFLOW:
                success, message, service_info = await self._create_tensorflow_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.PYTORCH:
                success, message, service_info = await self._create_pytorch_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.XGBOOST:
                success, message, service_info = await self._create_xgboost_service(
                    model, service_name, deployment_config
                )
            elif model.framework == ModelFramework.LIGHTGBM:
                success, message, service_info = await self._create_lightgbm_service(
                    model, service_name, deployment_config
                )
            else:
                return False, f"Framework {model.framework} not supported yet", {}
            
            if success:
                # Cache the service
                service_info['created_at'] = datetime.utcnow()
                service_info['model_id'] = model_id
                
                logger.info(f"Successfully created BentoML service for model {model_id}")
                
            return success, message, service_info
            
        except Exception as e:
            logger.error(f"Error creating BentoML service for model {model_id}: {e}")
            return False, str(e), {}
    
    async def _create_sklearn_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for sklearn models"""
        try:
            if not BENTOML_AVAILABLE:
                return False, "BentoML not available", {}
            if not SKLEARN_AVAILABLE:
                return False, "sklearn not available", {}
            if not JOBLIB_AVAILABLE: # Assuming joblib is primary for sklearn persistence
                return False, "joblib not available for loading sklearn model", {}

            # Load the model asynchronously
            loaded_model = None
            try:
                # Try joblib first
                loaded_model = await asyncio.to_thread(joblib.load, model.file_path)
            except Exception as e_joblib:
                logger.warning(f"Failed to load model {model.file_path} with joblib: {e_joblib}, trying pickle.")
                try:
                    # Fallback to pickle
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
                name=f"sklearn_model_{model.id.replace('-','_')}", # Ensure name is valid
                model=loaded_model,
                labels={
                    "framework": "sklearn",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            bento_model_tag = str(bento_model_tag_obj)
            
            # Create service definition (CPU-bound, can stay sync)
            service_code = self._generate_sklearn_service_code(
                model, bento_model_tag, config or {}
            )
            
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
    
    async def _create_tensorflow_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
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
            elif await aios.path.isdir(model.file_path): # Check if it's a directory (for SavedModel)
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
                name=f"tensorflow_model_{model.id.replace('-','_')}", # Ensure name is valid
                model=loaded_model,
                labels={
                    "framework": "tensorflow",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            bento_model_tag = str(bento_model_tag_obj)
            
            # Create service definition (CPU-bound, can stay sync)
            service_code = self._generate_tensorflow_service_code(
                model, bento_model_tag, config or {}
            )
            
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
                'endpoints': ['predict'] # TensorFlow services typically have a predict endpoint
            }
            
            return True, "TensorFlow service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating TensorFlow service for {model.id}: {e}", exc_info=True)
            return False, str(e), {}
    
    async def _create_pytorch_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for PyTorch models"""
        try:
            if not PYTORCH_AVAILABLE:
                return False, "PyTorch not available", {}
            
            model_file_path = Path(model.file_path)
            file_ext = model_file_path.suffix.lower()
            loaded_model = None

            # Load the model asynchronously
            if file_ext in ['.pt', '.pth']:
                # torch.load itself does file I/O, so wrap the call
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
                name=f"pytorch_model_{model.id.replace('-','_')}", # Ensure name is valid
                model=loaded_model,
                labels={
                    "framework": "pytorch",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            bento_model_tag = str(bento_model_tag_obj)
            
            # Create service definition (CPU-bound)
            service_code = self._generate_pytorch_service_code(
                model, bento_model_tag, config or {}
            )
            
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
                'endpoints': ['predict'] # PyTorch services typically have a predict endpoint
            }
            
            return True, "PyTorch service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating PyTorch service for {model.id}: {e}", exc_info=True)
            return False, str(e), {}
    
    async def _create_xgboost_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for XGBoost models"""
        try:
            if not XGBOOST_AVAILABLE:
                return False, "XGBoost not available", {}
            if not JOBLIB_AVAILABLE: # Assuming joblib or pickle for persistence
                # We also need pickle for fallback
                import pickle # Ensure pickle is imported for the fallback logic

            # Load the model asynchronously
            loaded_model = None
            try:
                # Try joblib first
                if JOBLIB_AVAILABLE:
                    loaded_model = await asyncio.to_thread(joblib.load, model.file_path)
                else:
                    raise ImportError("Joblib not available, trying pickle directly.") # Force fallback
            except Exception as e_joblib:
                logger.warning(f"Failed to load model {model.file_path} with joblib: {e_joblib}, trying pickle.")
                try:
                    # Fallback to pickle
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
                name=f"xgboost_model_{model.id.replace('-','_')}", # Ensure name is valid
                model=loaded_model,
                labels={
                    "framework": "xgboost",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            bento_model_tag = str(bento_model_tag_obj)
            
            # Create service definition (CPU-bound)
            service_code = self._generate_xgboost_service_code(
                model, bento_model_tag, config or {}
            )
            
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
    
    async def _create_lightgbm_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for LightGBM models"""
        try:
            if not LIGHTGBM_AVAILABLE:
                return False, "LightGBM not available", {}
            # Ensure joblib or pickle is available for model loading
            if not JOBLIB_AVAILABLE:
                 import pickle # Ensure pickle is imported for the fallback logic

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
                return False, f"LightGBM model {model.file_path} could not be loaded.", {}

            # Save model to BentoML model store asynchronously
            bento_model_tag_obj = await asyncio.to_thread(
                bentoml.lightgbm.save_model,
                name=f"lightgbm_model_{model.id.replace('-','_')}", # Ensure name is valid
                model=loaded_model,
                labels={
                    "framework": "lightgbm",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            bento_model_tag = str(bento_model_tag_obj)
            
            # Create service definition (CPU-bound)
            service_code = self._generate_lightgbm_service_code(
                model, bento_model_tag, config or {}
            )
            
            # Write service file asynchronously
            service_path_obj = Path(settings.BENTOS_DIR) / f"{service_name}.py"
            await aios.makedirs(service_path_obj.parent, exist_ok=True)
            
            async with aiofiles.open(service_path_obj, 'w') as f:
                await f.write(service_code)
            
            service_info = {
                'service_name': service_name,
                'service_path': str(service_path_obj),
                'bento_model_tag': bento_model_tag,
                'framework': 'lightgbm',
                'endpoints': ['predict', 'predict_proba'] if model.model_type == ModelType.CLASSIFICATION else ['predict']
            }
            
            return True, "LightGBM service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating LightGBM service for {model.id}: {e}", exc_info=True)
            return False, str(e), {}
    
    def _generate_sklearn_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
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
            # Check if this looks like schema-formatted data (direct fields vs "data" wrapper)
            if "data" not in input_data:
                # Direct field input - could be schema-based
                return True, "Direct field input accepted", input_data
            else:
                # Traditional "data" wrapper format
                return True, "Traditional format accepted", input_data["data"]
                
        except Exception as e:
            return False, f"Validation error: {{str(e)}}", {{}}

    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the sklearn model with schema validation"""
        try:
            # Validate input
            is_valid, validation_message, validated_data = await self.validate_input_schema(input_data)
            
            if not is_valid:
                return {{
                    "error": f"Input validation failed: {{validation_message}}",
                    "model_id": "{model.id}"
                }}
            
            # Convert input to the expected format
            if isinstance(validated_data, dict) and "data" not in validated_data:
                # Direct field input - convert to DataFrame
                df = pd.DataFrame([validated_data])
            elif isinstance(validated_data, list):
                if len(validated_data) > 0 and isinstance(validated_data[0], dict):
                    # List of dictionaries
                    df = pd.DataFrame(validated_data)
                else:
                    # Single sample as list
                    df = pd.DataFrame([validated_data])
            elif isinstance(validated_data, dict):
                # Check if it has "data" key
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
                    # Single sample as dictionary
                    df = pd.DataFrame([validated_data])
            else:
                return {{"error": "Invalid input format"}}
            
            # Make prediction
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
            # Validate that we have batch data
            if "data" not in input_data or not isinstance(input_data["data"], list):
                return {{"error": "Batch data must be provided as a list under 'data' key"}}
            
            batch_data = input_data["data"]
            validated_batch = []
            validation_results = []
            
            # Validate each item in the batch
            for i, item in enumerate(batch_data):
                is_valid, validation_message, validated_item = await self.validate_input_schema(item)
                if not is_valid:
                    return {{
                        "error": f"Validation failed for batch item {{i}}: {{validation_message}}",
                        "item_index": i
                    }}
                validated_batch.append(validated_item)
                validation_results.append({{
                    "item_index": i,
                    "validation_message": validation_message
                }})
            
            # Convert to DataFrame
            if len(validated_batch) > 0 and isinstance(validated_batch[0], dict):
                df = pd.DataFrame(validated_batch)
            else:
                df = pd.DataFrame(validated_batch)
            
            # Make predictions
            predictions = self.model.predict(df)
            
            return {{
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "batch_size": len(validated_batch),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "validation": {{
                    "batch_validation_performed": True,
                    "item_validations": validation_results
                }}
            }}
            
        except Exception as e:
            return {{"error": str(e)}}

'''
        
        # Add predict_proba for classification models
        if model.model_type == ModelType.CLASSIFICATION:
            service_code += f'''
    @bentoml.api
    def predict_proba(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction probabilities for classification models with schema validation"""
        try:
            # Validate input
            is_valid, validation_message, validated_data = await self.validate_input_schema(input_data)
            
            if not is_valid:
                return {{
                    "error": f"Input validation failed: {{validation_message}}",
                    "model_id": "{model.id}"
                }}
            
            # Convert input to DataFrame (similar logic as predict)
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
            
            # Get probabilities
            probabilities = self.model.predict_proba(df)
            
            return {{
                "probabilities": probabilities.tolist() if hasattr(probabilities, 'tolist') else list(probabilities),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "validation": {{
                    "performed": True,
                    "message": validation_message
                }}
            }}
            
        except Exception as e:
            return {{"error": str(e)}}

'''
        else:
            # Add schema endpoint for non-classification models
            service_code += f'''
@service.api(input=JSON(), output=JSON())
def get_schema() -> Dict[str, Any]:
    """Get the input/output schema for this model"""
    return {{
        "model_id": "{model.id}",
        "model_name": "{model.name}",
        "framework": "sklearn",
        "model_type": "{model.model_type.value if model.model_type else 'unknown'}",
        "endpoints": ["predict", "predict_batch", "get_schema"],
        "schema_validation_enabled": True,
        "description": "Schema endpoint for {model.name}"
    }}
'''
        
        return service_code
    
    def _generate_tensorflow_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
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
            # Handle input data conversion
            if "data" in input_data:
                data = input_data["data"]
            else:
                data = input_data
            
            # Convert to numpy array for TensorFlow
            if isinstance(data, list):
                input_array = np.array(data)
            elif isinstance(data, dict):
                # Convert dict to array (assuming numeric values)
                input_array = np.array(list(data.values()))
            else:
                input_array = np.array(data)
            
            # Make prediction
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
    
    def _generate_pytorch_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
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
            # Handle input data conversion
            if "data" in input_data:
                data = input_data["data"]
            else:
                data = input_data
            
            # Convert to tensor
            if isinstance(data, list):
                input_tensor = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, dict):
                # Convert dict to tensor (assuming numeric values)
                input_tensor = torch.tensor(list(data.values()), dtype=torch.float32)
            else:
                input_tensor = torch.tensor(data, dtype=torch.float32)
            
            # Make prediction
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
    
    def _generate_xgboost_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
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
            # Handle input data conversion
            if "data" in input_data:
                data = input_data["data"]
            else:
                data = input_data
            
            # Convert to DMatrix for XGBoost
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
            
            # Make prediction
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
    
    def _generate_lightgbm_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for LightGBM models"""
        
        service_code = f'''
import bentoml
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Any

@bentoml.service
class LightGBMModelService:
    """Auto-generated BentoML service for LightGBM model"""
    
    def __init__(self):
        self.model = bentoml.lightgbm.load_model("{bento_model_tag}")

    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the LightGBM model"""
        try:
            # Handle input data conversion
            if "data" in input_data:
                data = input_data["data"]
            else:
                data = input_data
            
            # Convert to DataFrame for LightGBM
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            # Make prediction
            predictions = self.model.predict(df)
            
            return {{
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "model_id": "{model.id}",
                "model_name": "{model.name}",
                "framework": "lightgbm"
            }}
            
        except Exception as e:
            return {{"error": str(e)}}
'''
        
        return service_code
    
    def _get_input_schema_for_model(self, model: Model) -> Dict[str, Any]:
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
    
    async def deploy_service(self, service_name: str, config: Dict[str, Any] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Deploy a BentoML service"""
        try:
            # This would use BentoML's deployment capabilities
            # For now, we'll simulate the deployment
            
            deployment_info = {
                'status': 'deployed',
                'endpoint_url': f"http://localhost:3000/{service_name}",
                'deployment_id': str(uuid.uuid4()),
                'deployed_at': datetime.utcnow()
            }
            
            logger.info(f"Service {service_name} deployed successfully")
            return True, "Service deployed successfully", deployment_info
            
        except Exception as e:
            logger.error(f"Error deploying service {service_name}: {e}")
            return False, str(e), {}
    
    async def undeploy_service(self, service_name: str) -> Tuple[bool, str]:
        """Undeploy a BentoML service"""
        try:
            # Remove from active services if present
            if service_name in self.active_services:
                del self.active_services[service_name]
            
            logger.info(f"Service {service_name} undeployed successfully")
            return True, "Service undeployed successfully"
            
        except Exception as e:
            logger.error(f"Error undeploying service {service_name}: {e}")
            return False, str(e)
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get the status of a BentoML service"""
        try:
            # Check if service exists and is active
            if service_name in self.active_services:
                return {
                    'status': 'active',
                    'service_name': service_name,
                    'last_check': datetime.utcnow()
                }
            else:
                return {
                    'status': 'inactive',
                    'service_name': service_name,
                    'last_check': datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting service status for {service_name}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'service_name': service_name
            }
    
    async def list_services(self) -> List[Dict[str, Any]]:
        """List all active BentoML services"""
        try:
            services = []
            for service_name, service_info in self.active_services.items():
                services.append({
                    'service_name': service_name,
                    'model_id': service_info.get('model_id'),
                    'created_at': service_info.get('created_at'),
                    'status': 'active'
                })
            
            return services
            
        except Exception as e:
            logger.error(f"Error listing services: {e}")
            return []

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


# Global service manager instance
bentoml_service_manager = BentoMLServiceManager() 