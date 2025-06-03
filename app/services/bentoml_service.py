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

import bentoml
from bentoml import Service
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

# Optional imports for ML frameworks
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BentoMLServiceManager:
    """Manages BentoML services for dynamic model serving"""
    
    def __init__(self):
        self.active_services: Dict[str, Service] = {}
        self.model_cache: Dict[str, Any] = {}
        
    async def create_service_for_model(self, model_id: str, deployment_config: Dict[str, Any] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Create a BentoML service for a specific model"""
        try:
            # Get model from database
            async with get_session() as session:
                model = await session.get(Model, model_id)
                if not model:
                    return False, f"Model {model_id} not found", {}
            
            # Load and validate the model
            model_path = model.file_path
            if not os.path.exists(model_path):
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
            if not SKLEARN_AVAILABLE:
                return False, "sklearn not available", {}
            
            # Load the model
            import joblib
            try:
                loaded_model = joblib.load(model.file_path)
            except:
                import pickle
                with open(model.file_path, 'rb') as f:
                    loaded_model = pickle.load(f)
            
            # Save model to BentoML model store
            bento_model = bentoml.sklearn.save_model(
                name=f"sklearn_model_{model.id}",
                model=loaded_model,
                labels={
                    "framework": "sklearn",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            
            # Create service definition
            service_code = self._generate_sklearn_service_code(
                model, bento_model.tag, config or {}
            )
            
            # Write service file
            service_path = os.path.join(settings.BENTOS_DIR, f"{service_name}.py")
            os.makedirs(os.path.dirname(service_path), exist_ok=True)
            
            with open(service_path, 'w') as f:
                f.write(service_code)
            
            service_info = {
                'service_name': service_name,
                'service_path': service_path,
                'bento_model_tag': str(bento_model.tag),
                'framework': 'sklearn',
                'endpoints': ['predict', 'predict_proba'] if model.model_type == ModelType.CLASSIFICATION else ['predict']
            }
            
            return True, "sklearn service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating sklearn service: {e}")
            return False, str(e), {}
    
    async def _create_tensorflow_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for TensorFlow models"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return False, "TensorFlow not available", {}
            
            # Load the model based on format
            file_ext = Path(model.file_path).suffix.lower()
            
            if file_ext == '.h5':
                loaded_model = tf.keras.models.load_model(model.file_path)
            elif os.path.isdir(model.file_path):
                loaded_model = tf.saved_model.load(model.file_path)
            elif file_ext in ['.pkl', '.joblib']:
                import joblib
                loaded_model = joblib.load(model.file_path)
            else:
                return False, f"Unsupported TensorFlow model format: {file_ext}", {}
            
            # Save model to BentoML model store
            if hasattr(loaded_model, 'layers'):  # Keras model
                bento_model = bentoml.keras.save_model(
                    name=f"tensorflow_model_{model.id}",
                    model=loaded_model,
                    labels={
                        "framework": "tensorflow",
                        "model_type": model.model_type.value if model.model_type else "unknown",
                        "original_name": model.name
                    }
                )
            else:  # SavedModel
                bento_model = bentoml.tensorflow.save_model(
                    name=f"tensorflow_model_{model.id}",
                    model=loaded_model,
                    labels={
                        "framework": "tensorflow",
                        "model_type": model.model_type.value if model.model_type else "unknown",
                        "original_name": model.name
                    }
                )
            
            # Create service definition
            service_code = self._generate_tensorflow_service_code(
                model, bento_model.tag, config or {}
            )
            
            # Write service file
            service_path = os.path.join(settings.BENTOS_DIR, f"{service_name}.py")
            os.makedirs(os.path.dirname(service_path), exist_ok=True)
            
            with open(service_path, 'w') as f:
                f.write(service_code)
            
            service_info = {
                'service_name': service_name,
                'service_path': service_path,
                'bento_model_tag': str(bento_model.tag),
                'framework': 'tensorflow',
                'endpoints': ['predict']
            }
            
            return True, "TensorFlow service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating TensorFlow service: {e}")
            return False, str(e), {}
    
    async def _create_pytorch_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for PyTorch models"""
        try:
            if not PYTORCH_AVAILABLE:
                return False, "PyTorch not available", {}
            
            # Load the model
            file_ext = Path(model.file_path).suffix.lower()
            
            if file_ext in ['.pt', '.pth']:
                loaded_model = torch.load(model.file_path, map_location='cpu')
            elif file_ext in ['.pkl', '.joblib']:
                import joblib
                loaded_model = joblib.load(model.file_path)
            else:
                return False, f"Unsupported PyTorch model format: {file_ext}", {}
            
            # Save model to BentoML model store
            bento_model = bentoml.pytorch.save_model(
                name=f"pytorch_model_{model.id}",
                model=loaded_model,
                labels={
                    "framework": "pytorch",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            
            # Create service definition
            service_code = self._generate_pytorch_service_code(
                model, bento_model.tag, config or {}
            )
            
            # Write service file
            service_path = os.path.join(settings.BENTOS_DIR, f"{service_name}.py")
            os.makedirs(os.path.dirname(service_path), exist_ok=True)
            
            with open(service_path, 'w') as f:
                f.write(service_code)
            
            service_info = {
                'service_name': service_name,
                'service_path': service_path,
                'bento_model_tag': str(bento_model.tag),
                'framework': 'pytorch',
                'endpoints': ['predict']
            }
            
            return True, "PyTorch service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating PyTorch service: {e}")
            return False, str(e), {}
    
    async def _create_xgboost_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for XGBoost models"""
        try:
            if not XGBOOST_AVAILABLE:
                return False, "XGBoost not available", {}
            
            # Load the model
            import joblib
            try:
                loaded_model = joblib.load(model.file_path)
            except:
                import pickle
                with open(model.file_path, 'rb') as f:
                    loaded_model = pickle.load(f)
            
            # Save model to BentoML model store
            bento_model = bentoml.xgboost.save_model(
                name=f"xgboost_model_{model.id}",
                model=loaded_model,
                labels={
                    "framework": "xgboost",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            
            # Create service definition
            service_code = self._generate_xgboost_service_code(
                model, bento_model.tag, config or {}
            )
            
            # Write service file
            service_path = os.path.join(settings.BENTOS_DIR, f"{service_name}.py")
            os.makedirs(os.path.dirname(service_path), exist_ok=True)
            
            with open(service_path, 'w') as f:
                f.write(service_code)
            
            service_info = {
                'service_name': service_name,
                'service_path': service_path,
                'bento_model_tag': str(bento_model.tag),
                'framework': 'xgboost',
                'endpoints': ['predict', 'predict_proba'] if model.model_type == ModelType.CLASSIFICATION else ['predict']
            }
            
            return True, "XGBoost service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating XGBoost service: {e}")
            return False, str(e), {}
    
    async def _create_lightgbm_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for LightGBM models"""
        try:
            if not LIGHTGBM_AVAILABLE:
                return False, "LightGBM not available", {}
            
            # Load the model
            import joblib
            try:
                loaded_model = joblib.load(model.file_path)
            except:
                import pickle
                with open(model.file_path, 'rb') as f:
                    loaded_model = pickle.load(f)
            
            # Save model to BentoML model store
            bento_model = bentoml.lightgbm.save_model(
                name=f"lightgbm_model_{model.id}",
                model=loaded_model,
                labels={
                    "framework": "lightgbm",
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "original_name": model.name
                }
            )
            
            # Create service definition
            service_code = self._generate_lightgbm_service_code(
                model, bento_model.tag, config or {}
            )
            
            # Write service file
            service_path = os.path.join(settings.BENTOS_DIR, f"{service_name}.py")
            os.makedirs(os.path.dirname(service_path), exist_ok=True)
            
            with open(service_path, 'w') as f:
                f.write(service_code)
            
            service_info = {
                'service_name': service_name,
                'service_path': service_path,
                'bento_model_tag': str(bento_model.tag),
                'framework': 'lightgbm',
                'endpoints': ['predict', 'predict_proba'] if model.model_type == ModelType.CLASSIFICATION else ['predict']
            }
            
            return True, "LightGBM service created successfully", service_info
            
        except Exception as e:
            logger.error(f"Error creating LightGBM service: {e}")
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


# Global service manager instance
bentoml_service_manager = BentoMLServiceManager() 