"""
Model utilities for validation, file handling, and ML framework detection
Provides comprehensive support for multiple ML frameworks including TensorFlow and PyTorch
"""

import hashlib
import os
import pickle
import joblib
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import aiofiles
import aiofiles.os as aios
import asyncio
from datetime import datetime

from app.schemas.model import ModelFramework, ModelType, ModelValidationResult
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Optional imports for TensorFlow and PyTorch
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False
    logger.info("TensorFlow not available - limited TensorFlow model support")
except Exception as e:
    # Handle matplotlib and other import errors
    tf = None
    TENSORFLOW_AVAILABLE = False
    logger.info(f"TensorFlow not available due to dependency issue: {e}")

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False
    logger.info("PyTorch not available - limited PyTorch model support")

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None
    SKLEARN_AVAILABLE = False
    logger.info("Scikit-learn not available - limited scikit-learn model support")

# Add other optional imports at module level
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    xgboost = None
    XGBOOST_AVAILABLE = False
except Exception as e:
    # Handle XGBoost library not found and other import errors
    xgboost = None
    XGBOOST_AVAILABLE = False
    logger.info(f"XGBoost not available due to dependency issue: {e}")

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lightgbm = None
    LIGHTGBM_AVAILABLE = False
except Exception as e:
    # Handle LightGBM library not found and other import errors
    lightgbm = None
    LIGHTGBM_AVAILABLE = False
    logger.info(f"LightGBM not available due to dependency issue: {e}")


class ModelValidator:
    """Comprehensive model validation for multiple ML frameworks"""
    
    @staticmethod
    async def calculate_file_hash_async(file_path: str) -> str:
        """Calculate SHA-256 hash of a file asynchronously"""
        hash_sha256 = hashlib.sha256()
        try:
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(4096):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash asynchronously: {e}")
            raise
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate if file extension is supported"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in settings.ALLOWED_MODEL_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate if file size is within limits"""
        return file_size <= settings.MAX_FILE_SIZE
    
    @staticmethod
    async def _is_savedmodel_directory_async(dir_path: str) -> bool:
        """Check if directory is a TensorFlow SavedModel asynchronously"""
        if not await aios.path.isdir(dir_path):
            return False
        
        # Check for required SavedModel files
        # os.path.join is fine as it's path string manipulation, not I/O
        saved_model_pb = os.path.join(dir_path, "saved_model.pb") 
        variables_dir = os.path.join(dir_path, "variables")
        
        return await aios.path.exists(saved_model_pb) and await aios.path.isdir(variables_dir)
    
    @classmethod
    async def detect_framework_from_file_async(cls, file_path: str) -> Optional[ModelFramework]:
        """Detect ML framework from model file asynchronously"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Handle different file types
            if file_ext in ['.pkl', '.joblib']:
                return await cls._detect_pickle_framework_async(file_path)
            elif file_ext in ['.h5', '.pb', '.tflite']:
                # These are typically TensorFlow, direct detection.
                # If .pb is also used by PyTorch (e.g. TorchScript), more checks would be needed.
                return ModelFramework.TENSORFLOW 
            elif file_ext in ['.pt', '.pth']:
                # These are typically PyTorch.
                return ModelFramework.PYTORCH
            elif file_ext == '.onnx':
                return ModelFramework.ONNX
            elif file_ext == '.json':
                return await cls._detect_json_framework_async(file_path)
            elif file_ext == '.zip':
                return await cls._detect_zip_framework_async(file_path)
            
            # Check if it's a directory (for SavedModel)
            # os.path.isdir is a quick check, if true, then do async check for content
            if await aios.path.isdir(file_path) and await cls._is_savedmodel_directory_async(file_path):
                return ModelFramework.TENSORFLOW
            
            logger.warning(f"Could not detect framework for {file_path} with extension '{file_ext}'.")
            return None
        except Exception as e:
            logger.error(f"Error detecting framework for {file_path}: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def _detect_pickle_framework_async(file_path: str) -> Optional[ModelFramework]:
        """Detect framework for pickle/joblib files asynchronously"""
        try:
            model = None
            # Try joblib first
            try:
                model = await asyncio.to_thread(joblib.load, file_path)
            except Exception as e_joblib:
                logger.debug(f"joblib.load failed for {file_path}: {e_joblib}. Falling back to pickle.")
                # Fall back to pickle
                try:
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    model = await asyncio.to_thread(pickle.loads, content)
                except Exception as e_pickle:
                    logger.error(f"Error loading pickle file {file_path} after joblib failed: {e_pickle}")
                    return None # Could not load the file
            
            if model is None:
                logger.warning(f"Could not load model from {file_path} using joblib or pickle.")
                return None

            model_type_str = str(type(model))
            model_module = getattr(model, '__module__', '')
            model_class_name = model.__class__.__name__
            
            # Check for XGBoost more thoroughly
            if ('xgboost' in model_type_str.lower() or 
                'xgb' in model_module.lower() or 
                'xgboost' in model_module.lower() or
                'XGBClassifier' in model_class_name or
                'XGBRegressor' in model_class_name or
                'Booster' in model_class_name):
                return ModelFramework.XGBOOST
            
            # Check for LightGBM
            if ('lightgbm' in model_type_str.lower() or 
                'lgb' in model_module.lower() or
                'lightgbm' in model_module.lower() or
                'LGBMClassifier' in model_class_name or
                'LGBMRegressor' in model_class_name):
                return ModelFramework.LIGHTGBM
            
            # Check for sklearn
            if ('sklearn' in model_type_str.lower() or 
                'sklearn' in model_module.lower()):
                return ModelFramework.SKLEARN
            
            # Check for TensorFlow/Keras (when saved as pickle)
            if ('tensorflow' in model_type_str.lower() or 
                'keras' in model_type_str.lower() or
                'tf.' in model_module.lower()):
                return ModelFramework.TENSORFLOW
            
            # Check for PyTorch
            if ('torch' in model_type_str.lower() or 
                'pytorch' in model_module.lower()):
                return ModelFramework.PYTORCH
            
            # Default to sklearn for pickle files if no specific framework detected
            logger.info(f"Pickled model {file_path} defaulted to SKLEARN framework detection.")
            return ModelFramework.SKLEARN
            
        except Exception as e:
            logger.error(f"Error detecting framework from async pickle file {file_path}: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def _detect_json_framework_async(file_path: str) -> Optional[ModelFramework]:
        """Detect framework for JSON files asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            # json.loads is CPU bound, but usually very fast. For very large JSONs, consider to_thread.
            # For typical model config JSONs, direct await asyncio.to_thread(json.loads, content) might be overkill.
            # Let's use json.loads directly for now and monitor if it becomes a bottleneck.
            # If it's large schema files, then to_thread is better.
            # For now, assuming relatively small JSON model descriptors.
            data = await asyncio.to_thread(json.loads, content) # Using to_thread for safety with potentially larger JSONs
            
            # Check for framework indicators in JSON
            if isinstance(data, dict):
                data_str = str(data).lower() # This could be inefficient for very large dicts
                                         # Consider checking specific keys if structure is known
                if 'xgboost' in data_str:
                    return ModelFramework.XGBOOST
                elif 'lightgbm' in data_str:
                    return ModelFramework.LIGHTGBM
                elif 'h2o' in data_str:
                    return ModelFramework.H2O
                elif 'tensorflow' in data_str or 'keras' in data_str:
                    return ModelFramework.TENSORFLOW
                elif 'pytorch' in data_str or 'torch' in data_str:
                    return ModelFramework.PYTORCH
            
            return ModelFramework.CUSTOM
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error detecting JSON framework for {file_path}: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def _detect_zip_framework_async(file_path: str) -> Optional[ModelFramework]:
        """Detect framework for ZIP files (common for SavedModel) asynchronously"""
        
        def sync_zip_operations(path):
            try:
                with zipfile.ZipFile(path, 'r') as zip_file:
                    file_list = zip_file.namelist()
                    
                    # Check for TensorFlow SavedModel structure
                    if any('saved_model.pb' in f for f in file_list):
                        return ModelFramework.TENSORFLOW
                    
                    # Check for PyTorch files
                    if any(f.endswith(('.pt', '.pth')) for f in file_list):
                        return ModelFramework.PYTORCH
                    
                    # Check for other frameworks (e.g. custom bento with model.json)
                    if any('model.json' in f or 'config.json' in f for f in file_list):
                        # This could be a custom bento or other structured zip.
                        # Further checks might be needed if more specific detection is required.
                        return ModelFramework.CUSTOM 
                
                return ModelFramework.CUSTOM # Default if specific structures not found
            except zipfile.BadZipFile as e:
                logger.error(f"Bad ZIP file {path}: {e}")
                return None
            except Exception as e:
                # Catching generic Exception to handle any other zipfile-related errors
                logger.error(f"Error processing ZIP file {path}: {e}", exc_info=True)
                return None

        try:
            return await asyncio.to_thread(sync_zip_operations, file_path)
        except Exception as e: # Should not happen if sync_zip_operations handles its exceptions
            logger.error(f"Error in _detect_zip_framework_async for {file_path}: {e}", exc_info=True)
            return None
    
    @classmethod
    async def detect_model_type_async(cls, file_path: str, framework: ModelFramework) -> Optional[ModelType]:
        """Detect model type based on the loaded model asynchronously"""
        try:
            if framework == ModelFramework.SKLEARN:
                return await cls._detect_sklearn_model_type_async(file_path)
            elif framework in [ModelFramework.XGBOOST, ModelFramework.LIGHTGBM]:
                return await cls._detect_boosting_model_type_async(file_path)
            elif framework == ModelFramework.TENSORFLOW:
                return await cls._detect_tensorflow_model_type_async(file_path)
            elif framework == ModelFramework.PYTORCH:
                return await cls._detect_pytorch_model_type_async(file_path)
            
            logger.warning(f"Model type detection not implemented for framework: {framework}")
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error detecting model type for {file_path} (framework: {framework}): {e}", exc_info=True)
            return ModelType.OTHER

    @staticmethod
    async def _detect_sklearn_model_type_async(file_path: str) -> ModelType:
        """Detect sklearn model type asynchronously"""
        try:
            model = None
            try:
                # Try joblib first
                model = await asyncio.to_thread(joblib.load, file_path)
            except Exception as e_joblib:
                logger.debug(f"joblib.load failed for {file_path} in _detect_sklearn_model_type_async: {e_joblib}. Falling back to pickle.")
                try:
                    # Fall back to pickle
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    model = await asyncio.to_thread(pickle.loads, content)
                except Exception as e_pickle:
                    logger.error(f"Error loading sklearn model from pickle {file_path}: {e_pickle}")
                    return ModelType.OTHER # Could not load

            if model is None:
                logger.warning(f"Could not load sklearn model from {file_path} for type detection.")
                return ModelType.OTHER
            
            model_name = model.__class__.__name__.lower()
            
            # Classification models
            if any(word in model_name for word in ['classifier', 'svc', 'naive', 'logistic']):
                return ModelType.CLASSIFICATION
            
            # Regression models
            elif any(word in model_name for word in ['regressor', 'svr', 'linear']):
                return ModelType.REGRESSION
            
            # Clustering models
            elif any(word in model_name for word in ['kmeans', 'cluster', 'dbscan', 'agglomerative']):
                return ModelType.CLUSTERING
            
            # Other types (e.g., transformers, decomposers)
            else:
                # Check for predict_proba for classification as a fallback
                if hasattr(model, 'predict_proba'):
                    return ModelType.CLASSIFICATION
                # Check for predict for regression/generic prediction
                elif hasattr(model, 'predict'):
                    return ModelType.REGRESSION # Or a more generic type if available
                # Check for fit_transform for transformers/clustering
                elif hasattr(model, 'fit_transform') or hasattr(model, 'transform'):
                    return ModelType.OTHER # Could be clustering or a preprocessor
                
                logger.info(f"Could not determine specific sklearn model type for {model_name}, defaulting to OTHER.")
                return ModelType.OTHER
            
        except Exception as e:
            logger.error(f"Error detecting sklearn model type for {file_path}: {e}", exc_info=True)
            return ModelType.OTHER
    
    @staticmethod
    async def _detect_boosting_model_type_async(file_path: str) -> ModelType:
        """Detect XGBoost/LightGBM model type asynchronously"""
        try:
            model = None
            try:
                # Try joblib first
                model = await asyncio.to_thread(joblib.load, file_path)
            except Exception as e_joblib:
                logger.debug(f"joblib.load failed for {file_path} in _detect_boosting_model_type_async: {e_joblib}. Falling back to pickle.")
                try:
                    # Fall back to pickle
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    model = await asyncio.to_thread(pickle.loads, content)
                except Exception as e_pickle:
                    logger.error(f"Error loading boosting model from pickle {file_path}: {e_pickle}")
                    return ModelType.OTHER # Could not load

            if model is None:
                logger.warning(f"Could not load boosting model from {file_path} for type detection.")
                return ModelType.OTHER

            model_name = model.__class__.__name__.lower()
            
            # Check for predict_proba method for classification
            if hasattr(model, 'predict_proba'):
                return ModelType.CLASSIFICATION
            # Check for predict method for regression
            elif hasattr(model, 'predict'):
                return ModelType.REGRESSION
            else:
                logger.info(f"Could not determine specific boosting model type for {model_name}, defaulting to OTHER.")
                return ModelType.OTHER

        except Exception as e:
            logger.error(f"Error detecting boosting model type for {file_path}: {e}", exc_info=True)
            return ModelType.OTHER
    
    @staticmethod
    async def _detect_tensorflow_model_type_async(file_path: str) -> ModelType:
        """Detect TensorFlow model type asynchronously"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping TensorFlow model type detection.")
            return ModelType.OTHER

        try:
            file_ext = Path(file_path).suffix.lower()
            model_info = {"type": ModelType.OTHER} # Use a dict to pass by reference

            def sync_tf_load_and_analyze(path_str, ext_str, info_dict):
                model = None
                if ext_str == '.h5':
                    model = tf.keras.models.load_model(path_str)
                    info_dict["type"] = ModelValidator._analyze_keras_model(model)
                elif ext_str == '.pb': # Assuming a frozen graph or SavedModel .pb
                    # This is more complex. A .pb can be part of SavedModel or a frozen graph.
                    # If it's a SavedModel, the directory structure is key.
                    # For simplicity here, if it's a standalone .pb, it's hard to infer type without loading.
                    # We might need to try loading it as a SavedModel if its parent dir looks like one.
                    # For now, assume OTHER if it's a loose .pb file.
                    # If file_path is actually a directory for SavedModel, it's handled below.
                    logger.info(f"Standalone .pb file {path_str} detected, type detection might be limited.")
                    # Try to load as SavedModel if parent looks like it
                    parent_dir = Path(path_str).parent
                    if ModelValidator._is_savedmodel_directory_sync_for_tf_detection(str(parent_dir)): # Temp sync helper
                         model = tf.saved_model.load(str(parent_dir))
                         info_dict["type"] = ModelValidator._analyze_savedmodel(model)
                    else: # If not, might be a frozen graph, harder to analyze generically
                        info_dict["type"] = ModelType.OTHER
                elif ext_str == '.tflite':
                    interpreter = tf.lite.Interpreter(model_path=path_str)
                    info_dict["type"] = ModelValidator._analyze_tflite_model(interpreter)
                elif os.path.isdir(path_str) and ModelValidator._is_savedmodel_directory_sync_for_tf_detection(path_str): # Temp sync helper
                    model = tf.saved_model.load(path_str)
                    info_dict["type"] = ModelValidator._analyze_savedmodel(model)
                else:
                    logger.warning(f"Unsupported TensorFlow file/directory for type detection: {path_str}")
                    info_dict["type"] = ModelType.OTHER

            await asyncio.to_thread(sync_tf_load_and_analyze, file_path, file_ext, model_info)
            return model_info["type"]

        except Exception as e:
            logger.error(f"Error detecting TensorFlow model type for {file_path}: {e}", exc_info=True)
            return ModelType.OTHER

    @staticmethod
    def _is_savedmodel_directory_sync_for_tf_detection(dir_path: str) -> bool:
        """Synchronous version for internal TF type detection use only. Avoids await in sync_tf_load_and_analyze."""
        if not os.path.isdir(dir_path):
            return False
        saved_model_pb = os.path.join(dir_path, "saved_model.pb")
        variables_dir = os.path.join(dir_path, "variables")
        return os.path.exists(saved_model_pb) and os.path.isdir(variables_dir)

    @staticmethod
    def _analyze_keras_model(model) -> ModelType:
        """Analyze Keras model to determine type"""
        try:
            if hasattr(model, 'layers') and model.layers:
                # Check output layer
                output_layer = model.layers[-1]
                output_shape = output_layer.output_shape
                
                if hasattr(output_layer, 'activation'):
                    activation = str(output_layer.activation).lower()
                    
                    # Classification indicators
                    if 'softmax' in activation or 'sigmoid' in activation:
                        return ModelType.CLASSIFICATION
                    
                    # Regression indicators (linear activation or no activation)
                    elif 'linear' in activation or activation == 'none':
                        return ModelType.REGRESSION
                
                # Check output shape for hints
                if isinstance(output_shape, (list, tuple)) and len(output_shape) > 1:
                    output_size = output_shape[-1]
                    if output_size == 1:
                        return ModelType.REGRESSION  # Single output often regression
                    elif output_size > 1:
                        return ModelType.CLASSIFICATION  # Multiple outputs often classification
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error analyzing Keras model: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _analyze_savedmodel(model) -> ModelType:
        """Analyze SavedModel to determine type"""
        try:
            # Try to get signature information
            if hasattr(model, 'signatures'):
                for signature_key, signature in model.signatures.items():
                    if hasattr(signature, 'outputs'):
                        for output_key, output_spec in signature.outputs.items():
                            # Analyze output specifications
                            if hasattr(output_spec, 'shape') and output_spec.shape:
                                output_shape = output_spec.shape.as_list()
                                if len(output_shape) >= 2:
                                    output_size = output_shape[-1]
                                    if output_size == 1:
                                        return ModelType.REGRESSION
                                    elif output_size > 1:
                                        return ModelType.CLASSIFICATION
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error analyzing SavedModel: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _analyze_tflite_model(interpreter) -> ModelType:
        """Analyze TensorFlow Lite model to determine type"""
        try:
            output_details = interpreter.get_output_details()
            if output_details:
                output_shape = output_details[0]['shape']
                if len(output_shape) >= 2:
                    output_size = output_shape[-1]
                    if output_size == 1:
                        return ModelType.REGRESSION
                    elif output_size > 1:
                        return ModelType.CLASSIFICATION
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error analyzing TensorFlow Lite model: {e}")
            return ModelType.OTHER
    
    @staticmethod
    async def _detect_pytorch_model_type_async(file_path: str) -> ModelType:
        """Detect PyTorch model type asynchronously"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping PyTorch model type detection.")
            return ModelType.OTHER

        try:
            model_info = {"type": ModelType.OTHER} # Use a dict to pass by reference

            def sync_pytorch_load_and_analyze(path_str, info_dict):
                # Try loading full model first (common for .pt, .pth)
                try:
                    # torch.load itself does file I/O and CPU work
                    model = torch.load(path_str, map_location='cpu') 
                    if isinstance(model, dict): # Likely a state_dict
                        info_dict["type"] = ModelValidator._analyze_pytorch_state_dict(model)
                    else: # Assumed to be a full model
                        info_dict["type"] = ModelValidator._analyze_pytorch_model(model)
                    return # Successfully processed
                except Exception as e_load:
                    logger.debug(f"Failed to load PyTorch model/state_dict directly from {path_str}: {e_load}")
                
                # Fallback for .pkl/.joblib (less common for PyTorch but possible)
                file_ext = Path(path_str).suffix.lower()
                if file_ext in ['.pkl', '.joblib']:
                    try:
                        # Try joblib first
                        model = joblib.load(path_str)
                    except Exception as e_joblib:
                        logger.debug(f"joblib.load failed for PyTorch pkl {path_str}: {e_joblib}. Trying pickle.")
                        try:
                            with open(path_str, 'rb') as f_pkl: # Standard sync open inside thread
                                model = pickle.load(f_pkl)
                        except Exception as e_pickle:
                            logger.error(f"Failed to load PyTorch model from pkl/joblib {path_str} with pickle: {e_pickle}")
                            info_dict["type"] = ModelType.OTHER
                            return
                    
                    if isinstance(model, dict): # State dict in a pickle
                        info_dict["type"] = ModelValidator._analyze_pytorch_state_dict(model)
                    else: # Full model in a pickle
                        info_dict["type"] = ModelValidator._analyze_pytorch_model(model)
                    return
                
                info_dict["type"] = ModelType.OTHER # If all attempts fail
            
            await asyncio.to_thread(sync_pytorch_load_and_analyze, file_path, model_info)
            return model_info["type"]

        except Exception as e:
            logger.error(f"Error detecting PyTorch model type for {file_path}: {e}", exc_info=True)
            return ModelType.OTHER

    @staticmethod
    def _analyze_pytorch_model(model) -> ModelType:
        """Analyze PyTorch model to determine type"""
        try:
            # Check if it's a nn.Module
            if hasattr(model, 'modules'):
                modules = list(model.modules())
                
                # Look for final layer patterns
                for module in reversed(modules):
                    module_name = module.__class__.__name__.lower()
                    
                    # Classification indicators
                    if any(word in module_name for word in ['softmax', 'logsoftmax', 'sigmoid']):
                        return ModelType.CLASSIFICATION
                    
                    # Check linear layers for output size
                    if 'linear' in module_name and hasattr(module, 'out_features'):
                        out_features = module.out_features
                        if out_features == 1:
                            return ModelType.REGRESSION
                        elif out_features > 1:
                            return ModelType.CLASSIFICATION
            
            # Check if it's a state dict
            elif isinstance(model, dict):
                return ModelValidator._analyze_pytorch_state_dict(model)
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error analyzing PyTorch model: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _analyze_pytorch_state_dict(state_dict) -> ModelType:
        """Analyze PyTorch state dict to determine type"""
        try:
            if isinstance(state_dict, dict):
                # Look for final layer weights
                for key in state_dict.keys():
                    if key.endswith(('.weight', '.bias')) and 'fc' in key.lower():
                        # Final fully connected layer
                        weight_tensor = state_dict[key]
                        if hasattr(weight_tensor, 'shape'):
                            if 'weight' in key:
                                output_size = weight_tensor.shape[0]
                                if output_size == 1:
                                    return ModelType.REGRESSION
                                elif output_size > 1:
                                    return ModelType.CLASSIFICATION
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error analyzing PyTorch state dict: {e}")
            return ModelType.OTHER
    
    @classmethod
    async def validate_model_file_async(cls, file_path: str) -> ModelValidationResult:
        """Validate model file asynchronously, including structure, framework, type, and metadata"""
        errors = []
        warnings = []
        detected_framework: Optional[ModelFramework] = None
        detected_model_type: Optional[ModelType] = None
        metadata: Dict[str, Any] = {}

        # Basic file checks
        if not await aios.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            return ModelValidationResult(
                is_valid=False, 
                errors=errors, 
                warnings=warnings,
                framework_detected=None,
                model_type_detected=None,
                metadata=metadata
            )
        
        filename = Path(file_path).name
        file_size = (await aios.stat(file_path)).st_size

        if not cls.validate_file_extension(filename):
            errors.append(f"Unsupported file extension: {filename}")
        
        if not cls.validate_file_size(file_size):
            errors.append(f"File size exceeds limit: {file_size} bytes")

        # Framework detection
        try:
            detected_framework = await cls.detect_framework_from_file_async(file_path)
        except Exception as e:
            errors.append(f"Error detecting framework: {str(e)}")
            logger.error(f"Framework detection failed for {file_path}: {e}", exc_info=True)

        if detected_framework:
            metadata['detected_framework'] = detected_framework.value
            # Model type detection (depends on framework)
            try:
                detected_model_type = await cls.detect_model_type_async(file_path, detected_framework)
            except Exception as e:
                errors.append(f"Error detecting model type: {str(e)}")
                logger.error(f"Model type detection failed for {file_path}: {e}", exc_info=True)
            
            if detected_model_type:
                metadata['detected_model_type'] = detected_model_type.value

            # Metadata extraction (depends on framework)
            try:
                specific_metadata = {}
                if detected_framework == ModelFramework.SKLEARN:
                    specific_metadata = await cls._get_sklearn_metadata_async(file_path)
                elif detected_framework == ModelFramework.XGBOOST:
                    specific_metadata = await cls._get_xgboost_metadata_async(file_path)
                elif detected_framework == ModelFramework.LIGHTGBM:
                    specific_metadata = await cls._get_lightgbm_metadata_async(file_path)
                elif detected_framework == ModelFramework.TENSORFLOW:
                    specific_metadata = await cls._get_tensorflow_metadata_async(file_path)
                elif detected_framework == ModelFramework.PYTORCH:
                    specific_metadata = await cls._get_pytorch_metadata_async(file_path)
                metadata.update(specific_metadata)
            except Exception as e:
                errors.append(f"Error extracting metadata: {str(e)}")
                logger.error(f"Metadata extraction failed for {file_path}: {e}", exc_info=True)
        else:
            if not errors: # Only add this warning if no other errors are present
                warnings.append("Could not determine model framework. Limited validation performed.")

        # Calculate file hash
        try:
            file_hash = await cls.calculate_file_hash_async(file_path)
            metadata['file_hash'] = file_hash
        except Exception as e:
            errors.append(f"Error calculating file hash: {str(e)}")
            logger.error(f"File hash calculation failed for {file_path}: {e}", exc_info=True)
        
        metadata['file_name'] = filename
        metadata['file_size'] = file_size
        
        is_valid = not errors
        
        return ModelValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            framework_detected=detected_framework,
            model_type_detected=detected_model_type,
            metadata=metadata
        )

    @staticmethod
    async def _get_sklearn_metadata_async(file_path: str) -> Dict[str, Any]:
        """Get metadata from sklearn model asynchronously"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn is not available. Skipping sklearn metadata extraction.")
            return {}
        
        metadata = {}
        try:
            def sync_load_and_meta(path):
                _metadata = {}
                model = None
                try:
                    model = joblib.load(path)
                except Exception as e_joblib:
                    logger.debug(f"joblib.load failed for sklearn metadata {path}: {e_joblib}. Trying pickle.")
                    try:
                        with open(path, 'rb') as f:
                            model = pickle.load(f)
                    except Exception as e_pickle:
                        logger.error(f"Pickle load failed for sklearn metadata {path}: {e_pickle}")
                        return {"error": f"Could not load model: {e_pickle}"}
                
                if model is None:
                    return {"error": "Could not load sklearn model for metadata."}

                _metadata['model_class'] = model.__class__.__name__
                if hasattr(model, 'get_params'):
                    try:
                        _metadata['parameters'] = model.get_params()
                    except Exception as e_params:
                        logger.warning(f"Could not get_params for sklearn model {path}: {e_params}")
                        _metadata['parameters'] = "Error retrieving params"
                if hasattr(model, 'feature_names_in_'):
                    try:
                        _metadata['feature_names_in'] = list(model.feature_names_in_)
                    except Exception as e_feat:
                         logger.warning(f"Could not get feature_names_in_ for sklearn model {path}: {e_feat}")
                if hasattr(model, 'n_features_in_'):
                    _metadata['n_features_in'] = model.n_features_in_
                return _metadata

            metadata = await asyncio.to_thread(sync_load_and_meta, file_path)
        except Exception as e:
            logger.error(f"Error getting sklearn metadata for {file_path}: {e}", exc_info=True)
            metadata['error'] = f"Error extracting sklearn metadata: {str(e)}"
        return metadata

    @staticmethod
    async def _get_xgboost_metadata_async(file_path: str) -> Dict[str, Any]:
        """Get metadata from XGBoost model asynchronously"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost is not available. Skipping XGBoost metadata extraction.")
            return {}

        metadata = {}
        try:
            def sync_load_and_meta(path):
                _metadata = {}
                model = None
                try:
                    model = joblib.load(path)
                except Exception as e_joblib:
                    logger.debug(f"joblib.load failed for xgboost metadata {path}: {e_joblib}. Trying pickle.")
                    try:
                        with open(path, 'rb') as f:
                            model = pickle.load(f)
                    except Exception as e_pickle:
                        logger.error(f"Pickle load failed for xgboost metadata {path}: {e_pickle}")
                        return {"error": f"Could not load model: {e_pickle}"}
                
                if model is None:
                    return {"error": "Could not load XGBoost model for metadata."}

                _metadata['model_class'] = model.__class__.__name__
                if hasattr(model, 'get_params'):
                    try:
                        _metadata['parameters'] = model.get_params()
                    except Exception as e_params:
                        logger.warning(f"Could not get_params for XGBoost model {path}: {e_params}")
                if hasattr(model, 'feature_names') and model.feature_names is not None:
                    _metadata['feature_names'] = model.feature_names
                elif hasattr(model, 'feature_name') and callable(model.feature_name):
                    try:
                         _metadata['feature_names'] = model.feature_name()
                    except: # Some older versions might error or not have it as a callable
                        pass 
                if hasattr(model, 'n_features_in_'):
                     _metadata['n_features_in'] = model.n_features_in_
                # XGBoost specific params often in booster
                if hasattr(model, 'booster') and hasattr(model.booster, 'attributes'):
                     _metadata['booster_attributes'] = model.booster.attributes()
                return _metadata

            metadata = await asyncio.to_thread(sync_load_and_meta, file_path)
        except Exception as e:
            logger.error(f"Error getting XGBoost metadata for {file_path}: {e}", exc_info=True)
            metadata['error'] = f"Error extracting XGBoost metadata: {str(e)}"
        return metadata

    @staticmethod
    async def _get_lightgbm_metadata_async(file_path: str) -> Dict[str, Any]:
        """Extract LightGBM-specific metadata asynchronously"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM is not available. Skipping LightGBM metadata extraction.")
            return {}
        
        metadata = {}
        try:
            def sync_load_and_meta(path):
                _metadata = {}
                model = None
                try:
                    model = joblib.load(path)
                except Exception as e_joblib:
                    logger.debug(f"joblib.load failed for lightgbm metadata {path}: {e_joblib}. Trying pickle.")
                    try:
                        with open(path, 'rb') as f:
                            model = pickle.load(f)
                    except Exception as e_pickle:
                        logger.error(f"Pickle load failed for lightgbm metadata {path}: {e_pickle}")
                        return {"error": f"Could not load model: {e_pickle}"}
                
                if model is None:
                    return {"error": "Could not load LightGBM model for metadata."}

                _metadata['model_class'] = model.__class__.__name__
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        _metadata['parameters'] = params # Store all params
                        _metadata['n_estimators'] = params.get('n_estimators')
                        _metadata['objective'] = params.get('objective')
                        _metadata['num_leaves'] = params.get('num_leaves')
                    except Exception as e_params:
                        logger.warning(f"Could not get_params for LightGBM model {path}: {e_params}")    
                if hasattr(model, 'feature_name') and callable(model.feature_name):
                    try:
                        _metadata['feature_names'] = model.feature_name()
                    except Exception as e_feat_name:
                        logger.warning(f"Could not get feature_name for LightGBM model {path}: {e_feat_name}")
                if hasattr(model, 'n_features_'):
                    _metadata['n_features'] = model.n_features_
                elif hasattr(model, 'n_features_in_'):
                     _metadata['n_features_in'] = model.n_features_in_
                return _metadata

            metadata = await asyncio.to_thread(sync_load_and_meta, file_path)
        except Exception as e:
            logger.error(f"Error extracting LightGBM metadata for {file_path}: {e}", exc_info=True)
            metadata['error'] = f"Error extracting LightGBM metadata: {str(e)}"
        return metadata

    @staticmethod
    async def _get_tensorflow_metadata_async(file_path: str) -> Dict[str, Any]:
        """Extract TensorFlow-specific metadata asynchronously"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping TensorFlow metadata extraction.")
            return {'tensorflow_available': False}
        
        metadata = {'tensorflow_available': True}
        file_ext = Path(file_path).suffix.lower()

        def sync_tf_meta_extraction(path_str, ext_str, current_meta_unused): # Renamed current_meta to current_meta_unused as it's not used
            _metadata_update = {}
            try:
                if ext_str == '.h5':
                    model = tf.keras.models.load_model(path_str)
                    _metadata_update.update({
                        'model_type': 'keras_h5',
                        'layer_count': len(model.layers),
                        'trainable_params': model.count_params(),
                        'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else None,
                        'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else None
                    })
                    if hasattr(model, 'optimizer') and model.optimizer is not None:
                        _metadata_update['optimizer'] = model.optimizer.__class__.__name__
                
                elif ext_str == '.tflite':
                    interpreter = tf.lite.Interpreter(model_path=path_str)
                    interpreter.allocate_tensors()
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    _metadata_update.update({
                        'model_type': 'tflite',
                        'input_count': len(input_details),
                        'output_count': len(output_details),
                        'input_shapes': [detail['shape'].tolist() for detail in input_details],
                        'output_shapes': [detail['shape'].tolist() for detail in output_details]
                    })
                
                elif os.path.isdir(path_str) or (ext_str == '.pb' and ModelValidator._is_savedmodel_directory_sync_for_tf_detection(path_str)):
                    target_load_path = path_str
                    if ext_str == '.pb' and not os.path.isdir(path_str):
                        parent_dir = str(Path(path_str).parent)
                        if ModelValidator._is_savedmodel_directory_sync_for_tf_detection(parent_dir):
                           target_load_path = parent_dir
                    
                    model = tf.saved_model.load(target_load_path)
                    _metadata_update.update({
                        'model_type': 'savedmodel',
                        'signature_keys': list(model.signatures.keys()) if hasattr(model, 'signatures') else []
                    })
                else:
                    logger.info(f"TensorFlow metadata extraction: Unsupported file type or structure at {path_str}")
                    _metadata_update['load_info'] = "Unsupported TF file for metadata extraction"

            except Exception as e_tf_meta:
                logger.error(f"Error during sync TensorFlow metadata extraction for {path_str}: {e_tf_meta}")
                _metadata_update['load_error'] = str(e_tf_meta)
            return _metadata_update

        try:
            # Pass an empty dict or relevant part of metadata if sync_tf_meta_extraction needs it
            extracted_meta = await asyncio.to_thread(sync_tf_meta_extraction, file_path, file_ext, {})
            metadata.update(extracted_meta)
        except Exception as e:
            logger.error(f"Error extracting TensorFlow metadata for {file_path}: {e}", exc_info=True)
            metadata['error'] = f"Outer error extracting TensorFlow metadata: {str(e)}"
        return metadata

    @staticmethod
    async def _get_pytorch_metadata_async(file_path: str) -> Dict[str, Any]:
        """Extract PyTorch-specific metadata asynchronously"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping PyTorch metadata extraction.")
            return {'pytorch_available': False}
        
        metadata = {'pytorch_available': True}

        def sync_pytorch_meta_extraction(path_str):
            _metadata_update = {}
            try:
                # Attempt to load as a full model or state_dict first
                # torch.load is I/O and CPU bound
                loaded_object = torch.load(path_str, map_location='cpu')
                
                if hasattr(loaded_object, 'state_dict') and callable(loaded_object.state_dict):
                    # It's a model object
                    state_dict = loaded_object.state_dict()
                    _metadata_update.update({
                        'model_type': 'pytorch_model',
                        'parameter_count': sum(p.numel() for p in loaded_object.parameters()),
                        'trainable_params': sum(p.numel() for p in loaded_object.parameters() if p.requires_grad),
                        'layer_count': len(list(loaded_object.modules())) -1, # Exclude the model itself
                        'model_class': loaded_object.__class__.__name__
                    })
                elif isinstance(loaded_object, dict):
                    # It's a state dict
                    state_dict = loaded_object
                    _metadata_update.update({
                        'model_type': 'pytorch_state_dict',
                        'parameter_count': sum(tensor.numel() for tensor in state_dict.values() if hasattr(tensor, 'numel')),
                        'keys_count': len(state_dict.keys()),
                        'layer_keys_sample': list(state_dict.keys())[:5] # Sample of keys
                    })
                else:
                    _metadata_update.update({
                        'model_type': 'pytorch_other',
                        'object_type': str(type(loaded_object))
                    })

            except Exception as e_torch_load:
                logger.error(f"Error during sync PyTorch metadata extraction for {path_str}: {e_torch_load}")
                _metadata_update['load_error'] = str(e_torch_load)
            return _metadata_update

        try:
            extracted_meta = await asyncio.to_thread(sync_pytorch_meta_extraction, file_path)
            metadata.update(extracted_meta)
        except Exception as e:
            logger.error(f"Error extracting PyTorch metadata for {file_path}: {e}", exc_info=True)
            metadata['error'] = f"Outer error extracting PyTorch metadata: {str(e)}"
        return metadata


class ModelFileManager:
    """File management utilities for model storage"""
    
    @staticmethod
    async def get_model_storage_path_async(model_id: str, filename: str) -> str:
        """Get the storage path for a model file asynchronously"""
        # Create subdirectories based on model_id to avoid too many files in one directory
        subdir = model_id[:2]  # Use first 2 characters for subdirectory
        storage_dir = os.path.join(settings.MODELS_DIR, subdir)
        await aios.makedirs(storage_dir, exist_ok=True)
        
        # Add model_id prefix to filename to ensure uniqueness
        safe_filename = f"{model_id}_{Path(filename).name}" # Ensure filename is just the name part
        return os.path.join(storage_dir, safe_filename)
    
    @staticmethod
    async def save_uploaded_file_async(file_content: bytes, model_id: str, filename: str) -> str:
        """Save uploaded file content to storage asynchronously"""
        try:
            storage_path = await ModelFileManager.get_model_storage_path_async(model_id, filename)
            
            async with aiofiles.open(storage_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Model file saved to: {storage_path}")
            return storage_path
        except Exception as e:
            logger.error(f"Error saving model file: {e}")
            raise
    
    @staticmethod
    async def save_directory_model_async(source_dir: str, model_id: str, dirname: str) -> str:
        """Save directory-based model (like SavedModel) to storage asynchronously"""
        try:
            import shutil # Keep shutil import local to the threaded function or ensure it's at top-level if used elsewhere sync
            storage_path = await ModelFileManager.get_model_storage_path_async(model_id, dirname)
            
            # Remove if exists and copy
            if await aios.path.exists(storage_path):
                await asyncio.to_thread(shutil.rmtree, storage_path)
            
            await asyncio.to_thread(shutil.copytree, source_dir, storage_path)
            logger.info(f"Model directory saved to: {storage_path}")
            return storage_path
        except Exception as e:
            logger.error(f"Error saving model directory: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def delete_model_file_async(file_path: str) -> bool:
        """Delete a model file or directory asynchronously"""
        try:
            if await aios.path.exists(file_path):
                if await aios.path.isdir(file_path):
                    import shutil # Keep import local if only used in thread
                    await asyncio.to_thread(shutil.rmtree, file_path)
                else:
                    await aios.remove(file_path)
                logger.info(f"Model file/directory deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting model file {file_path}: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def get_file_info_async(file_path: str) -> Dict[str, Any]:
        """Get file information asynchronously"""
        try:
            if not await aios.path.exists(file_path):
                return {'exists': False}
            
            if await aios.path.isdir(file_path):
                # Calculate directory size asynchronously
                total_size = 0
                async for dirpath, dirnames, filenames in aios.walk(file_path):
                    for filename in filenames:
                        try:
                            fp = os.path.join(dirpath, filename) # os.path.join is fine
                            total_size += (await aios.stat(fp)).st_size
                        except OSError: # Skip files that can't be stat'd (e.g. broken symlinks)
                            pass 
                stat_result = await aios.stat(file_path)
                return {
                    'exists': True,
                    'size': total_size,
                    'created_at': datetime.fromtimestamp(stat_result.st_ctime),
                    'updated_at': datetime.fromtimestamp(stat_result.st_mtime),
                    'is_directory': True
                }
            else:
                stat_result = await aios.stat(file_path)
                return {
                    'exists': True,
                    'size': stat_result.st_size,
                    'created_at': datetime.fromtimestamp(stat_result.st_ctime),
                    'updated_at': datetime.fromtimestamp(stat_result.st_mtime),
                    'is_directory': False
                }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}", exc_info=True)
            return {
                'exists': False,
                'error': str(e)
            } 