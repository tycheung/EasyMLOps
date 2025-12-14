"""
Core framework detection logic
Handles file type detection and delegates to framework-specific detectors
"""

import os
import pickle
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import aiofiles
import aiofiles.os as aios
import asyncio

# Optional import for joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False
    logging.warning("joblib not available - model loading will use pickle fallback")

from app.schemas.model import ModelFramework, ModelType

logger = logging.getLogger(__name__)

# Import framework-specific detectors
from app.utils.model_utils.frameworks.sklearn_detector import SklearnDetector
from app.utils.model_utils.frameworks.xgboost_detector import XGBoostDetector
from app.utils.model_utils.frameworks.lightgbm_detector import LightGBMDetector
from app.utils.model_utils.frameworks.tensorflow_detector import TensorFlowDetector
from app.utils.model_utils.frameworks.pytorch_detector import PyTorchDetector


class FrameworkDetector:
    """Framework detection, type analysis, and metadata extraction"""
    
    @staticmethod
    async def _is_savedmodel_directory_async(dir_path: str) -> bool:
        """Check if directory is a TensorFlow SavedModel asynchronously"""
        if not await aios.path.isdir(dir_path):
            return False
        
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
                return ModelFramework.TENSORFLOW 
            elif file_ext in ['.pt', '.pth']:
                return ModelFramework.PYTORCH
            elif file_ext == '.onnx':
                return ModelFramework.ONNX
            elif file_ext == '.json':
                return await cls._detect_json_framework_async(file_path)
            elif file_ext == '.zip':
                return await cls._detect_zip_framework_async(file_path)
            
            # Check if it's a directory (for SavedModel)
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
            # Try joblib first if available
            if JOBLIB_AVAILABLE:
                try:
                    model = await asyncio.to_thread(joblib.load, file_path)
                except Exception as e_joblib:
                    logger.debug(f"joblib.load failed for {file_path}: {e_joblib}. Falling back to pickle.")
                    try:
                        async with aiofiles.open(file_path, 'rb') as f:
                            content = await f.read()
                        model = await asyncio.to_thread(pickle.loads, content)
                    except Exception as e_pickle:
                        logger.error(f"Error loading pickle file {file_path} after joblib failed: {e_pickle}")
                        return None
            else:
                # Fallback to pickle if joblib not available
                try:
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    model = await asyncio.to_thread(pickle.loads, content)
                except Exception as e_pickle:
                    logger.error(f"Error loading pickle file {file_path}: {e_pickle}")
                    return None
            
            if model is None:
                logger.warning(f"Could not load model from {file_path} using joblib or pickle.")
                return None

            model_type_str = str(type(model))
            model_module = getattr(model, '__module__', '')
            model_class_name = model.__class__.__name__
            
            # Check for XGBoost
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
            
            # Check for TensorFlow/Keras
            if ('tensorflow' in model_type_str.lower() or 
                'keras' in model_type_str.lower() or
                'tf.' in model_module.lower()):
                return ModelFramework.TENSORFLOW
            
            # Check for PyTorch
            if ('torch' in model_type_str.lower() or 
                'pytorch' in model_module.lower()):
                return ModelFramework.PYTORCH
            
            # Default to sklearn for pickle files
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
            data = await asyncio.to_thread(json.loads, content)
            
            if isinstance(data, dict):
                data_str = str(data).lower()
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
        """Detect framework for ZIP files asynchronously"""
        
        def sync_zip_operations(path):
            try:
                with zipfile.ZipFile(path, 'r') as zip_file:
                    file_list = zip_file.namelist()
                    
                    if any('saved_model.pb' in f for f in file_list):
                        return ModelFramework.TENSORFLOW
                    
                    if any(f.endswith(('.pt', '.pth')) for f in file_list):
                        return ModelFramework.PYTORCH
                    
                    if any('model.json' in f or 'config.json' in f for f in file_list):
                        return ModelFramework.CUSTOM 
                
                return ModelFramework.CUSTOM
            except zipfile.BadZipFile as e:
                logger.error(f"Bad ZIP file {path}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error processing ZIP file {path}: {e}", exc_info=True)
                return None

        try:
            return await asyncio.to_thread(sync_zip_operations, file_path)
        except Exception as e:
            logger.error(f"Error in _detect_zip_framework_async for {file_path}: {e}", exc_info=True)
            return None
    
    @classmethod
    async def detect_model_type_async(cls, file_path: str, framework: ModelFramework) -> Optional[ModelType]:
        """Detect model type based on the loaded model asynchronously"""
        try:
            if framework == ModelFramework.SKLEARN:
                return await SklearnDetector.detect_model_type_async(file_path)
            elif framework == ModelFramework.XGBOOST:
                return await XGBoostDetector.detect_model_type_async(file_path)
            elif framework == ModelFramework.LIGHTGBM:
                return await LightGBMDetector.detect_model_type_async(file_path)
            elif framework == ModelFramework.TENSORFLOW:
                return await TensorFlowDetector.detect_model_type_async(file_path)
            elif framework == ModelFramework.PYTORCH:
                return await PyTorchDetector.detect_model_type_async(file_path)
            
            logger.warning(f"Model type detection not implemented for framework: {framework}")
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error detecting model type for {file_path} (framework: {framework}): {e}", exc_info=True)
            return ModelType.OTHER
    
    @classmethod
    async def get_framework_metadata_async(cls, file_path: str, framework: ModelFramework) -> Dict[str, Any]:
        """Get framework-specific metadata asynchronously"""
        if framework == ModelFramework.SKLEARN:
            return await SklearnDetector.get_metadata_async(file_path)
        elif framework == ModelFramework.XGBOOST:
            return await XGBoostDetector.get_metadata_async(file_path)
        elif framework == ModelFramework.LIGHTGBM:
            return await LightGBMDetector.get_metadata_async(file_path)
        elif framework == ModelFramework.TENSORFLOW:
            return await TensorFlowDetector.get_metadata_async(file_path)
        elif framework == ModelFramework.PYTORCH:
            return await PyTorchDetector.get_metadata_async(file_path)
        else:
            return {}

