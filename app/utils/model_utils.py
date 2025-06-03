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

from app.schemas.model import ModelFramework, ModelType, ModelValidationResult
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Optional imports for TensorFlow and PyTorch
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.info("TensorFlow not available - limited TensorFlow model support")

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.info("PyTorch not available - limited PyTorch model support")


class ModelValidator:
    """Comprehensive model validation for multiple ML frameworks"""
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
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
    def _is_savedmodel_directory(dir_path: str) -> bool:
        """Check if directory is a TensorFlow SavedModel"""
        if not os.path.isdir(dir_path):
            return False
        
        # Check for required SavedModel files
        saved_model_pb = os.path.join(dir_path, "saved_model.pb")
        variables_dir = os.path.join(dir_path, "variables")
        
        return os.path.exists(saved_model_pb) and os.path.isdir(variables_dir)
    
    @classmethod
    def detect_framework_from_file(cls, file_path: str) -> Optional[ModelFramework]:
        """Detect ML framework from model file"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Handle different file types
            if file_ext in ['.pkl', '.joblib']:
                return cls._detect_pickle_framework(file_path)
            elif file_ext in ['.h5', '.pb', '.tflite']:
                return ModelFramework.TENSORFLOW
            elif file_ext in ['.pt', '.pth']:
                return ModelFramework.PYTORCH
            elif file_ext == '.onnx':
                return ModelFramework.ONNX
            elif file_ext == '.json':
                return cls._detect_json_framework(file_path)
            elif file_ext == '.zip':
                return cls._detect_zip_framework(file_path)
            
            # Check if it's a directory (for SavedModel)
            if os.path.isdir(file_path) and cls._is_savedmodel_directory(file_path):
                return ModelFramework.TENSORFLOW
            
            return None
        except Exception as e:
            logger.error(f"Error detecting framework: {e}")
            return None
    
    @staticmethod
    def _detect_pickle_framework(file_path: str) -> Optional[ModelFramework]:
        """Detect framework for pickle/joblib files"""
        try:
            # Try joblib first
            try:
                model = joblib.load(file_path)
            except:
                # Fall back to pickle
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            model_type_str = str(type(model))
            model_module = getattr(model, '__module__', '')
            
            # Check for different frameworks
            if 'sklearn' in model_type_str or 'sklearn' in model_module:
                return ModelFramework.SKLEARN
            elif 'xgboost' in model_type_str or 'xgb' in model_module:
                return ModelFramework.XGBOOST
            elif 'lightgbm' in model_type_str or 'lgb' in model_module:
                return ModelFramework.LIGHTGBM
            elif 'h2o' in model_type_str or 'h2o' in model_module:
                return ModelFramework.H2O
            elif 'torch' in model_type_str or 'pytorch' in model_module:
                return ModelFramework.PYTORCH
            elif 'tensorflow' in model_type_str or 'keras' in model_type_str:
                return ModelFramework.TENSORFLOW
            
            return ModelFramework.CUSTOM
        except Exception as e:
            logger.error(f"Error detecting pickle framework: {e}")
            return None
    
    @staticmethod
    def _detect_json_framework(file_path: str) -> Optional[ModelFramework]:
        """Detect framework for JSON files"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check for framework indicators in JSON
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
        except Exception as e:
            logger.error(f"Error detecting JSON framework: {e}")
            return None
    
    @staticmethod
    def _detect_zip_framework(file_path: str) -> Optional[ModelFramework]:
        """Detect framework for ZIP files (common for SavedModel)"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                # Check for TensorFlow SavedModel structure
                if any('saved_model.pb' in f for f in file_list):
                    return ModelFramework.TENSORFLOW
                
                # Check for PyTorch files
                if any(f.endswith(('.pt', '.pth')) for f in file_list):
                    return ModelFramework.PYTORCH
                
                # Check for other frameworks
                if any('model.json' in f or 'config.json' in f for f in file_list):
                    return ModelFramework.CUSTOM
            
            return ModelFramework.CUSTOM
        except Exception as e:
            logger.error(f"Error detecting ZIP framework: {e}")
            return None
    
    @classmethod
    def detect_model_type(cls, file_path: str, framework: ModelFramework) -> Optional[ModelType]:
        """Detect model type based on the loaded model"""
        try:
            if framework == ModelFramework.SKLEARN:
                return cls._detect_sklearn_model_type(file_path)
            elif framework in [ModelFramework.XGBOOST, ModelFramework.LIGHTGBM]:
                return cls._detect_boosting_model_type(file_path)
            elif framework == ModelFramework.TENSORFLOW:
                return cls._detect_tensorflow_model_type(file_path)
            elif framework == ModelFramework.PYTORCH:
                return cls._detect_pytorch_model_type(file_path)
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error detecting model type: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _detect_sklearn_model_type(file_path: str) -> ModelType:
        """Detect sklearn model type"""
        try:
            try:
                model = joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            model_name = model.__class__.__name__.lower()
            
            # Classification models
            if any(word in model_name for word in ['classifier', 'svc', 'naive', 'logistic']):
                return ModelType.CLASSIFICATION
            
            # Regression models
            elif any(word in model_name for word in ['regressor', 'svr', 'linear']):
                return ModelType.REGRESSION
            
            # Clustering models
            elif any(word in model_name for word in ['cluster', 'kmeans', 'dbscan']):
                return ModelType.CLUSTERING
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error detecting sklearn model type: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _detect_boosting_model_type(file_path: str) -> ModelType:
        """Detect boosting model type (XGBoost, LightGBM)"""
        try:
            try:
                model = joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Check objective or model attributes
            if hasattr(model, 'objective'):
                objective = getattr(model, 'objective', '').lower()
                if 'class' in objective or 'binary' in objective:
                    return ModelType.CLASSIFICATION
                elif 'reg' in objective:
                    return ModelType.REGRESSION
            
            # Check for common attributes
            model_str = str(model).lower()
            if 'class' in model_str:
                return ModelType.CLASSIFICATION
            elif 'reg' in model_str:
                return ModelType.REGRESSION
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error detecting boosting model type: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _detect_tensorflow_model_type(file_path: str) -> ModelType:
        """Detect TensorFlow model type"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available for model type detection")
                return ModelType.OTHER
            
            file_ext = Path(file_path).suffix.lower()
            
            # Handle different TensorFlow formats
            if file_ext == '.h5':
                model = tf.keras.models.load_model(file_path)
                return ModelValidator._analyze_keras_model(model)
            
            elif file_ext == '.pb':
                # For .pb files, try to load as SavedModel or frozen graph
                try:
                    model = tf.saved_model.load(file_path)
                    return ModelValidator._analyze_savedmodel(model)
                except:
                    # Fallback for frozen graphs
                    return ModelType.OTHER
            
            elif file_ext == '.tflite':
                # TensorFlow Lite models - analyze structure
                interpreter = tf.lite.Interpreter(model_path=file_path)
                interpreter.allocate_tensors()
                return ModelValidator._analyze_tflite_model(interpreter)
            
            elif os.path.isdir(file_path):
                # SavedModel directory
                model = tf.saved_model.load(file_path)
                return ModelValidator._analyze_savedmodel(model)
            
            elif file_ext in ['.pkl', '.joblib']:
                # Pickled TensorFlow/Keras model
                try:
                    model = joblib.load(file_path)
                except:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                
                if hasattr(model, 'layers'):  # Keras model
                    return ModelValidator._analyze_keras_model(model)
            
            return ModelType.OTHER
            
        except Exception as e:
            logger.error(f"Error detecting TensorFlow model type: {e}")
            return ModelType.OTHER
    
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
    def _detect_pytorch_model_type(file_path: str) -> ModelType:
        """Detect PyTorch model type"""
        try:
            if not PYTORCH_AVAILABLE:
                logger.warning("PyTorch not available for model type detection")
                return ModelType.OTHER
            
            file_ext = Path(file_path).suffix.lower()
            
            # Handle different PyTorch formats
            if file_ext in ['.pt', '.pth']:
                # Load PyTorch model
                try:
                    model = torch.load(file_path, map_location='cpu')
                    return ModelValidator._analyze_pytorch_model(model)
                except:
                    # Try loading as state dict
                    state_dict = torch.load(file_path, map_location='cpu')
                    return ModelValidator._analyze_pytorch_state_dict(state_dict)
            
            elif file_ext in ['.pkl', '.joblib']:
                # Pickled PyTorch model
                try:
                    model = joblib.load(file_path)
                except:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                
                if hasattr(model, 'state_dict') or str(type(model)).startswith('<class \'torch.'):
                    return ModelValidator._analyze_pytorch_model(model)
            
            return ModelType.OTHER
            
        except Exception as e:
            logger.error(f"Error detecting PyTorch model type: {e}")
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
    def validate_model_file(cls, file_path: str) -> ModelValidationResult:
        """Comprehensive model validation"""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                errors.append("Model file does not exist")
                return ModelValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    metadata=metadata
                )
            
            # Get file info
            if os.path.isdir(file_path):
                # Handle directory models (SavedModel)
                file_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, dirnames, filenames in os.walk(file_path)
                               for filename in filenames)
                filename = os.path.basename(file_path)
            else:
                file_size = os.path.getsize(file_path)
                filename = os.path.basename(file_path)
            
            # Validate file extension (skip for directories)
            if not os.path.isdir(file_path) and not cls.validate_file_extension(filename):
                errors.append(f"Unsupported file extension. Allowed: {settings.ALLOWED_MODEL_EXTENSIONS}")
            
            # Validate file size
            if not cls.validate_file_size(file_size):
                errors.append(f"File size ({file_size} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)")
            
            # Add file size warning if large
            if file_size > 100 * 1024 * 1024:  # 100MB
                warnings.append("Large model file size may impact loading performance")
            
            # Detect framework
            framework_detected = cls.detect_framework_from_file(file_path)
            if framework_detected:
                metadata['framework_detected'] = framework_detected.value
                
                # Detect model type
                model_type_detected = cls.detect_model_type(file_path, framework_detected)
                if model_type_detected:
                    metadata['model_type_detected'] = model_type_detected.value
            else:
                warnings.append("Could not detect ML framework automatically")
            
            # Try to load and get additional metadata
            try:
                if framework_detected == ModelFramework.SKLEARN:
                    metadata.update(cls._get_sklearn_metadata(file_path))
                elif framework_detected == ModelFramework.XGBOOST:
                    metadata.update(cls._get_xgboost_metadata(file_path))
                elif framework_detected == ModelFramework.LIGHTGBM:
                    metadata.update(cls._get_lightgbm_metadata(file_path))
                elif framework_detected == ModelFramework.TENSORFLOW:
                    metadata.update(cls._get_tensorflow_metadata(file_path))
                elif framework_detected == ModelFramework.PYTORCH:
                    metadata.update(cls._get_pytorch_metadata(file_path))
            except Exception as e:
                warnings.append(f"Could not extract detailed metadata: {str(e)}")
            
            # Add basic file metadata
            if os.path.isdir(file_path):
                metadata.update({
                    'file_size': file_size,
                    'file_type': 'directory',
                    'file_hash': 'directory_hash_not_supported'
                })
            else:
                metadata.update({
                    'file_size': file_size,
                    'file_extension': Path(filename).suffix.lower(),
                    'file_hash': cls.calculate_file_hash(file_path)
                })
            
            is_valid = len(errors) == 0
            
            return ModelValidationResult(
                is_valid=is_valid,
                framework_detected=framework_detected,
                model_type_detected=model_type_detected if framework_detected else None,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            errors.append(f"Validation failed: {str(e)}")
            
            return ModelValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
    
    @staticmethod
    def _get_sklearn_metadata(file_path: str) -> Dict[str, Any]:
        """Extract sklearn-specific metadata"""
        try:
            try:
                model = joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            metadata = {
                'model_class': model.__class__.__name__,
                'model_module': model.__module__
            }
            
            # Try to get feature count
            if hasattr(model, 'n_features_in_'):
                metadata['feature_count'] = model.n_features_in_
            elif hasattr(model, 'coef_'):
                if hasattr(model.coef_, 'shape'):
                    metadata['feature_count'] = model.coef_.shape[-1]
            
            # Try to get class count for classifiers
            if hasattr(model, 'classes_'):
                metadata['class_count'] = len(model.classes_)
                metadata['classes'] = model.classes_.tolist() if hasattr(model.classes_, 'tolist') else list(model.classes_)
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting sklearn metadata: {e}")
            return {}
    
    @staticmethod
    def _get_xgboost_metadata(file_path: str) -> Dict[str, Any]:
        """Extract XGBoost-specific metadata"""
        try:
            try:
                model = joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            metadata = {
                'model_class': model.__class__.__name__
            }
            
            if hasattr(model, 'get_params'):
                params = model.get_params()
                metadata['n_estimators'] = params.get('n_estimators')
                metadata['objective'] = params.get('objective')
                metadata['max_depth'] = params.get('max_depth')
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting XGBoost metadata: {e}")
            return {}
    
    @staticmethod
    def _get_lightgbm_metadata(file_path: str) -> Dict[str, Any]:
        """Extract LightGBM-specific metadata"""
        try:
            try:
                model = joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            metadata = {
                'model_class': model.__class__.__name__
            }
            
            if hasattr(model, 'get_params'):
                params = model.get_params()
                metadata['n_estimators'] = params.get('n_estimators')
                metadata['objective'] = params.get('objective')
                metadata['num_leaves'] = params.get('num_leaves')
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting LightGBM metadata: {e}")
            return {}
    
    @staticmethod
    def _get_tensorflow_metadata(file_path: str) -> Dict[str, Any]:
        """Extract TensorFlow-specific metadata"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return {'tensorflow_available': False}
            
            metadata = {'tensorflow_available': True}
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.h5':
                model = tf.keras.models.load_model(file_path)
                metadata.update({
                    'model_type': 'keras_h5',
                    'layer_count': len(model.layers),
                    'trainable_params': model.count_params(),
                    'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else None,
                    'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else None
                })
                
                if hasattr(model, 'optimizer'):
                    metadata['optimizer'] = model.optimizer.__class__.__name__
            
            elif file_ext == '.tflite':
                interpreter = tf.lite.Interpreter(model_path=file_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                metadata.update({
                    'model_type': 'tflite',
                    'input_count': len(input_details),
                    'output_count': len(output_details),
                    'input_shapes': [detail['shape'].tolist() for detail in input_details],
                    'output_shapes': [detail['shape'].tolist() for detail in output_details]
                })
            
            elif os.path.isdir(file_path) or file_ext == '.pb':
                try:
                    model = tf.saved_model.load(file_path)
                    metadata.update({
                        'model_type': 'savedmodel',
                        'signature_keys': list(model.signatures.keys()) if hasattr(model, 'signatures') else []
                    })
                except Exception as e:
                    metadata['load_error'] = str(e)
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting TensorFlow metadata: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _get_pytorch_metadata(file_path: str) -> Dict[str, Any]:
        """Extract PyTorch-specific metadata"""
        try:
            if not PYTORCH_AVAILABLE:
                return {'pytorch_available': False}
            
            metadata = {'pytorch_available': True}
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.pt', '.pth']:
                try:
                    model = torch.load(file_path, map_location='cpu')
                    
                    if hasattr(model, 'state_dict'):
                        # It's a model object
                        state_dict = model.state_dict()
                        metadata.update({
                            'model_type': 'pytorch_model',
                            'parameter_count': sum(p.numel() for p in model.parameters()),
                            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                            'layer_count': len(list(model.modules())) - 1,  # Exclude the model itself
                            'model_class': model.__class__.__name__
                        })
                    elif isinstance(model, dict):
                        # It's a state dict
                        metadata.update({
                            'model_type': 'pytorch_state_dict',
                            'parameter_count': sum(tensor.numel() for tensor in model.values()),
                            'keys_count': len(model.keys()),
                            'layer_keys': list(model.keys())[:10]  # First 10 keys as sample
                        })
                    else:
                        metadata.update({
                            'model_type': 'pytorch_other',
                            'object_type': str(type(model))
                        })
                
                except Exception as e:
                    # Try loading as state dict
                    state_dict = torch.load(file_path, map_location='cpu')
                    if isinstance(state_dict, dict):
                        metadata.update({
                            'model_type': 'pytorch_state_dict',
                            'parameter_count': sum(tensor.numel() for tensor in state_dict.values() if hasattr(tensor, 'numel')),
                            'keys_count': len(state_dict.keys()),
                            'load_error': str(e)
                        })
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting PyTorch metadata: {e}")
            return {'error': str(e)}


class ModelFileManager:
    """File management utilities for model storage"""
    
    @staticmethod
    def get_model_storage_path(model_id: str, filename: str) -> str:
        """Get the storage path for a model file"""
        # Create subdirectories based on model_id to avoid too many files in one directory
        subdir = model_id[:2]  # Use first 2 characters for subdirectory
        storage_dir = os.path.join(settings.MODELS_DIR, subdir)
        os.makedirs(storage_dir, exist_ok=True)
        
        # Add model_id prefix to filename to ensure uniqueness
        safe_filename = f"{model_id}_{filename}"
        return os.path.join(storage_dir, safe_filename)
    
    @staticmethod
    def save_uploaded_file(file_content: bytes, model_id: str, filename: str) -> str:
        """Save uploaded file content to storage"""
        try:
            storage_path = ModelFileManager.get_model_storage_path(model_id, filename)
            
            with open(storage_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Model file saved to: {storage_path}")
            return storage_path
        except Exception as e:
            logger.error(f"Error saving model file: {e}")
            raise
    
    @staticmethod
    def save_directory_model(source_dir: str, model_id: str, dirname: str) -> str:
        """Save directory-based model (like SavedModel) to storage"""
        try:
            import shutil
            storage_path = ModelFileManager.get_model_storage_path(model_id, dirname)
            
            # Remove if exists and copy
            if os.path.exists(storage_path):
                shutil.rmtree(storage_path)
            
            shutil.copytree(source_dir, storage_path)
            logger.info(f"Model directory saved to: {storage_path}")
            return storage_path
        except Exception as e:
            logger.error(f"Error saving model directory: {e}")
            raise
    
    @staticmethod
    def delete_model_file(file_path: str) -> bool:
        """Delete a model file or directory"""
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                logger.info(f"Model file/directory deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting model file: {e}")
            return False
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            if not os.path.exists(file_path):
                return {}
            
            if os.path.isdir(file_path):
                # Calculate directory size
                total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, dirnames, filenames in os.walk(file_path)
                               for filename in filenames)
                stat = os.stat(file_path)
                return {
                    'size': total_size,
                    'created': stat.st_ctime,
                    'modified': stat.st_mtime,
                    'exists': True,
                    'is_directory': True
                }
            else:
                stat = os.stat(file_path)
                return {
                    'size': stat.st_size,
                    'created': stat.st_ctime,
                    'modified': stat.st_mtime,
                    'exists': True,
                    'is_directory': False
                }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {'exists': False} 