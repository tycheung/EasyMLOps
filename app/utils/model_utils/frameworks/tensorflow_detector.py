"""
TensorFlow framework detection and metadata extraction
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict

from app.schemas.model import ModelType

logger = logging.getLogger(__name__)

# Optional import for tensorflow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    logger.info(f"TensorFlow not available due to dependency issue: {e}")


class TensorFlowDetector:
    """TensorFlow specific detection and metadata extraction"""
    
    @staticmethod
    def _is_savedmodel_directory_sync(dir_path: str) -> bool:
        """Synchronous version for internal TF type detection use only"""
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
                output_layer = model.layers[-1]
                output_shape = output_layer.output_shape
                
                if hasattr(output_layer, 'activation'):
                    activation = str(output_layer.activation).lower()
                    
                    if 'softmax' in activation or 'sigmoid' in activation:
                        return ModelType.CLASSIFICATION
                    elif 'linear' in activation or activation == 'none':
                        return ModelType.REGRESSION
                
                if isinstance(output_shape, (list, tuple)) and len(output_shape) > 1:
                    output_size = output_shape[-1]
                    if output_size == 1:
                        return ModelType.REGRESSION
                    elif output_size > 1:
                        return ModelType.CLASSIFICATION
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error analyzing Keras model: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _analyze_savedmodel(model) -> ModelType:
        """Analyze SavedModel to determine type"""
        try:
            if hasattr(model, 'signatures'):
                for signature_key, signature in model.signatures.items():
                    if hasattr(signature, 'outputs'):
                        for output_key, output_spec in signature.outputs.items():
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
    async def detect_model_type_async(file_path: str) -> ModelType:
        """Detect TensorFlow model type asynchronously"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping TensorFlow model type detection.")
            return ModelType.OTHER

        try:
            file_ext = Path(file_path).suffix.lower()
            model_info = {"type": ModelType.OTHER}

            def sync_tf_load_and_analyze(path_str, ext_str, info_dict):
                model = None
                if ext_str == '.h5':
                    model = tf.keras.models.load_model(path_str)
                    info_dict["type"] = TensorFlowDetector._analyze_keras_model(model)
                elif ext_str == '.pb':
                    logger.info(f"Standalone .pb file {path_str} detected, type detection might be limited.")
                    parent_dir = Path(path_str).parent
                    if TensorFlowDetector._is_savedmodel_directory_sync(str(parent_dir)):
                         model = tf.saved_model.load(str(parent_dir))
                         info_dict["type"] = TensorFlowDetector._analyze_savedmodel(model)
                    else:
                        info_dict["type"] = ModelType.OTHER
                elif ext_str == '.tflite':
                    interpreter = tf.lite.Interpreter(model_path=path_str)
                    info_dict["type"] = TensorFlowDetector._analyze_tflite_model(interpreter)
                elif os.path.isdir(path_str) and TensorFlowDetector._is_savedmodel_directory_sync(path_str):
                    model = tf.saved_model.load(path_str)
                    info_dict["type"] = TensorFlowDetector._analyze_savedmodel(model)
                else:
                    logger.warning(f"Unsupported TensorFlow file/directory for type detection: {path_str}")
                    info_dict["type"] = ModelType.OTHER

            await asyncio.to_thread(sync_tf_load_and_analyze, file_path, file_ext, model_info)
            return model_info["type"]

        except Exception as e:
            logger.error(f"Error detecting TensorFlow model type for {file_path}: {e}", exc_info=True)
            return ModelType.OTHER
    
    @staticmethod
    async def get_metadata_async(file_path: str) -> Dict[str, Any]:
        """Extract TensorFlow-specific metadata asynchronously"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping TensorFlow metadata extraction.")
            return {'tensorflow_available': False}
        
        metadata = {'tensorflow_available': True}
        file_ext = Path(file_path).suffix.lower()

        def sync_tf_meta_extraction(path_str, ext_str, current_meta_unused):
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
                
                elif os.path.isdir(path_str) or (ext_str == '.pb' and TensorFlowDetector._is_savedmodel_directory_sync(path_str)):
                    target_load_path = path_str
                    if ext_str == '.pb' and not os.path.isdir(path_str):
                        parent_dir = str(Path(path_str).parent)
                        if TensorFlowDetector._is_savedmodel_directory_sync(parent_dir):
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
            extracted_meta = await asyncio.to_thread(sync_tf_meta_extraction, file_path, file_ext, {})
            metadata.update(extracted_meta)
        except Exception as e:
            logger.error(f"Error extracting TensorFlow metadata for {file_path}: {e}", exc_info=True)
            metadata['error'] = f"Outer error extracting TensorFlow metadata: {str(e)}"
        return metadata

