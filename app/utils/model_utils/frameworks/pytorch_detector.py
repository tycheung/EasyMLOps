"""
PyTorch framework detection and metadata extraction
"""

import pickle
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict

from app.schemas.model import ModelType

logger = logging.getLogger(__name__)

# Optional import for joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

# Optional import for torch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False


class PyTorchDetector:
    """PyTorch specific detection and metadata extraction"""
    
    @staticmethod
    def _analyze_pytorch_model(model) -> ModelType:
        """Analyze PyTorch model to determine type"""
        try:
            if hasattr(model, 'modules'):
                modules = list(model.modules())
                
                for module in reversed(modules):
                    module_name = module.__class__.__name__.lower()
                    
                    if any(word in module_name for word in ['softmax', 'logsoftmax', 'sigmoid']):
                        return ModelType.CLASSIFICATION
                    
                    if 'linear' in module_name and hasattr(module, 'out_features'):
                        out_features = module.out_features
                        if out_features == 1:
                            return ModelType.REGRESSION
                        elif out_features > 1:
                            return ModelType.CLASSIFICATION
            
            elif isinstance(model, dict):
                return PyTorchDetector._analyze_pytorch_state_dict(model)
            
            return ModelType.OTHER
        except Exception as e:
            logger.error(f"Error analyzing PyTorch model: {e}")
            return ModelType.OTHER
    
    @staticmethod
    def _analyze_pytorch_state_dict(state_dict) -> ModelType:
        """Analyze PyTorch state dict to determine type"""
        try:
            if isinstance(state_dict, dict):
                for key in state_dict.keys():
                    if key.endswith(('.weight', '.bias')) and 'fc' in key.lower():
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
    
    @staticmethod
    async def detect_model_type_async(file_path: str) -> ModelType:
        """Detect PyTorch model type asynchronously"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping PyTorch model type detection.")
            return ModelType.OTHER

        try:
            model_info = {"type": ModelType.OTHER}

            def sync_pytorch_load_and_analyze(path_str, info_dict):
                try:
                    model = torch.load(path_str, map_location='cpu') 
                    if isinstance(model, dict):
                        info_dict["type"] = PyTorchDetector._analyze_pytorch_state_dict(model)
                    else:
                        info_dict["type"] = PyTorchDetector._analyze_pytorch_model(model)
                    return
                except Exception as e_load:
                    logger.debug(f"Failed to load PyTorch model/state_dict directly from {path_str}: {e_load}")
                
                file_ext = Path(path_str).suffix.lower()
                if file_ext in ['.pkl', '.joblib']:
                    if JOBLIB_AVAILABLE:
                        try:
                            model = joblib.load(path_str)
                        except Exception as e_joblib:
                            logger.debug(f"joblib.load failed for PyTorch pkl {path_str}: {e_joblib}. Trying pickle.")
                            try:
                                with open(path_str, 'rb') as f_pkl:
                                    model = pickle.load(f_pkl)
                            except Exception as e_pickle:
                                logger.error(f"Failed to load PyTorch model from pkl/joblib {path_str} with pickle: {e_pickle}")
                                info_dict["type"] = ModelType.OTHER
                                return
                    else:
                        try:
                            with open(path_str, 'rb') as f_pkl:
                                model = pickle.load(f_pkl)
                        except Exception as e_pickle:
                            logger.error(f"Failed to load PyTorch model from pkl {path_str} with pickle: {e_pickle}")
                            info_dict["type"] = ModelType.OTHER
                            return
                    
                    if isinstance(model, dict):
                        info_dict["type"] = PyTorchDetector._analyze_pytorch_state_dict(model)
                    else:
                        info_dict["type"] = PyTorchDetector._analyze_pytorch_model(model)
                    return
                
                info_dict["type"] = ModelType.OTHER
            
            await asyncio.to_thread(sync_pytorch_load_and_analyze, file_path, model_info)
            return model_info["type"]

        except Exception as e:
            logger.error(f"Error detecting PyTorch model type for {file_path}: {e}", exc_info=True)
            return ModelType.OTHER
    
    @staticmethod
    async def get_metadata_async(file_path: str) -> Dict[str, Any]:
        """Extract PyTorch-specific metadata asynchronously"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping PyTorch metadata extraction.")
            return {'pytorch_available': False}
        
        metadata = {'pytorch_available': True}

        def sync_pytorch_meta_extraction(path_str):
            _metadata_update = {}
            try:
                loaded_object = torch.load(path_str, map_location='cpu')
                
                if hasattr(loaded_object, 'state_dict') and callable(loaded_object.state_dict):
                    state_dict = loaded_object.state_dict()
                    _metadata_update.update({
                        'model_type': 'pytorch_model',
                        'parameter_count': sum(p.numel() for p in loaded_object.parameters()),
                        'trainable_params': sum(p.numel() for p in loaded_object.parameters() if p.requires_grad),
                        'layer_count': len(list(loaded_object.modules())) -1,
                        'model_class': loaded_object.__class__.__name__
                    })
                elif isinstance(loaded_object, dict):
                    state_dict = loaded_object
                    _metadata_update.update({
                        'model_type': 'pytorch_state_dict',
                        'parameter_count': sum(tensor.numel() for tensor in state_dict.values() if hasattr(tensor, 'numel')),
                        'keys_count': len(state_dict.keys()),
                        'layer_keys_sample': list(state_dict.keys())[:5]
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

