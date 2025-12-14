"""
Scikit-learn framework detection and metadata extraction
"""

import pickle
import logging
import aiofiles
import asyncio
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

# Optional import for sklearn
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None
    SKLEARN_AVAILABLE = False


class SklearnDetector:
    """Scikit-learn specific detection and metadata extraction"""
    
    @staticmethod
    async def detect_model_type_async(file_path: str) -> ModelType:
        """Detect sklearn model type asynchronously"""
        try:
            model = None
            if JOBLIB_AVAILABLE:
                try:
                    model = await asyncio.to_thread(joblib.load, file_path)
                except Exception as e_joblib:
                    logger.debug(f"joblib.load failed for {file_path} in _detect_sklearn_model_type_async: {e_joblib}. Falling back to pickle.")
                    try:
                        async with aiofiles.open(file_path, 'rb') as f:
                            content = await f.read()
                        model = await asyncio.to_thread(pickle.loads, content)
                    except Exception as e_pickle:
                        logger.error(f"Error loading sklearn model from pickle {file_path}: {e_pickle}")
                        return ModelType.OTHER
            else:
                try:
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    model = await asyncio.to_thread(pickle.loads, content)
                except Exception as e_pickle:
                    logger.error(f"Error loading sklearn model from pickle {file_path}: {e_pickle}")
                    return ModelType.OTHER

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
            
            # Other types
            else:
                if hasattr(model, 'predict_proba'):
                    return ModelType.CLASSIFICATION
                elif hasattr(model, 'predict'):
                    return ModelType.REGRESSION
                elif hasattr(model, 'fit_transform') or hasattr(model, 'transform'):
                    return ModelType.OTHER
                
                logger.info(f"Could not determine specific sklearn model type for {model_name}, defaulting to OTHER.")
                return ModelType.OTHER
            
        except Exception as e:
            logger.error(f"Error detecting sklearn model type for {file_path}: {e}", exc_info=True)
            return ModelType.OTHER
    
    @staticmethod
    async def get_metadata_async(file_path: str) -> Dict[str, Any]:
        """Get metadata from sklearn model asynchronously"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn is not available. Skipping sklearn metadata extraction.")
            return {}
        
        metadata = {}
        try:
            def sync_load_and_meta(path):
                _metadata = {}
                model = None
                if JOBLIB_AVAILABLE:
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
                else:
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

