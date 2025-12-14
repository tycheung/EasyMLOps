"""
XGBoost framework detection and metadata extraction
"""

import pickle
import logging
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

# Optional import for xgboost
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    xgboost = None
    XGBOOST_AVAILABLE = False
except Exception as e:
    xgboost = None
    XGBOOST_AVAILABLE = False


class XGBoostDetector:
    """XGBoost specific detection and metadata extraction"""
    
    @staticmethod
    async def detect_model_type_async(file_path: str) -> ModelType:
        """Detect XGBoost model type asynchronously"""
        try:
            model = None
            if JOBLIB_AVAILABLE:
                try:
                    model = await asyncio.to_thread(joblib.load, file_path)
                except Exception as e_joblib:
                    logger.debug(f"joblib.load failed for {file_path} in _detect_boosting_model_type_async: {e_joblib}. Falling back to pickle.")
                    try:
                        import aiofiles
                        async with aiofiles.open(file_path, 'rb') as f:
                            content = await f.read()
                        model = await asyncio.to_thread(pickle.loads, content)
                    except Exception as e_pickle:
                        logger.error(f"Error loading boosting model from pickle {file_path}: {e_pickle}")
                        return ModelType.OTHER
            else:
                try:
                    import aiofiles
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    model = await asyncio.to_thread(pickle.loads, content)
                except Exception as e_pickle:
                    logger.error(f"Error loading boosting model from pickle {file_path}: {e_pickle}")
                    return ModelType.OTHER

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
    async def get_metadata_async(file_path: str) -> Dict[str, Any]:
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
                    except:
                        pass 
                if hasattr(model, 'n_features_in_'):
                     _metadata['n_features_in'] = model.n_features_in_
                if hasattr(model, 'booster') and hasattr(model.booster, 'attributes'):
                     _metadata['booster_attributes'] = model.booster.attributes()
                return _metadata

            metadata = await asyncio.to_thread(sync_load_and_meta, file_path)
        except Exception as e:
            logger.error(f"Error getting XGBoost metadata for {file_path}: {e}", exc_info=True)
            metadata['error'] = f"Error extracting XGBoost metadata: {str(e)}"
        return metadata

