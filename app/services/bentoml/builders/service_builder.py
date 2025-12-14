"""
BentoML Service Builder - Facade pattern
Delegates to framework-specific builders
"""

from typing import Any, Dict, Tuple
import logging

from app.models.model import Model
from app.schemas.model import ModelFramework
from app.services.bentoml.builders.sklearn_builder import SklearnBuilder
from app.services.bentoml.builders.tensorflow_builder import TensorFlowBuilder
from app.services.bentoml.builders.pytorch_builder import PyTorchBuilder
from app.services.bentoml.builders.xgboost_builder import XGBoostBuilder
from app.services.bentoml.builders.lightgbm_builder import LightGBMBuilder

logger = logging.getLogger(__name__)


class ServiceBuilder:
    """Framework-specific service builders for BentoML - Facade pattern"""
    
    def __init__(self):
        """Initialize all framework-specific builders"""
        self.sklearn_builder = SklearnBuilder()
        self.tensorflow_builder = TensorFlowBuilder()
        self.pytorch_builder = PyTorchBuilder()
        self.xgboost_builder = XGBoostBuilder()
        self.lightgbm_builder = LightGBMBuilder()
    
    async def create_sklearn_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for sklearn models"""
        return await self.sklearn_builder.create_service(model, service_name, config)
    
    async def create_tensorflow_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for TensorFlow models"""
        return await self.tensorflow_builder.create_service(model, service_name, config)
    
    async def create_pytorch_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for PyTorch models"""
        return await self.pytorch_builder.create_service(model, service_name, config)
    
    async def create_xgboost_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for XGBoost models"""
        return await self.xgboost_builder.create_service(model, service_name, config)
    
    async def create_lightgbm_service(self, model: Model, service_name: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Create BentoML service for LightGBM models"""
        return await self.lightgbm_builder.create_service(model, service_name, config)
    
    def generate_sklearn_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for sklearn models"""
        return self.sklearn_builder.generate_service_code(model, bento_model_tag, config)
    
    def generate_tensorflow_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for TensorFlow models"""
        return self.tensorflow_builder.generate_service_code(model, bento_model_tag, config)
    
    def generate_pytorch_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for PyTorch models"""
        return self.pytorch_builder.generate_service_code(model, bento_model_tag, config)
    
    def generate_xgboost_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for XGBoost models"""
        return self.xgboost_builder.generate_service_code(model, bento_model_tag, config)
    
    def generate_lightgbm_service_code(self, model: Model, bento_model_tag: str, config: Dict[str, Any]) -> str:
        """Generate BentoML service code for LightGBM models"""
        return self.lightgbm_builder.generate_service_code(model, bento_model_tag, config)

