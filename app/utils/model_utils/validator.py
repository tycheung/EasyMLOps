"""
Core model validation utilities
Provides file validation, hash calculation, and main validation orchestration
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import aiofiles
import aiofiles.os as aios
import asyncio

from app.schemas.model import ModelFramework, ModelType, ModelValidationResult
from app.config import get_settings
from app.utils.model_utils.frameworks import FrameworkDetector

settings = get_settings()
logger = logging.getLogger(__name__)


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
            detected_framework = await FrameworkDetector.detect_framework_from_file_async(file_path)
        except Exception as e:
            errors.append(f"Error detecting framework: {str(e)}")
            logger.error(f"Framework detection failed for {file_path}: {e}", exc_info=True)

        if detected_framework:
            metadata['detected_framework'] = detected_framework.value
            # Model type detection (depends on framework)
            try:
                detected_model_type = await FrameworkDetector.detect_model_type_async(file_path, detected_framework)
            except Exception as e:
                errors.append(f"Error detecting model type: {str(e)}")
                logger.error(f"Model type detection failed for {file_path}: {e}", exc_info=True)
            
            if detected_model_type:
                metadata['detected_model_type'] = detected_model_type.value

            # Metadata extraction (depends on framework)
            try:
                specific_metadata = await FrameworkDetector.get_framework_metadata_async(file_path, detected_framework)
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

