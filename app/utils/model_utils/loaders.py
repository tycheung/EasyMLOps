"""
Model file loading and management utilities
Handles file storage, retrieval, and deletion operations
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict
import aiofiles
import aiofiles.os as aios
import asyncio
from datetime import datetime

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


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
        safe_filename = f"{model_id}_{Path(filename).name}"
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
            import shutil
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
                    import shutil
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
                            fp = os.path.join(dirpath, filename)
                            total_size += (await aios.stat(fp)).st_size
                        except OSError:
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

