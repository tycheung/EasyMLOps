"""
Schema versioning operations
Handles saving, updating, deleting, and retrieving schema versions
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import asyncio

import aiofiles
import aiofiles.os as aios

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class SchemaVersioning:
    """Schema versioning operations"""
    
    async def save_model_schema(self, model_id: str, schema_type: str, schema_data: Dict[str, Any], version: str = "1.0") -> Dict[str, Any]:
        """Saves a model schema to the filesystem asynchronously."""
        try:
            # Ensure model directory exists (can be sync as it's a pre-check)
            model_base_dir = Path(settings.MODELS_DIR) / model_id
            if not await aios.path.exists(model_base_dir):
                 logger.warning(f"Base directory for model {model_id} not found at {model_base_dir}")

            schema_dir = model_base_dir / "schemas" / schema_type / version
            await aios.makedirs(schema_dir, exist_ok=True)
            
            file_path = schema_dir / "schema.json"
            
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(schema_data, indent=2))
            
            logger.info(f"Saved schema for model {model_id}, type {schema_type}, version {version} to {file_path}")
            return {
                "message": "Schema saved successfully",
                "model_id": model_id,
                "schema_type": schema_type,
                "version": version,
                "path": str(file_path)
            }
        except Exception as e:
            logger.error(f"Error saving schema for model {model_id}: {e}")
            raise
    
    async def update_model_schema(self, schema_id: str, schema_data: Dict[str, Any], version: Optional[str] = None) -> Dict[str, Any]:
        """Updates an existing model schema asynchronously. schema_id is expected to be model_id/schema_type."""
        try:
            parts = schema_id.split('/')
            if len(parts) < 2:
                raise ValueError("Invalid schema_id format. Expected model_id/schema_type[/version]")
            
            model_id = parts[0]
            schema_type = parts[1]
            target_version = version or (parts[2] if len(parts) > 2 else "1.0")

            schema_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type / target_version
            file_path = schema_dir / "schema.json"

            if not await aios.path.exists(file_path):
                logger.error(f"Schema file not found at {file_path} for update.")
                raise FileNotFoundError(f"Schema file not found at {file_path}. Cannot update.")

            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(schema_data, indent=2))
            
            logger.info(f"Updated schema for {schema_id}, version {target_version} at {file_path}")
            return {
                "message": "Schema updated successfully",
                "schema_id": schema_id,
                "model_id": model_id,
                "schema_type": schema_type,
                "version": target_version,
                "path": str(file_path)
            }
        except FileNotFoundError as fnf_e:
            logger.error(f"Error updating schema {schema_id}: {fnf_e}")
            raise
        except Exception as e:
            logger.error(f"Error updating schema {schema_id}: {e}")
            raise
    
    async def delete_model_schema(self, schema_id: str) -> bool:
        """Deletes a model schema (or a specific version) from the filesystem asynchronously."""
        try:
            parts = schema_id.split('/')
            if len(parts) < 2:
                raise ValueError("Invalid schema_id format. Expected model_id/schema_type[/version]")

            model_id = parts[0]
            schema_type = parts[1]
            version = parts[2] if len(parts) > 2 else None

            model_schema_base_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type

            if not await aios.path.exists(model_schema_base_dir):
                logger.warning(f"Schema directory not found for {schema_id} at {model_schema_base_dir}. Nothing to delete.")
                return False

            if version:
                # Delete specific version
                schema_file_path = model_schema_base_dir / version / "schema.json"
                version_dir_path = model_schema_base_dir / version
                if await aios.path.exists(schema_file_path):
                    await aios.remove(schema_file_path)
                    try:
                        await aios.rmdir(version_dir_path)
                        logger.info(f"Removed empty version directory: {version_dir_path}")
                    except OSError:
                        pass
                    logger.info(f"Deleted schema version {schema_id} from {schema_file_path}")
                    return True
                else:
                    logger.warning(f"Schema version file not found for {schema_id} at {schema_file_path}. Nothing to delete.")
                    return False
            else:
                # Delete all versions (the entire schema_type directory)
                if await aios.path.exists(model_schema_base_dir):
                    await asyncio.to_thread(shutil.rmtree, model_schema_base_dir)
                    logger.info(f"Deleted all schema versions for {model_id}/{schema_type} from {model_schema_base_dir}")
                    return True
                else:
                    logger.warning(f"Schema directory not found for {model_id}/{schema_type}. Nothing to delete.")
                    return False

        except ValueError as ve:
            logger.error(f"Invalid schema_id for delete: {ve}")
            return False
        except Exception as e:
            logger.error(f"Error deleting schema {schema_id}: {e}")
            return False
    
    async def get_schema_versions(self, schema_id: str) -> List[Dict[str, Any]]:
        """Retrieves all available versions of a schema asynchronously."""
        versions_info = []
        try:
            parts = schema_id.split('/')
            if len(parts) != 2:
                raise ValueError("Invalid schema_id format. Expected model_id/schema_type")

            model_id = parts[0]
            schema_type = parts[1]

            schema_type_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type

            if not await aios.path.exists(schema_type_dir) or not await aios.path.isdir(schema_type_dir):
                logger.info(f"Schema type directory not found for {schema_id} at {schema_type_dir}")
                return []

            # Iterate over version directories
            version_dirs = await aios.listdir(schema_type_dir)
            for version_name in version_dirs:
                version_path = schema_type_dir / version_name
                if await aios.path.isdir(version_path):
                    schema_file_path = version_path / "schema.json"
                    if await aios.path.exists(schema_file_path):
                        try:
                            async with aiofiles.open(schema_file_path, "r") as f:
                                content = await f.read()
                                schema_data = json.loads(content)
                            
                            try:
                                stat_result = await aios.stat(schema_file_path)
                                last_modified = datetime.fromtimestamp(stat_result.st_mtime).isoformat()
                            except Exception as stat_err:
                                logger.warning(f"Could not get stat for {schema_file_path}: {stat_err}")
                                last_modified = datetime.utcnow().isoformat()

                            versions_info.append({
                                "version": version_name,
                                "schema_id": f"{schema_id}/{version_name}",
                                "model_id": model_id,
                                "schema_type": schema_type,
                                "description": schema_data.get("description", "N/A"),
                                "retrieved_at": datetime.utcnow().isoformat(),
                                "last_modified": last_modified,
                            })
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Error decoding JSON for schema {schema_file_path}: {json_err}")
                        except Exception as e:
                            logger.error(f"Error reading schema version {version_path}: {e}")
            
            versions_info.sort(key=lambda x: x.get("version", "0.0.0"), reverse=True)
            return versions_info
        except ValueError as ve:
            logger.error(f"Invalid schema_id for get_schema_versions: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving schema versions for {schema_id}: {e}")
            raise
    
    async def create_schema_version(
        self, 
        schema_id_base: str,
        schema_data: Dict[str, Any], 
        version: str, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new version of a schema asynchronously."""
        try:
            parts = schema_id_base.split('/')
            if len(parts) != 2:
                raise ValueError("Invalid schema_id_base format. Expected model_id/schema_type")

            model_id = parts[0]
            schema_type = parts[1]

            if description and "description" not in schema_data:
                schema_data["description"] = description

            schema_version_dir = Path(settings.MODELS_DIR) / model_id / "schemas" / schema_type / version
            
            if await aios.path.exists(schema_version_dir):
                raise FileExistsError(f"Schema version {version} already exists for {schema_id_base}")

            await aios.makedirs(schema_version_dir, exist_ok=True)
            
            file_path = schema_version_dir / "schema.json"
            
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(schema_data, indent=2))
            
            logger.info(f"Created schema version {version} for {schema_id_base} at {file_path}")
            return {
                "message": "Schema version created successfully",
                "schema_id": f"{schema_id_base}/{version}",
                "model_id": model_id,
                "schema_type": schema_type,
                "version": version,
                "description": description or schema_data.get("description"),
                "path": str(file_path),
                "created_at": datetime.utcnow().isoformat()
            }
        except FileExistsError as fee:
            logger.error(f"Error creating schema version for {schema_id_base}: {fee}")
            raise
        except ValueError as ve:
            logger.error(f"Invalid input for create_schema_version: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error creating schema version for {schema_id_base}: {e}")
            raise

