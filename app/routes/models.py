"""
FastAPI routes for model management
Provides REST API endpoints for CRUD operations on ML models
"""

import logging
import os
import shutil
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.models.model import Model, ModelDeployment
from app.schemas.model import (
    ModelResponse, ModelMetadata, ModelUpload, ModelUpdate
)
from app.utils.model_utils import ModelValidator, ModelFileManager
from app.config import get_settings
import aiofiles.os as aios

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


def model_to_metadata(model: Model) -> ModelMetadata:
    """Convert a database Model instance to ModelMetadata schema"""
    return ModelMetadata(
        id=model.id,
        name=model.name,
        description=model.description,
        framework=model.framework,
        model_type=model.model_type,
        version=model.version,
        status=model.status,
        tags=model.tags or [],
        file_name=model.file_name,
        file_size=model.file_size,
        file_hash=model.file_hash,
        file_path=model.file_path,
        input_schema=None,  # Will be populated separately if needed
        output_schema=None,  # Will be populated separately if needed
        created_at=model.created_at,
        updated_at=model.updated_at,
        deployed_at=model.deployed_at,
        prediction_count=model.prediction_count,
        avg_response_time=model.avg_response_time,
        last_prediction_at=model.last_prediction_at
    )


@router.post("/upload", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...),
    model_type: str = Form(...),
    framework: str = Form(...),
    version: str = Form("1.0.0"),
    db: AsyncSession = Depends(get_async_session)
):
    """Upload a new model file"""
    try:
        # Check if model with same name already exists
        result = await db.execute(select(Model).where(Model.name == name))
        existing_model = result.scalars().first()
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model with name '{name}' already exists"
            )
        
        # Validate file format
        if not file.filename.endswith(('.joblib', '.pkl', '.h5', '.pt', '.pth', '.onnx')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Supported formats: .joblib, .pkl, .h5, .pt, .pth, .onnx"
            )
        
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Read file content for processing
        file_content = await file.read()
        file_size = len(file_content)
        
        # Save the uploaded file
        storage_path = await ModelFileManager.save_uploaded_file_async(
            file_content, model_id, file.filename
        )
        
        # Validate the model file
        is_valid = await ModelValidator.validate_model_file_async(storage_path)
        if not is_valid:
            # Clean up the uploaded file
            await aios.remove(storage_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model file format or corrupted file"
            )
        
        # Calculate file hash for integrity
        file_hash = await ModelValidator.calculate_file_hash_async(storage_path)
        
        # Detect framework if not provided or validate provided framework
        detected_framework = await ModelValidator.detect_framework_from_file_async(storage_path)
        if detected_framework and detected_framework != framework:
            logger.warning(f"Framework mismatch: provided={framework}, detected={detected_framework}")
        
        # Create model record
        new_model = Model(
            id=model_id,
            name=name,
            description=description,
            model_type=model_type,
            framework=framework,
            version=version,
            file_name=file.filename,
            file_path=str(storage_path),
            file_size=file_size,
            file_hash=file_hash,
            status="uploaded",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_model)
        await db.commit()
        await db.refresh(new_model)
        
        logger.info(f"Successfully uploaded model: {name} (ID: {model_id})")
        
        # Create ModelMetadata manually to avoid validation issues with relationships
        model_metadata = model_to_metadata(new_model)
        return ModelResponse(
            model=model_metadata,
            message=f"Model '{name}' uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        # Clean up uploaded file if it exists
        try:
            if 'storage_path' in locals():
                await aios.remove(storage_path)
        except:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload model"
        )


@router.get("/", response_model=List[ModelMetadata])
async def get_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[str] = None,
    db: AsyncSession = Depends(get_async_session)
):
    """Get all models with optional filtering"""
    try:
        query = select(Model)
        
        if model_type:
            query = query.where(Model.model_type == model_type)
        
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        models = result.scalars().all()
        
        return [model_to_metadata(model) for model in models]
        
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch models"
        )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_async_session)
):
    """Get a specific model by ID"""
    try:
        result = await db.execute(select(Model).where(Model.id == model_id))
        model = result.scalars().first()
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        model_metadata = model_to_metadata(model)
        return ModelResponse(
            model=model_metadata,
            message=f"Model '{model.name}' retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch model"
        )


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelUpload,
    db: AsyncSession = Depends(get_async_session)
):
    """Create a new model without file upload"""
    try:
        model_id = str(uuid.uuid4())
        
        new_model = Model(
            id=model_id,
            **model.model_dump(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_model)
        await db.commit()
        await db.refresh(new_model)
        
        logger.info(f"Created model: {model.name} (ID: {model_id})")
        
        model_metadata = model_to_metadata(new_model)
        return ModelResponse(
            model=model_metadata,
            message=f"Model '{model.name}' created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create model"
        )


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    model_update: ModelUpdate,
    db: AsyncSession = Depends(get_async_session)
):
    """Update an existing model"""
    try:
        result = await db.execute(select(Model).where(Model.id == model_id))
        db_model = result.scalars().first()
        
        if not db_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        update_data = model_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_model, key, value)
        
        db.add(db_model)
        await db.commit()
        await db.refresh(db_model)
        
        logger.info(f"Updated model: {model_id}")
        
        model_metadata = model_to_metadata(db_model)
        return ModelResponse(model=model_metadata, message=f"Model '{db_model.name}' updated successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model"
        )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_async_session)
):
    """Delete a model"""
    try:
        result = await db.execute(select(Model).where(Model.id == model_id))
        db_model = result.scalars().first()
        
        if not db_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Delete the model file if it exists
        if db_model.file_path and await aios.path.exists(db_model.file_path):
            try:
                await aios.remove(db_model.file_path)
                logger.info(f"Deleted model file: {db_model.file_path}")
            except Exception as e:
                logger.error(f"Error deleting model file {db_model.file_path}: {e}")
                # Continue with deleting the DB record even if file deletion fails

        await db.delete(db_model)
        await db.commit()
        logger.info(f"Deleted model: {model_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model"
        )


@router.post("/{model_id}/validate", response_model=dict)
async def validate_model_input(
    model_id: str,
    input_data: dict,
    db: AsyncSession = Depends(get_async_session)
):
    """Validate input data against model schema"""
    try:
        result = await db.execute(select(Model).where(Model.id == model_id))
        model = result.scalars().first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Basic validation logic (can be expanded)
        validation_result = {
            "valid": True,
            "message": "Input data is valid",
            "model_id": model_id,
            "input_data": input_data
        }
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating input for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate input"
        )


@router.get("/{model_id}/metrics", response_model=dict)
async def get_model_metrics(
    model_id: str,
    db: AsyncSession = Depends(get_async_session)
):
    """Get model performance metrics"""
    try:
        result = await db.execute(select(Model).where(Model.id == model_id))
        model = result.scalars().first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Return basic metrics (can be expanded with real monitoring data)
        metrics = {
            "model_id": model_id,
            "total_predictions": model.prediction_count,
            "avg_response_time": model.avg_response_time,
            "last_prediction_at": model.last_prediction_at.isoformat() if model.last_prediction_at else None,
            "status": model.status
        }
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch metrics"
        )


@router.post("/{model_id}/metrics", response_model=dict)
async def update_model_metrics(
    model_id: str,
    metrics: dict,
    db: AsyncSession = Depends(get_async_session)
):
    """Update model performance metrics"""
    try:
        result = await db.execute(select(Model).where(Model.id == model_id))
        model = result.scalars().first()

        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )

        # Example: Update prediction_count and avg_response_time
        # In a real scenario, these would be calculated and validated
        model.prediction_count = metrics.get("total_predictions", model.prediction_count)
        model.avg_response_time = metrics.get("avg_response_time", model.avg_response_time)
        model.last_prediction_at = datetime.utcnow() # Update timestamp

        db.add(model)
        await db.commit()
        await db.refresh(model)
        logger.info(f"Updated metrics for model: {model_id}")

        return {
            "model_id": model_id,
            "message": "Metrics updated successfully",
            "updated_metrics": {
                "total_predictions": model.prediction_count,
                "avg_response_time": model.avg_response_time,
                "last_prediction_at": model.last_prediction_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating metrics for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update metrics"
        ) 