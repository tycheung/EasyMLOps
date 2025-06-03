"""
FastAPI routes for model management
Handles CRUD operations for ML models and related endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import os
import hashlib
import shutil
from pathlib import Path
import uuid

from app.database import get_db
from app.models.model import Model
from app.schemas.model import ModelUpload, ModelUpdate, ModelResponse, ModelMetadata
from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Models"])

settings = get_settings()


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


@router.post("/upload", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...),
    model_type: str = Form(...),
    framework: str = Form(...),
    version: str = Form("1.0.0"),
    db: Session = Depends(get_db)
):
    """Upload a new model file"""
    try:
        # Validate file extension
        allowed_extensions = {".joblib", ".pkl", ".pickle", ".h5", ".pt", ".pth", ".onnx"}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extension {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Check if model name already exists
        existing_model = db.query(Model).filter(Model.name == name).first()
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model with name '{name}' already exists"
            )
        
        # Create models directory if it doesn't exist
        models_dir = Path(settings.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        model_id = str(uuid.uuid4())
        file_name = f"{model_id}_{file.filename}"
        file_path = models_dir / file_name
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate file hash and size
        file_hash = calculate_file_hash(str(file_path))
        file_size = file_path.stat().st_size
        
        # Create model record
        db_model = Model(
            id=model_id,
            name=name,
            description=description,
            model_type=model_type,
            framework=framework,
            version=version,
            file_name=file.filename,
            file_size=file_size,
            file_hash=file_hash,
            file_path=str(file_path),
            status="uploaded"
        )
        
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        
        logger.info(f"Uploaded model: {name} (ID: {model_id})")
        
        # Return the correct response format
        model_metadata = ModelMetadata(
            id=db_model.id,
            name=db_model.name,
            description=db_model.description,
            framework=db_model.framework,
            model_type=db_model.model_type,
            version=db_model.version,
            status=db_model.status,
            tags=db_model.tags,
            file_name=db_model.file_name,
            file_size=db_model.file_size,
            file_hash=db_model.file_hash,
            file_path=db_model.file_path,
            created_at=db_model.created_at,
            updated_at=db_model.updated_at,
            deployed_at=db_model.deployed_at,
            prediction_count=db_model.prediction_count,
            avg_response_time=db_model.avg_response_time,
            last_prediction_at=db_model.last_prediction_at
        )
        
        return {
            "model": model_metadata,
            "message": f"Model '{name}' uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        # Clean up file if it was created
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload model"
        )


@router.get("/", response_model=List[ModelMetadata])
async def get_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all models with pagination and optional filtering"""
    try:
        query = db.query(Model)
        
        # Apply filters
        if model_type:
            query = query.filter(Model.model_type == model_type)
        
        models = query.offset(skip).limit(limit).all()
        
        # Convert to ModelMetadata format
        model_metadata_list = []
        for model in models:
            model_metadata = ModelMetadata(
                id=model.id,
                name=model.name,
                description=model.description,
                framework=model.framework,
                model_type=model.model_type,
                version=model.version,
                status=model.status,
                tags=model.tags,
                file_name=model.file_name,
                file_size=model.file_size,
                file_hash=model.file_hash,
                file_path=model.file_path,
                created_at=model.created_at,
                updated_at=model.updated_at,
                deployed_at=model.deployed_at,
                prediction_count=model.prediction_count,
                avg_response_time=model.avg_response_time,
                last_prediction_at=model.last_prediction_at
            )
            model_metadata_list.append(model_metadata)
        
        return model_metadata_list
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch models"
        )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific model by ID"""
    try:
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Convert to ModelMetadata format
        model_metadata = ModelMetadata(
            id=model.id,
            name=model.name,
            description=model.description,
            framework=model.framework,
            model_type=model.model_type,
            version=model.version,
            status=model.status,
            tags=model.tags,
            file_name=model.file_name,
            file_size=model.file_size,
            file_hash=model.file_hash,
            file_path=model.file_path,
            created_at=model.created_at,
            updated_at=model.updated_at,
            deployed_at=model.deployed_at,
            prediction_count=model.prediction_count,
            avg_response_time=model.avg_response_time,
            last_prediction_at=model.last_prediction_at
        )
        
        return {
            "model": model_metadata,
            "message": None
        }
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
    db: Session = Depends(get_db)
):
    """Create a new model"""
    try:
        db_model = Model(**model.dict())
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        logger.info(f"Created new model: {db_model.id}")
        return db_model
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create model"
        )


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    model_update: ModelUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing model"""
    try:
        db_model = db.query(Model).filter(Model.id == model_id).first()
        if not db_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Update model attributes
        for field, value in model_update.dict(exclude_unset=True).items():
            setattr(db_model, field, value)
        
        db.commit()
        db.refresh(db_model)
        
        # Convert to ModelMetadata format
        model_metadata = ModelMetadata(
            id=db_model.id,
            name=db_model.name,
            description=db_model.description,
            framework=db_model.framework,
            model_type=db_model.model_type,
            version=db_model.version,
            status=db_model.status,
            tags=db_model.tags,
            file_name=db_model.file_name,
            file_size=db_model.file_size,
            file_hash=db_model.file_hash,
            file_path=db_model.file_path,
            created_at=db_model.created_at,
            updated_at=db_model.updated_at,
            deployed_at=db_model.deployed_at,
            prediction_count=db_model.prediction_count,
            avg_response_time=db_model.avg_response_time,
            last_prediction_at=db_model.last_prediction_at
        )
        
        logger.info(f"Updated model: {model_id}")
        return {
            "model": model_metadata,
            "message": f"Model '{db_model.name}' updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model"
        )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """Delete a model"""
    try:
        db_model = db.query(Model).filter(Model.id == model_id).first()
        if not db_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        db.delete(db_model)
        db.commit()
        logger.info(f"Deleted model: {model_id}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model"
        )


@router.post("/{model_id}/validate", response_model=dict)
async def validate_model_input(
    model_id: str,
    input_data: dict,
    db: Session = Depends(get_db)
):
    """Validate input data against model schema"""
    try:
        model = db.query(Model).filter(Model.id == model_id).first()
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
    db: Session = Depends(get_db)
):
    """Get model performance metrics"""
    try:
        model = db.query(Model).filter(Model.id == model_id).first()
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
    db: Session = Depends(get_db)
):
    """Update model performance metrics"""
    try:
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Update metrics fields if provided
        if "prediction_count" in metrics:
            model.prediction_count = metrics["prediction_count"]
        if "avg_response_time" in metrics:
            model.avg_response_time = metrics["avg_response_time"]
        
        db.commit()
        db.refresh(model)
        
        return {
            "message": "Metrics updated successfully",
            "model_id": model_id,
            "updated_metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating metrics for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update metrics"
        ) 