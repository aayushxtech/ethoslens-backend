from fastapi import APIRouter, HTTPException, Depends, Path, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db
from app.models.model import Model
from app.schemas.model_schema import ModelCreate, ModelUpdate, ModelResponse, ModelDetailResponse

# Add tags for documentation organization
router = APIRouter(tags=["models"])

# ------------------------------------------
#  Create Model
# ------------------------------------------
@router.post("/create", response_model=ModelResponse)
async def create_model(
    data: ModelCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new model record.
    
    Requires a unique model_id and valid created_by user ID.
    """
    existing = db.query(Model).filter(Model.model_id == data.model_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model with this ID already exists")

    new_model = Model(
        model_id=data.model_id,
        created_by=data.created_by,
        report=data.report or {},
        score=data.score,
        suggestions=data.suggestions or ""
    )

    try:
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        return new_model
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


# ------------------------------------------
#  List All Models
# ------------------------------------------
@router.get("/list", response_model=List[ModelResponse])
async def list_models(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
):
    """List all models with pagination support"""
    models = db.query(Model).offset(skip).limit(limit).all()
    return models


# ------------------------------------------
#  Get Single Model by model_id
# ------------------------------------------
@router.get("/{model_id}", response_model=ModelDetailResponse)
async def get_model(
    model_id: str = Path(..., description="Unique model identifier"),
    include_events: bool = Query(False, description="Include related jailbreak events"),
    db: Session = Depends(get_db)
):
    """
    Get a single model by its unique model_id.
    
    Optionally include related jailbreak events.
    """
    query = db.query(Model).filter(Model.model_id == model_id)
    
    if include_events:
        query = query.options(joinedload(Model.jailbreak_events))
    
    model = query.first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model


# ------------------------------------------
#  Update Model (Score / Suggestions / Report)
# ------------------------------------------
@router.patch("/{model_id}/update", response_model=ModelResponse)
async def update_model(
    model_id: str,
    data: ModelUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing model's attributes"""
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Update only the fields that were provided
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(model, key, value)

    try:
        db.commit()
        db.refresh(model)
        return model
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


# ------------------------------------------
#  Delete Model
# ------------------------------------------
@router.delete("/{model_id}/delete", response_model=dict)
async def delete_model(
    model_id: str = Path(..., description="Model identifier to delete"),
    db: Session = Depends(get_db)
):
    """Delete a model and all related jailbreak events"""
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # All related jailbreak_events will be deleted via cascade
        db.delete(model)
        db.commit()
        return {"message": "Model deleted successfully", "model_id": model_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")