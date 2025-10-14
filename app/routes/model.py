from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, Dict

from app.db.session import get_db
from app.models.model import Model

router = APIRouter()

# ------------------------------------------
#  Pydantic Schemas
# ------------------------------------------

class ModelCreate(BaseModel):
    model_id: str
    created_by: int
    report: Optional[Dict] = {}
    score: Optional[float] = 0
    suggestions: Optional[str] = ""


class ModelUpdate(BaseModel):
    score: Optional[float] = None
    suggestions: Optional[str] = None
    report: Optional[Dict] = None


# ------------------------------------------
#  Create Model
# ------------------------------------------
@router.post("/create")
async def create_model(data: ModelCreate, db: Session = Depends(get_db)):
    existing = db.query(Model).filter(Model.model_id == data.model_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model with this ID already exists")

    new_model = Model(
        model_id=data.model_id,
        created_by=data.created_by,
        report=data.report,
        score=data.score,
        suggestions=data.suggestions
    )

    db.add(new_model)
    db.commit()
    db.refresh(new_model)

    return {
        "message": "Model created successfully",
        "model": {
            "id": new_model.id,
            "model_id": new_model.model_id,
            "score": new_model.score,
            "suggestions": new_model.suggestions
        }
    }


# ------------------------------------------
#  List All Models
# ------------------------------------------
@router.get("/list")
async def list_models(db: Session = Depends(get_db)):
    models = db.query(Model).all()
    return models


# ------------------------------------------
#  Get Single Model by model_id
# ------------------------------------------
@router.get("/{model_id}")
async def get_model(model_id: str, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


# ------------------------------------------
#  Update Model (Score / Suggestions / Report)
# ------------------------------------------
@router.patch("/{model_id}/update")
async def update_model(model_id: str, data: ModelUpdate, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if data.score is not None:
        model.score = data.score
    if data.suggestions:
        model.suggestions = data.suggestions
    if data.report is not None:
        model.report = data.report

    db.commit()
    db.refresh(model)

    return {"message": "Model updated successfully", "model": model}


# ------------------------------------------
#  Delete Model
# ------------------------------------------
@router.delete("/{model_id}/delete")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(model)
    db.commit()

    return {"message": "Model deleted successfully"}