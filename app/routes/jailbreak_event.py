from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Path, Query
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import os
import shutil
import json
from datetime import datetime

from app.db.session import get_db
from app.models.model import Model
from app.models.jailbreak_event import JailbreakEvent
from app.schemas.jailbreak_event_schema import JailbreakEventCreate, JailbreakEventResponse, JailbreakEventUpdate

router = APIRouter(tags=["jailbreak_events"])

@router.post("/models/{model_id}/jailbreak_events", response_model=JailbreakEventResponse)
async def create_jailbreak_event(
    model_id: str = Path(..., description="Model identifier (string)"),
    jailbreak_prompt: str = Form(..., description="The prompt used for jailbreak attempt"),
    jailbreak_result: Optional[str] = Form(None, description="JSON string of jailbreak result or raw text"),
    proof: Optional[UploadFile] = File(None, description="Optional proof file upload"),
    db: Session = Depends(get_db)
):
    """
    Create a new jailbreak event record for a model, optionally with proof file.
    
    - Requires form data with jailbreak_prompt
    - Optionally accepts jailbreak_result as JSON string
    - Optionally accepts a proof file upload
    """
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Parse jailbreak_result if provided
    parsed_result: Dict[str, Any] = {}
    if jailbreak_result:
        try:
            parsed_result = json.loads(jailbreak_result)
        except json.JSONDecodeError:
            parsed_result = {"raw": jailbreak_result}

    # Create event instance
    event = JailbreakEvent(
        model_id=model.id,
        jailbreak_prompt=jailbreak_prompt,
        jailbreak_result=parsed_result,
        created_at=datetime.utcnow()
    )

    # Handle proof file upload if provided
    if proof:
        try:
            proofs_dir = os.path.join(os.getcwd(), "proofs")
            os.makedirs(proofs_dir, exist_ok=True)
            filename = f"{model.id}_{int(datetime.utcnow().timestamp())}_{proof.filename}"
            dest_path = os.path.join(proofs_dir, filename)
            
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(proof.file, f)
                
            event.proof_path = dest_path
            event.proof_url = f"/proofs/{filename}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save proof file: {str(e)}")

    # Save to database
    try:
        db.add(event)
        db.commit()
        db.refresh(event)
        return event
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/models/{model_id}/jailbreak_events", response_model=List[JailbreakEventResponse])
async def list_jailbreak_events(
    model_id: str = Path(..., description="Model identifier"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    db: Session = Depends(get_db)
):
    """List all jailbreak events for a specific model"""
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    events = db.query(JailbreakEvent)\
        .filter(JailbreakEvent.model_id == model.id)\
        .order_by(JailbreakEvent.created_at.desc())\
        .limit(limit)\
        .all()
    
    return events

@router.get("/models/by-pk/{model_pk}/jailbreak_events", response_model=List[JailbreakEventResponse])
async def list_jailbreak_events_by_pk(
    model_pk: int = Path(..., description="Model primary key (numeric ID)"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all jailbreak events for a model by its numeric primary key"""
    model = db.query(Model).filter(Model.id == model_pk).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    events = db.query(JailbreakEvent)\
        .filter(JailbreakEvent.model_id == model.id)\
        .order_by(JailbreakEvent.created_at.desc())\
        .limit(limit)\
        .all()
        
    return events

@router.get("/jailbreak_events/{event_id}", response_model=JailbreakEventResponse)
async def get_jailbreak_event(
    event_id: int = Path(..., description="Jailbreak event ID"),
    db: Session = Depends(get_db)
):
    """Get a specific jailbreak event by ID"""
    event = db.query(JailbreakEvent).filter(JailbreakEvent.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Jailbreak event not found")
    return event

@router.patch("/jailbreak_events/{event_id}", response_model=JailbreakEventResponse)
async def update_jailbreak_event(
    event_id: int,
    update_data: JailbreakEventUpdate,
    db: Session = Depends(get_db)
):
    """Update a jailbreak event record (partial update)"""
    event = db.query(JailbreakEvent).filter(JailbreakEvent.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Jailbreak event not found")
    
    # Update only fields that are provided
    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(event, key, value)
    
    db.commit()
    db.refresh(event)
    return event