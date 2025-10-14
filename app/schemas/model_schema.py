from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List, Any
from datetime import datetime
from .jailbreak_event_schema import JailbreakEventResponse

# Base model with shared attributes
class ModelBase(BaseModel):
    created_by: int
    report: Optional[Dict[str, Any]] = {}
    score: float = 0.0
    suggestions: Optional[str] = None

# Used for creating new models
class ModelCreate(ModelBase):
    model_id: str

# Used for updating existing models
class ModelUpdate(BaseModel):
    report: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    suggestions: Optional[str] = None

# Response model with all attributes including database id
class ModelResponse(ModelBase):
    id: int
    model_id: str
    # Optional nested relationship for use with joined loads
    jailbreak_events: Optional[List['JailbreakEventResponse']] = None
    
    # Pydantic v2 style config
    model_config = ConfigDict(from_attributes=True)

# Detailed response with nested jailbreak events
class ModelDetailResponse(ModelResponse):
    pass