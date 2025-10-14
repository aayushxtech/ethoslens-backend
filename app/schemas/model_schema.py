from pydantic import BaseModel
from typing import Optional, Dict
from uuid import UUID

class ModelBase(BaseModel):
    created_by: str
    report: Optional[Dict] = None
    score: Optional[float] = 0.0
    suggestion: Optional[str] = None

class ModelCreate(ModelBase):
    pass

class ModelResponse(ModelBase):
    id: UUID

    class Config:
        orm_mode = True