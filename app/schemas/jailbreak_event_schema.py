from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime

# Base model with required fields
class JailbreakEventBase(BaseModel):
    jailbreak_prompt: str
    
# Used for creating events
class JailbreakEventCreate(JailbreakEventBase):
    jailbreak_result: Optional[Dict[str, Any]] = None
    proof_url: Optional[str] = None

# Used for updating events (if needed)
class JailbreakEventUpdate(BaseModel):
    jailbreak_result: Optional[Dict[str, Any]] = None
    proof_url: Optional[str] = None
    proof_path: Optional[str] = None

# Response model with all attributes
class JailbreakEventResponse(JailbreakEventBase):
    id: int
    model_id: int
    jailbreak_result: Optional[Dict[str, Any]] = None
    proof_url: Optional[str] = None
    proof_path: Optional[str] = None
    created_at: datetime
    
    # Pydantic v2 style config
    model_config = ConfigDict(from_attributes=True)