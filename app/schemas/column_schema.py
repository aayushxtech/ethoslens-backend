from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class ColumnStats(BaseModel):
    name: str
    column_type: str  # numeric, categorical, boolean, other
    missing_count: int
    missing_percentage: float
    unique_count: int
    stats: Dict[str, Any]  # numeric stats, top/freq for categorical

    # New optional fields returned in responses
    example_value: Optional[Any] = None
    top_values: Optional[Dict[str, int]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


class ColumnResponse(ColumnStats):
    is_target: bool = False
    is_sensitive: bool = False

    class Config:
        orm_mode = True
