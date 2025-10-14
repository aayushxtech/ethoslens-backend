from typing import Dict, Any
from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class ColumnStats(BaseModel):
    name: str
    column_type: str  # numeric, categorical, boolean, other
    missing_count: int
    missing_percentage: float
    unique_count: int
    stats: Dict[str, Any]  # numeric stats, top/freq for categorical

    class Config:
        orm_mode = True


class ColumnResponse(ColumnStats):
    is_target: bool = False
    is_sensitive: bool = False

    class Config:
        orm_mode = True
