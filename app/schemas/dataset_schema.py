from pydantic import BaseModel
from typing import List, Optional, Any


class ColumnSchema(BaseModel):
    column_name: str
    column_type: str
    null_count: Optional[int]
    unique_count: Optional[int]
    example_value: Optional[Any]

    class Config:
        orm_mode = True


class DatasetResponse(BaseModel):
    id: int
    name: str
    file_type: str
    file_path: str
    row_count: Optional[int]
    target_column: Optional[str]
    sensitive_columns: Optional[List[str]] = None
    columns: List[ColumnSchema]  # List of columns in the dataset
    status: str

    class Config:
        orm_mode = True


class ColumnSuggestion(BaseModel):
    target_column: str
    sensitive_column: str

    class Config:
        orm_mode = True


class ConfirmColumnsRequest(BaseModel):
    target_column: str
    sensitive_columns: List[str] = []

    class Config:
        orm_mode = True
