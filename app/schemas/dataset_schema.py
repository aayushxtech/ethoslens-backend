from typing import Dict, Any
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from app.schemas.column_schema import ColumnResponse


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
    target_column: Optional[str] = None
    sensitive_columns: List[str] = []

    class Config:
        orm_mode = True


class ConfirmColumnsRequest(BaseModel):
    target_column: str
    sensitive_columns: List[str] = []

    class Config:
        orm_mode = True


class DatasetEvaluationResponse(BaseModel):
    dataset_id: int
    dataset_name: str
    total_rows: int
    columns: List[ColumnResponse]


class TestResult(BaseModel):
    test_name: str
    status: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None


class EvaluationResponse(BaseModel):
    dataset_id: int
    dataset_name: str
    fairness: Optional[List[TestResult]] = []
