from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Optional, List
from app.services.dataset import process_upload
from app.db.session import get_db
from app.models.dataset import Dataset
from app.schemas.dataset_schema import DatasetResponse, ColumnSchema, ColumnSuggestion, ConfirmColumnsRequest
from app.models.columns import DatasetColumn  # <- added import

from app.utils.file_utils import allowed_file
from app.utils.llm_utils import call_llm

import pandas as pd
import json

router = APIRouter()


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db), user_id: int = 1):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    dataset = process_upload(file, db=db, uploaded_by=user_id)

    # Convert schema JSON to ColumnSchema list
    columns = [ColumnSchema(**col) for col in (dataset.schema or [])]

    sensitive_cols = dataset.sensitive_columns if dataset.sensitive_columns else []

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        file_type=dataset.file_type,
        file_path=dataset.file_path,
        row_count=dataset.row_count,
        target_column=dataset.target_column,
        sensitive_columns=sensitive_cols,
        columns=columns,
        status=dataset.status
    )
