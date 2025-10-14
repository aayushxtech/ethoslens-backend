from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Optional, List
from app.services.dataset import process_upload
from app.db.session import get_db
from app.models.dataset import Dataset
from app.schemas.dataset_schema import DatasetResponse, ColumnSchema, ColumnSuggestion, ConfirmColumnsRequest

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


def _update_dataset_columns(dataset: Dataset, target: Optional[str], sensitive: List[str], db: Session, status: str = "confirmed") -> Dataset:
    """
    Helper: persist target and sensitive columns (saves as JSON list).
    Returns refreshed dataset.
    """
    dataset.target_column = target
    dataset.sensitive_columns = sensitive or []
    dataset.status = status
    db.commit()
    db.refresh(dataset)
    return dataset


@router.post("/{dataset_id}/suggest_columns", response_model=ColumnSuggestion)
async def suggest_columns(dataset_id: int, db: Session = Depends(get_db)):
    # Fetch dataset from DB
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load CSV (first 1000 rows for dtype inference)
    df = pd.read_csv(dataset.file_path, nrows=1000)

    # Map column names to generic types with sample values
    columns_info = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().unique()[:3].tolist()

        if "int" in dtype or "float" in dtype:
            col_type = "numeric"
        elif "object" in dtype:
            col_type = "categorical"
        elif "bool" in dtype:
            col_type = "boolean"
        else:
            col_type = "other"

        columns_info[col] = {
            "type": col_type,
            "samples": sample_values
        }

    # Prepare LLM prompt
    prompt = f"""
    You are an expert data science assistant specializing in machine learning dataset analysis. Analyze the provided dataset schema and identify key columns for predictive modeling.

    # DATASET SCHEMA
    {json.dumps(columns_info, indent=2)}

    # YOUR TASKS

    ## 1. Identify Target Column
    - The target column is the dependent variable you want to predict
    - Look for columns representing outcomes, labels, prices, scores, or classifications
    - Common patterns: price, cost, value, amount, rating, score, class, category, label, target, outcome, result, churn, fraud, diagnosis
    - Consider semantic meaning from column names
    - If multiple candidates exist, choose the most likely prediction target
    - Return `null` only if no clear target exists

    ## 2. Identify Sensitive/Protected Columns
    - Columns containing personally identifiable information (PII) or protected attributes
    - Include: names, addresses, emails, phone numbers, SSN, IDs, dates of birth
    - Include: demographic data like race, ethnicity, gender, age, religion, national origin, disability status, marital status
    - Financial: credit card numbers, account numbers, salary (if individual-level)
    - If no sensitive columns exist, return empty array

    # OUTPUT REQUIREMENTS
    - Respond with ONLY a valid JSON object
    - No markdown formatting, no explanations, no additional text
    - Strict format:

    {{
    "target_column": "column_name_or_null",
    "sensitive_columns": ["column1", "column2"]
    }}

    Begin analysis now.
    """

    # Call LLM
    llm_response = call_llm(prompt, temperature=0.1)

    # Parse response with robust error handling
    try:
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```"):
            lines = cleaned_response.split("\n")
            json_lines = [l for l in lines if not l.startswith("```")]
            cleaned_response = "\n".join(json_lines).strip()

        suggestions = json.loads(cleaned_response)

        if not isinstance(suggestions, dict):
            raise ValueError("Response is not a JSON object")

        target = suggestions.get("target_column")
        sensitive = suggestions.get("sensitive_columns", [])
        if target is not None and not isinstance(target, str):
            target = None
        if not isinstance(sensitive, list):
            sensitive = []

        # Persist suggestions by calling the shared helper (updates DB)
        _update_dataset_columns(
            dataset, target, sensitive, db, status="suggested")

        return ColumnSuggestion(
            target_column=target,
            sensitive_columns=sensitive
        )

    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: return empty suggestion (no DB change)
        return ColumnSuggestion(
            target_column=None,
            sensitive_columns=[]
        )


@router.post("/{dataset_id}/update_columns", response_model=DatasetResponse)
async def update_columns(dataset_id: int, request: ConfirmColumnsRequest, db: Session = Depends(get_db)):
    """
    Explicit endpoint to update target and sensitive columns.
    This can be called by clients or internally after suggestions.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate columns exist in the actual file
    df = pd.read_csv(dataset.file_path, nrows=1)
    all_columns = df.columns.tolist()

    if request.target_column not in all_columns:
        raise HTTPException(
            status_code=400, detail=f"Target column '{request.target_column}' does not exist in dataset. Available: {all_columns}")
    for col in request.sensitive_columns:
        if col not in all_columns:
            raise HTTPException(
                status_code=400, detail=f"Sensitive column '{col}' does not exist in dataset. Available: {all_columns}")

    # Persist using helper
    dataset = _update_dataset_columns(
        dataset, request.target_column, request.sensitive_columns, db, status="confirmed")

    # Build columns list from stored schema
    columns = [ColumnSchema(**col) for col in (dataset.schema or [])]

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        file_type=dataset.file_type,
        file_path=dataset.file_path,
        row_count=dataset.row_count,
        target_column=dataset.target_column,
        sensitive_columns=dataset.sensitive_columns or [],
        columns=columns,
        status=dataset.status
    )


@router.put("/{dataset_id}/confirm", response_model=DatasetResponse)
async def confirm_columns(
    dataset_id: int,
    request: Optional[ConfirmColumnsRequest] = Body(None),
    db: Session = Depends(get_db),
):
    """
    If no body is provided -> return the current suggestion (target + sensitive) so the UI can show it.
    If a ConfirmColumnsRequest is provided -> validate, persist the corrected values, and return the updated DatasetResponse.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # If client only wants to view current suggestion
    if request is None:
        columns = [ColumnSchema(**col) for col in (dataset.schema or [])]
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            file_type=dataset.file_type,
            file_path=dataset.file_path,
            row_count=dataset.row_count,
            target_column=dataset.target_column,
            sensitive_columns=dataset.sensitive_columns or [],
            columns=columns,
            status=dataset.status or "unknown",
        )

    # Validate provided column names against the actual file
    df = pd.read_csv(dataset.file_path, nrows=1)
    all_columns = df.columns.tolist()

    if request.target_column not in all_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{request.target_column}' does not exist in dataset. Available: {all_columns}",
        )
    for col in request.sensitive_columns:
        if col not in all_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Sensitive column '{col}' does not exist in dataset. Available: {all_columns}",
            )

    # Persist corrections and return updated dataset
    dataset = await _update_dataset_columns(dataset, request.target_column, request.sensitive_columns, db, status="confirmed")

    columns = [ColumnSchema(**col) for col in (dataset.schema or [])]
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        file_type=dataset.file_type,
        file_path=dataset.file_path,
        row_count=dataset.row_count,
        target_column=dataset.target_column,
        sensitive_columns=dataset.sensitive_columns or [],
        columns=columns,
        status=dataset.status,
    )
