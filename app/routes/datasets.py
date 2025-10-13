from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.dataset import process_upload
from app.db.session import get_db
from app.models.dataset import Dataset
from app.schemas.dataset_schema import DatasetResponse, ColumnSchema, ColumnSuggestion, ConfirmColumnsRequest

from app.utils.file_utils import allowed_file
from app.utils.llm_utils import call_llm

import pandas as pd
import json

router = APIRouter()


@router.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db), user_id: int = 1):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    dataset = process_upload(file, db=db, uploaded_by=user_id)

    # Convert schema JSON to ColumnSchema list
    columns = [ColumnSchema(**col) for col in dataset.schema]

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


@router.post("/datasets/{dataset_id}/suggest_columns", response_model=list[ColumnSuggestion])
async def suggest_columns(dataset_id: int, db: Session = Depends(get_db)):
    # Fetch dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load File
    df = pd.read_csv(dataset.file_path, nrows=1000)

    # Cols Info
    columns_info = {col: df[col].dropna().unique()[:5].tolist()
                    for col in df.columns}

    # Prompt
    prompt = f"""
    You are an expert AI data scientist assistant. Your task is to analyze the schema and sample data from a dataset and provide structured insights for machine learning purposes.

    ## CONTEXT
    Here is a preview of the dataset, containing column names and sample values:
    ```json
    {json.dumps(columns_info, indent=2)}
    ```

    Based on the provided context, perform the following two tasks:

    Identify the Target Column: Determine the most probable target column for a predictive model. A target column (or label) is the variable you want to predict. It often represents an outcome, a classification, or a value of interest (e.g., 'churn', 'sale_price', 'is_fraud'). If no column appears to be a clear target, specify null.

    Identify Sensitive Columns: List all columns that could be considered sensitive. These columns typically relate to demographics or protected characteristics that might introduce bias into a model and require fairness assessments (e.g., race, gender, age, national_origin, disability). If no columns appear to be sensitive, return an empty list.

    Your response must be a single, valid JSON object. Do not include any text or explanations outside of the JSON structure.
    {{
        "target_column":"column_name"
        "sensitive_columns": ["column_name"]
    }}
    """

    # Call LLM
    llm_response = call_llm(prompt)

    try:
        suggestions = json.loads(llm_response)
    except Exception:
        suggestions = {"target_column": None, "sensitive_columns": []}


@router.post("/datasets/{dataset_id}/confirm_columns", response_model=DatasetResponse)
async def confirm_columns(dataset_id: int, request: ConfirmColumnsRequest, db: Session = Depends(get_db)):
    # Fetch dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate columns
    import pandas as pd
    df = pd.read_csv(dataset.file_path, nrows=1)
    all_columns = df.columns.tolist()

    if request.target_column not in all_columns:
        raise HTTPException(
            status_code=400, detail="Target column does not exist in dataset")
    for col in request.sensitive_columns:
        if col not in all_columns:
            raise HTTPException(
                status_code=400, detail=f"Sensitive column '{col}' does not exist in dataset")

    # Update dataset
    dataset.target_column = request.target_column
    dataset.sensitive_columns = ",".join(request.sensitive_columns)
    dataset.status = "confirmed"
    db.commit()
    db.refresh(dataset)

    # Return response as list, not CSV
    return {
        "message": "Columns confirmed successfully",
        "dataset_id": dataset.id,
        "target_column": dataset.target_column,
        "sensitive_columns": request.sensitive_columns
    }


# Additional endpoints for listing datasets, retrieving details, updating metadata, and deleting datasets can be added here.
