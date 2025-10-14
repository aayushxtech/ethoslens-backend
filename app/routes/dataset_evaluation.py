from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
from typing import List, Optional

from app.db.session import get_db
from app.models.dataset import Dataset
from app.schemas.dataset_schema import DatasetEvaluationResponse
from app.schemas.column_schema import ColumnResponse
from app.utils.data_loader import load_dataframe as load_data
from app.utils.eval_utils import run_data_quality_tests
from app.utils.llm_utils import call_llm

router = APIRouter()


@router.get("/datasets/{dataset_id}/evaluation", response_model=DatasetEvaluationResponse)
def get_dataset_evaluation(dataset_id: int, db: Session = Depends(get_db)):
    """
    Load dataset, run data quality / fairness tests and return a basic evaluation.
    Uses:
    - run_data_quality_tests from app.utils.eval_utils
    - load_dataframe from app.utils.data_loader
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load the file (preview up to 1000 rows)
    try:
        df = load_data(dataset.file_path, nrows=1000)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading dataset: {e}")

    # Prepare sensitive column (eval_utils expects a single sensitive column name optionally)
    sensitive_column: Optional[str] = None
    if dataset.sensitive_columns:
        # dataset.sensitive_columns stored as JSON/list in DB; handle both list and JSON-string
        if isinstance(dataset.sensitive_columns, str):
            try:
                sc_list = json.loads(dataset.sensitive_columns)
            except Exception:
                sc_list = []
        else:
            sc_list = dataset.sensitive_columns or []
        if isinstance(sc_list, list) and len(sc_list) > 0:
            sensitive_column = sc_list[0]

    # Run tests (returns list of TestResult)
    test_results = run_data_quality_tests(
        df,
        target_column=dataset.target_column,
        sensitive_column=sensitive_column,
    )

    # Extract numeric_stats details (if present) to attach per-column stats
    numeric_stats: dict = {}
    for tr in test_results:
        if getattr(tr, "test_name", None) == "numeric_stats" and tr.details:
            numeric_stats = tr.details or {}
            break

    # Build columns list from stored schema (safe fallback to empty)
    columns_data: List[dict] = dataset.schema or []
    columns: List[ColumnResponse] = []
    total_rows = int(dataset.row_count or len(df) if df is not None else 0)

    for col in columns_data:
        try:
            if not isinstance(col, dict):
                continue

            name = col.get("column_name") or col.get("name")
            column_type = col.get("column_type") or col.get(
                "type") or "unknown"

            missing_count = col.get("null_count")
            if missing_count is None:
                missing_count = col.get("missing_count", 0)
            try:
                missing_count = int(
                    missing_count) if missing_count is not None else 0
            except Exception:
                missing_count = 0

            missing_percentage = 0.0
            if total_rows:
                missing_percentage = (missing_count / total_rows) * 100

            unique_count = col.get("unique_count", 0)
            try:
                unique_count = int(unique_count)
            except Exception:
                unique_count = 0

            # Use numeric_stats (from eval_utils) when available for this column,
            # otherwise fall back to any stored stats in the schema.
            stats = {}
            if isinstance(numeric_stats, dict) and name in numeric_stats:
                stats = numeric_stats.get(name, {})
            else:
                stats = col.get("stats") or {}

            columns.append(ColumnResponse(
                name=name,
                column_type=column_type,
                missing_count=missing_count,
                missing_percentage=float(missing_percentage),
                unique_count=unique_count,
                stats=stats,
                is_target=(name == dataset.target_column),
                is_sensitive=(isinstance(dataset.sensitive_columns,
                              list) and name in dataset.sensitive_columns)
            ))
        except Exception:
            continue

    # Return DatasetEvaluationResponse (basic fields + columns)
    return DatasetEvaluationResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        total_rows=int(dataset.row_count or len(df)),
        columns=columns,
        tests=test_results,
    )


@router.post("/datasets/{dataset_id}/predict_target_column")
def predict_target_column(dataset_id: int, user_input: str, db: Session = Depends(get_db)):
    """
    Predict the target column of a dataset using an LLM based on user input.
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Extract column names
    columns = [col.get("column_name") for col in (dataset.schema or [])]
    if not columns:
        raise HTTPException(
            status_code=400, detail="No columns found in the dataset schema")

    # Prepare the LLM prompt
    prompt = (
        f"You are an AI assistant. Based on the following dataset columns: {', '.join(columns)}, "
        f"and the user's input: '{user_input}', predict the most likely target column. "
        "The target column is the one that the user wants to predict or analyze."
        "Make sure to respond with only the column name, without any additional text."
    )

    # Call the LLM
    try:
        predicted_column = call_llm(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {e}")

    # Return only the predicted column name
    return predicted_column
