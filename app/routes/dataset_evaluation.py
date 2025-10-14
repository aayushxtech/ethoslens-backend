from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
from typing import List, Optional
import re
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.columns import DatasetColumn
from app.schemas.dataset_schema import DatasetEvaluationResponse
from app.schemas.column_schema import ColumnResponse
from app.utils.data_loader import load_dataframe as load_data
from app.utils.eval_utils import run_data_quality_tests
from app.utils.llm_utils import call_llm
from app.utils.bias_utils import BiasMetrics

router = APIRouter()


@router.get("/datasets/{dataset_id}/evaluation", response_model=DatasetEvaluationResponse)
def get_dataset_evaluation(dataset_id: int, db: Session = Depends(get_db)):
    """
    Load dataset, run data quality / fairness tests and return a basic evaluation.
    Optionally persist evaluation results to the database.
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

    # OPTIONAL: Persist evaluation results to the dataset
    try:
        # Convert test_results to JSON and store in dataset
        dataset.last_evaluation = {
            "timestamp": datetime.now().isoformat(),
            "tests": [
                {
                    "test_name": tr.test_name,
                    "status": tr.status,
                    "details": tr.details,
                    "suggestion": tr.suggestion
                }
                for tr in test_results
            ]
        }
        db.commit()
    except Exception as e:
        print(f"Warning: Could not persist evaluation results: {e}")
        # Don't fail the request if persistence fails

    # Extract numeric_stats details (if present) to attach per-column stats
    numeric_stats: dict = {}
    for tr in test_results:
        if getattr(tr, "test_name", None) == "numeric_stats" and tr.details:
            numeric_stats = tr.details or {}
            break

    # Fetch columns from DatasetColumn table instead of dataset.schema
    dataset_columns = db.query(DatasetColumn).filter(
        DatasetColumn.dataset_id == dataset_id
    ).all()

    columns: List[ColumnResponse] = []
    total_rows = int(dataset.row_count or len(df) if df is not None else 0)

    for col in dataset_columns:
        try:
            name = col.name
            column_type = col.column_type

            missing_count = col.missing_count or 0
            missing_percentage = col.missing_percentage or 0.0

            unique_count = col.unique_count or 0

            # Use numeric_stats (from eval_utils) when available for this column,
            # otherwise fall back to stored stats in the DatasetColumn
            stats = {}
            if isinstance(numeric_stats, dict) and name in numeric_stats:
                stats = numeric_stats.get(name, {})
            else:
                stats = col.stats or {}

            columns.append(ColumnResponse(
                name=name,
                column_type=column_type,
                missing_count=missing_count,
                missing_percentage=float(missing_percentage),
                unique_count=unique_count,
                stats=stats,
                is_target=col.is_target,  # Read directly from DatasetColumn
                is_sensitive=col.is_sensitive  # Read directly from DatasetColumn
            ))
        except Exception as e:
            print(f"Error processing column {col.name}: {e}")
            continue

    # Return DatasetEvaluationResponse (basic fields + columns)
    return DatasetEvaluationResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        total_rows=total_rows,
        columns=columns,
        tests=test_results,
    )


@router.post("/datasets/{dataset_id}/predict_target_column")
def predict_target_column(dataset_id: int, user_input: str, db: Session = Depends(get_db)):
    """
    Predict the target column of a dataset using an LLM based on user input.
    Update the target_column in the dataset and set is_target=True in dataset_columns table.
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Extract column names from DatasetColumn table instead of schema
    dataset_columns = db.query(DatasetColumn).filter(
        DatasetColumn.dataset_id == dataset_id
    ).all()

    if not dataset_columns:
        raise HTTPException(
            status_code=400, detail="No columns found in the dataset")

    columns = [col.name for col in dataset_columns]

    # Prepare the LLM prompt
    prompt = (
        f"You are an AI assistant. Based on the following dataset columns: {', '.join(columns)}, "
        f"and the user's input: '{user_input}', predict the most likely target column. "
        "The target column is the one that the user wants to predict or analyze. "
        "Make sure to respond with only the column name, without any additional text."
    )

    # Call the LLM
    try:
        predicted_column = call_llm(prompt).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {e}")

    # Validate that the predicted column exists in the dataset
    if predicted_column not in columns:
        raise HTTPException(
            status_code=400,
            detail=f"Predicted column '{predicted_column}' not found in dataset columns"
        )

    # Update the dataset's target_column
    try:
        dataset.target_column = predicted_column

        # Reset is_target for all columns in this dataset
        db.query(DatasetColumn).filter(
            DatasetColumn.dataset_id == dataset_id
        ).update({"is_target": False}, synchronize_session=False)

        # Set is_target=True for the predicted column
        updated_count = db.query(DatasetColumn).filter(
            DatasetColumn.dataset_id == dataset_id,
            DatasetColumn.name == predicted_column
        ).update({"is_target": True}, synchronize_session=False)

        if updated_count == 0:
            raise Exception(
                f"Column '{predicted_column}' not found in dataset_columns table")

        # Commit the changes to the database
        db.commit()
        db.refresh(dataset)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error updating target column: {e}")

    # Return only the predicted column name
    return predicted_column


@router.post("/datasets/{dataset_id}/find_potential_columns")
def find_potential_columns(dataset_id: int, target_column: str, dataset_analysis: dict, db: Session = Depends(get_db)):
    """
    Use an LLM to find potential columns related to the target column based on dataset analysis.
    Mark EXCLUDED features as sensitive (is_sensitive=True) in the dataset_columns table.
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Extract columns and dataset details
    columns = dataset_analysis.get("columns", [])
    if not columns:
        raise HTTPException(
            status_code=400, detail="No columns found in the dataset analysis")

    # Get actual column names from DatasetColumn table
    dataset_columns = db.query(DatasetColumn).filter(
        DatasetColumn.dataset_id == dataset_id
    ).all()

    column_names_in_db = {col.name for col in dataset_columns}

    # Prepare the LLM prompt
    column_names = [col["name"] for col in columns]
    prompt = (
        "You are an expert assistant for feature selection.\n\n"
        f"TARGET COLUMN: {target_column}\n\n"
        f"AVAILABLE COLUMNS: {', '.join(column_names)}\n\n"
        "DATASET ANALYSIS:\n"
        f"{json.dumps(dataset_analysis, indent=2)}\n\n"
        "TASK: Analyze the dataset and identify which columns are most relevant for predicting the target column.\n"
        "Consider statistical relationships, data quality, predictive potential, feature types and distributions.\n\n"
        "OUTPUT INSTRUCTIONS (STRICT):\n"
        "- Return ONLY a single valid JSON object, with no markdown fences or any extra commentary.\n"
        "- JSON schema must be exactly:\n"
        '{'
        '"selected_features": [\"col1\", \"col2\", ...],\n'
        '"reasoning": {\"col1\": \"reason\", \"col2\": \"reason\", ...},\n'
        '"excluded_features\": [\"colX\", ...],\n'
        '"exclusion_reasons\": \"brief explanation\"\n'
        '}\n'
        "Order features by relevance (most relevant first). Keep reasoning concise."
    )

    # Call the LLM
    try:
        response = call_llm(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {e}")

    # Try to extract JSON object from the LLM response robustly
    parsed_json = None
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if m:
            parsed_json = json.loads(m.group(0))
    except Exception:
        parsed_json = None

    # Fallback: try to parse simple comma/newline separated list of columns
    if not parsed_json:
        cleaned = response.strip()
        # Remove any markdown fences if present
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL).strip()
        # split on newlines or commas and keep known column names
        parts = [p.strip().strip('"\'')
                 for p in re.split(r'[\n,]+', cleaned) if p.strip()]
        selected = [p for p in parts if p in column_names]
        parsed_json = {
            "selected_features": selected,
            "reasoning": {},
            "excluded_features": [c for c in column_names if c not in selected],
            "exclusion_reasons": ""
        }

    # Normalize selected_features and excluded_features
    selected_features = parsed_json.get("selected_features") or []
    if isinstance(selected_features, str):
        selected_features = [selected_features]
    selected_features = [s for s in selected_features if isinstance(s, str)]

    excluded_features = parsed_json.get("excluded_features") or []
    if isinstance(excluded_features, str):
        excluded_features = [excluded_features]
    excluded_features = [s for s in excluded_features if isinstance(s, str)]

    # Filter to only columns that exist in the database
    excluded_features = [
        s for s in excluded_features if s in column_names_in_db]

    exclusion_reasons = parsed_json.get(
        "exclusion_reasons", "Excluded from feature selection")

    # Update the dataset and dataset_columns tables
    try:
        # Update sensitive_columns in the datasets table with EXCLUDED features
        existing_sensitive_columns = dataset.sensitive_columns or []
        if isinstance(existing_sensitive_columns, str):
            try:
                existing_sensitive_columns = json.loads(
                    existing_sensitive_columns)
            except Exception:
                existing_sensitive_columns = []

        updated_sensitive_columns = list(
            set(existing_sensitive_columns + excluded_features))
        dataset.sensitive_columns = updated_sensitive_columns

        # Reset is_sensitive for all columns first (optional)
        db.query(DatasetColumn).filter(
            DatasetColumn.dataset_id == dataset_id
        ).update({"is_sensitive": False, "reason": None}, synchronize_session=False)

        # Update is_sensitive=True and reason for each EXCLUDED feature
        updated_count = 0
        for col_name in excluded_features:
            count = db.query(DatasetColumn).filter(
                DatasetColumn.dataset_id == dataset_id,
                DatasetColumn.name == col_name
            ).update({
                "is_sensitive": True,
                "reason": exclusion_reasons
            }, synchronize_session=False)
            updated_count += count

        print(
            f"Updated {updated_count} columns out of {len(excluded_features)} excluded features as sensitive")

        # Commit the changes to the database
        db.commit()
        db.refresh(dataset)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error updating sensitive columns: {e}")

    # Return the parsed JSON from the LLM (guaranteed to be a JSON-serializable dict)
    return parsed_json


@router.post("/datasets/{dataset_id}/analyze_bias")
def analyze_bias_for_sensitive_columns(
    dataset_id: int,
    max_columns: int = 5,  # Limit number of columns analyzed
    sample_size: int = 5000,  # Sample size for large datasets
    db: Session = Depends(get_db)
):
    """
    Perform comprehensive bias analysis on columns marked as sensitive (is_sensitive=True).
    Analyzes bias between each sensitive column and the target column.

    Args:
        dataset_id: ID of the dataset
        max_columns: Maximum number of sensitive columns to analyze (default: 5)
        sample_size: Maximum rows to sample for analysis (default: 5000)

    Returns:
        Dictionary with bias analysis results for each sensitive column
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if target column is set
    if not dataset.target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column not set. Please predict or set target column first."
        )

    # Load the dataset
    try:
        df = load_data(dataset.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading dataset: {e}")

    # Sample large datasets to improve performance
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows from dataset of {len(df)} rows")

    # Validate target column exists
    if dataset.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{dataset.target_column}' not found in dataset"
        )

    # Fetch columns marked as sensitive (limit to max_columns)
    sensitive_columns = db.query(DatasetColumn).filter(
        DatasetColumn.dataset_id == dataset_id,
        DatasetColumn.is_sensitive == True
    ).limit(max_columns).all()

    if not sensitive_columns:
        raise HTTPException(
            status_code=400,
            detail="No sensitive columns found. Please run find_potential_columns first."
        )

    # Get all numeric feature columns for distribution analysis (limit to 2)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols
                    if col != dataset.target_column
                    # Limit to 2 features
                    and col not in [sc.name for sc in sensitive_columns]][:2]

    # Run bias analysis for each sensitive column
    bias_results = {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "target_column": dataset.target_column,
        "total_samples": len(df),
        "sampled": len(df) < int(dataset.row_count or 0) if dataset.row_count else False,
        "sensitive_columns_analyzed": [],
        "overall_bias_detected": False,
        "analyses": {}
    }

    for sensitive_col in sensitive_columns:
        col_name = sensitive_col.name

        # Validate column exists in dataframe
        if col_name not in df.columns:
            bias_results["analyses"][col_name] = {
                "error": f"Column '{col_name}' not found in dataset"
            }
            continue

        # Skip if column has too many unique values (high cardinality)
        unique_count = df[col_name].nunique()
        if unique_count > 50:
            bias_results["analyses"][col_name] = {
                "error": f"Column has too many unique values ({unique_count}). Bias analysis skipped.",
                "unique_count": unique_count
            }
            bias_results["sensitive_columns_analyzed"].append({
                "name": col_name,
                "reason": sensitive_col.reason or "No reason provided",
                "skipped": True
            })
            continue

        bias_results["sensitive_columns_analyzed"].append({
            "name": col_name,
            "reason": sensitive_col.reason or "No reason provided",
            "skipped": False
        })

        try:
            # Run lightweight bias analysis (no visualization, limited features)
            analysis_result = BiasMetrics.run_comprehensive_bias_analysis(
                df=df,
                sensitive_col=col_name,
                target_col=dataset.target_column,
                prediction_col=None,
                feature_cols=feature_cols  # Only 2 features
            )

            bias_results["analyses"][col_name] = analysis_result

            # Check if bias was detected
            if analysis_result.get("overall_bias_detected", False):
                bias_results["overall_bias_detected"] = True

        except Exception as e:
            bias_results["analyses"][col_name] = {
                "error": f"Error analyzing bias: {str(e)}"
            }

    return bias_results


@router.post("/datasets/{dataset_id}/analyze_bias_single")
def analyze_bias_single_column(
    dataset_id: int,
    sensitive_column: str,
    feature_columns: Optional[List[str]] = None,
    db: Session = Depends(get_db)
):
    """
    Perform bias analysis on a specific sensitive column.

    Args:
        dataset_id: ID of the dataset
        sensitive_column: Specific sensitive column to analyze
        feature_columns: Optional list of feature columns for distribution analysis

    Returns:
        Bias analysis results for the specified column
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if target column is set
    if not dataset.target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column not set. Please predict or set target column first."
        )

    # Load the dataset
    try:
        df = load_data(dataset.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading dataset: {e}")

    # Validate columns exist
    if sensitive_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Sensitive column '{sensitive_column}' not found in dataset"
        )

    if dataset.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{dataset.target_column}' not found in dataset"
        )

    # Get column info from database
    sensitive_col_info = db.query(DatasetColumn).filter(
        DatasetColumn.dataset_id == dataset_id,
        DatasetColumn.name == sensitive_column
    ).first()

    # If no feature columns provided, auto-detect numeric columns
    if not feature_columns:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_columns = [col for col in numeric_cols
                           if col != dataset.target_column
                           and col != sensitive_column][:3]  # Limit to 3

    # Validate feature columns
    valid_feature_cols = [col for col in (
        feature_columns or []) if col in df.columns]

    try:
        # Run comprehensive bias analysis
        bias_result = BiasMetrics.run_comprehensive_bias_analysis(
            df=df,
            sensitive_col=sensitive_column,
            target_col=dataset.target_column,
            prediction_col=None,
            feature_cols=valid_feature_cols
        )

        # Add metadata
        bias_result["dataset_id"] = dataset.id
        bias_result["dataset_name"] = dataset.name
        bias_result["column_reason"] = sensitive_col_info.reason if sensitive_col_info else None
        bias_result["is_marked_sensitive"] = sensitive_col_info.is_sensitive if sensitive_col_info else False

        return bias_result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing bias: {str(e)}"
        )


@router.post("/datasets/{dataset_id}/generate_report")
def generate_comprehensive_report(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Generate a comprehensive AI-powered bias and fairness report for a dataset.
    Collects all relevant information from the database and uses LLM to analyze and summarize.z

    Args:
        dataset_id: ID of the dataset

    Returns:
        Dictionary with comprehensive report including LLM analysis
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Fetch all dataset columns
    dataset_columns = db.query(DatasetColumn).filter(
        DatasetColumn.dataset_id == dataset_id
    ).all()

    if not dataset_columns:
        raise HTTPException(
            status_code=400, detail="No columns found for this dataset")

    # Collect dataset metadata
    report_data = {
        "dataset_info": {
            "id": dataset.id,
            "name": dataset.name,
            "file_type": dataset.file_type,
            "row_count": dataset.row_count,
            "column_count": len(dataset_columns),
            "target_column": dataset.target_column,
            "sensitive_columns": dataset.sensitive_columns or [],
            "status": dataset.status,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None
        },
        "columns": [],
        "target_column_info": None,
        "sensitive_columns_info": [],
        "evaluation_results": None,
        "bias_summary": {}
    }

    # Process each column
    target_col = None
    sensitive_cols = []

    for col in dataset_columns:
        col_info = {
            "name": col.name,
            "type": col.column_type,
            "missing_count": col.missing_count,
            "missing_percentage": col.missing_percentage,
            "unique_count": col.unique_count,
            "is_target": col.is_target,
            "is_sensitive": col.is_sensitive,
            "reason": col.reason,
            "stats": col.stats or {}
        }

        report_data["columns"].append(col_info)

        if col.is_target:
            target_col = col_info
            report_data["target_column_info"] = col_info

        if col.is_sensitive:
            sensitive_cols.append(col_info)
            report_data["sensitive_columns_info"].append(col_info)

    # Get last evaluation if available
    if hasattr(dataset, 'last_evaluation') and dataset.last_evaluation:
        report_data["evaluation_results"] = dataset.last_evaluation

    # Load dataset for basic statistics
    try:
        df = load_data(dataset.file_path, nrows=1000)

        # Quick bias check summary
        if target_col and sensitive_cols:
            for sens_col in sensitive_cols[:3]:  # Limit to 3 for performance
                col_name = sens_col["name"]
                if col_name in df.columns and dataset.target_column in df.columns:
                    try:
                        # Quick demographic parity check
                        unique_count = df[col_name].nunique()
                        if unique_count <= 50:  # Only if reasonable cardinality
                            selection_rates = {}
                            for group in df[col_name].unique():
                                group_data = df[df[col_name] == group]
                                if len(group_data) > 0:
                                    # Assume binary target or calculate mean
                                    rate = float(
                                        group_data[dataset.target_column].mean())
                                    selection_rates[str(group)] = rate

                            if selection_rates:
                                rates = list(selection_rates.values())
                                disparate_impact = min(
                                    rates) / max(rates) if max(rates) > 0 else 1.0

                                report_data["bias_summary"][col_name] = {
                                    "selection_rates": selection_rates,
                                    "disparate_impact": float(disparate_impact),
                                    "bias_detected": disparate_impact < 0.8
                                }
                    except Exception as e:
                        print(f"Error calculating bias for {col_name}: {e}")

    except Exception as e:
        print(f"Warning: Could not load dataset for statistics: {e}")

    # Prepare comprehensive prompt for LLM
    # Build a compact DB context from datasets and dataset_columns to pass to the LLM
    db_context = {
        "dataset": report_data["dataset_info"],
        "columns": [
            {
                "name": c["name"],
                "type": c["type"],
                "unique_count": c["unique_count"],
                "missing_percentage": c["missing_percentage"],
                "is_target": c["is_target"],
                "is_sensitive": c["is_sensitive"],
                "reason": c.get("reason"),
            }
            for c in report_data["columns"]
        ],
        "quick_bias": report_data.get("bias_summary", {}),
    }

    db_context_snippet = json.dumps(db_context, indent=2)

    prompt = (
        "You are an AI ethics and fairness expert. Use the DATABASE CONTEXT below (from tables "
        "'datasets' and 'dataset_columns') to generate a concise, actionable bias & fairness report.\n\n"
        "DATABASE CONTEXT:\n"
        f"{db_context_snippet}\n\n"
        "TASK: Produce a JSON report with these fields:\n"
        '{'
        '"executive_summary": "Brief overview (2-3 sentences)",'
        '"data_quality_assessment": {"overall_quality":"excellent|good|fair|poor","key_issues":[],"recommendations":[]},'
        '"bias_analysis": {"overall_bias_risk":"low|medium|high|critical","sensitive_features_analysis":"",'
        '"disparate_impact_assessment":"","key_concerns":[],"fairness_recommendations":[]},'
        '"target_column_analysis":{"distribution":"","predictability_assessment":"","potential_issues":[]},'
        '"ethical_considerations":{"privacy_concerns":[],"discrimination_risks":[],"transparency_recommendations":[]},'
        '"actionable_insights":[],"overall_recommendation":"approve|approve with caution|reject|needs more analysis"'
        '}\n\n'
        "Important: Return ONLY a single valid JSON object and nothing else."
    )

    # Call LLM to generate report
    try:
        llm_response = call_llm(prompt)

        # Try to parse JSON response
        try:
            # Extract JSON from response
            match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if match:
                llm_report = json.loads(match.group(0))
            else:
                llm_report = {
                    "error": "Could not parse LLM response as JSON", "raw_response": llm_response}
        except Exception as e:
            llm_report = {
                "error": f"JSON parsing error: {str(e)}", "raw_response": llm_response}

    except Exception as e:
        llm_report = {"error": f"LLM call failed: {str(e)}"}

    # Combine all information into final report
    final_report = {
        "report_metadata": {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0"
        },
        "dataset_summary": report_data["dataset_info"],
        "column_analysis": {
            "total_columns": len(report_data["columns"]),
            "target_column": report_data["target_column_info"],
            "sensitive_columns": report_data["sensitive_columns_info"],
            "all_columns": report_data["columns"]
        },
        "quick_bias_metrics": report_data["bias_summary"],
        "ai_generated_report": llm_report,
        "evaluation_history": report_data["evaluation_results"]
    }

    return final_report


@router.get("/datasets/{dataset_id}/report/summary")
def get_report_summary(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a quick summary of dataset without generating full LLM report.
    Useful for dashboard views.

    Args:
        dataset_id: ID of the dataset

    Returns:
        Quick summary with key metrics
    """
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Fetch columns
    dataset_columns = db.query(DatasetColumn).filter(
        DatasetColumn.dataset_id == dataset_id
    ).all()

    # Count sensitive and target columns
    sensitive_count = sum(1 for col in dataset_columns if col.is_sensitive)
    target_set = any(col.is_target for col in dataset_columns)

    # Calculate data quality score (simple heuristic)
    total_missing = sum(col.missing_percentage or 0 for col in dataset_columns)
    avg_missing = total_missing / \
        len(dataset_columns) if dataset_columns else 0

    quality_score = "excellent" if avg_missing < 5 else "good" if avg_missing < 15 else "fair" if avg_missing < 30 else "poor"

    summary = {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "row_count": dataset.row_count,
        "column_count": len(dataset_columns),
        "target_column": dataset.target_column,
        "target_column_set": target_set,
        "sensitive_columns_count": sensitive_count,
        "sensitive_columns": [col.name for col in dataset_columns if col.is_sensitive],
        "data_quality_score": quality_score,
        "average_missing_percentage": round(avg_missing, 2),
        "status": dataset.status,
        "has_evaluation": hasattr(dataset, 'last_evaluation') and dataset.last_evaluation is not None,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None
    }

    return summary
