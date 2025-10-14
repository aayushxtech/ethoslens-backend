from sqlalchemy.orm import Session
from app.models.dataset import Dataset
from app.utils.file_utils import save_file, parse_preview
from app.models.columns import DatasetColumn  # <- add this import
import pandas as pd


def process_upload(file, db: Session, uploaded_by: int = None):
    # Save file
    file_path = save_file(file)

    # Parse preview
    df = parse_preview(file_path)

    # Extract schema as JSON
    schema = []
    for col in df.columns:
        schema.append({
            "column_name": col,
            "column_type": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
            "example_value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else None
        })

    # Create Dataset entry
    dataset = Dataset(
        name=file.filename,
        file_type=file.filename.split(".")[-1].lower(),
        file_path=file_path,
        row_count=len(df),
        schema=schema,
        uploaded_by=uploaded_by
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # --- NEW: create DatasetColumn rows for each column in the schema ---
    columns_to_add = []
    for col in df.columns:
        series = df[col]
        # basic stats (describe for numeric) - keep JSON-serializable values
        stats = {}
        try:
            if pd.api.types.is_numeric_dtype(series):
                desc = series.dropna().describe().to_dict()
                # convert numpy types to Python native
                stats = {k: (float(v) if hasattr(v, "item") else v)
                         for k, v in desc.items()}
        except Exception:
            stats = {}

        missing_count = int(series.isnull().sum())
        missing_pct = float(missing_count / len(df) *
                            100) if len(df) > 0 else 0.0
        unique_count = int(series.nunique())

        columns_to_add.append(DatasetColumn(
            dataset_id=dataset.id,
            name=col,
            column_type=str(series.dtype),
            missing_count=missing_count,
            missing_percentage=missing_pct,
            unique_count=unique_count,
            stats=stats,
            is_target=False,
            is_sensitive=False
        ))

    if columns_to_add:
        db.add_all(columns_to_add)
        db.commit()
        # optional: refresh dataset to populate relationship if needed
        db.refresh(dataset)
    # --- end new code ---

    return dataset
