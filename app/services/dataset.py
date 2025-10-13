from sqlalchemy.orm import Session
from app.models.dataset import Dataset
from app.utils.file_utils import save_file, parse_preview


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

    return dataset
