import os
import uuid
import pandas as pd

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json', 'parquet', 'zip'}
UPLOAD_DIRECTORY = 'uploads/'

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


def allowed_file(filename: str) -> bool:
    ext = filename.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def save_file(file) -> str:
    """Save uploaded file locally with a unique name and return path"""
    ext = file.filename.split(".")[-1]
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(UPLOAD_DIRECTORY, unique_name)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def parse_preview(file_path: str, n_rows: int = 1000):
    """Load first n rows for preview"""
    ext = file_path.split(".")[-1].lower()
    if ext == "csv":
        df = pd.read_csv(file_path, nrows=n_rows)
    elif ext == "json":
        df = pd.read_json(file_path, lines=True)
        df = df.head(n_rows)
    elif ext == "xlsx":
        df = pd.read_excel(file_path, nrows=n_rows)
    elif ext == "parquet":
        df = pd.read_parquet(file_path)
        df = df.head(n_rows)
    elif ext == "zip":
        # Later: extract first file inside ZIP
        raise NotImplementedError("ZIP parsing not implemented yet")
    else:
        raise ValueError("Unsupported file type")

    return df
