# EthosLens Backend

Backend for EthosLens built with FastAPI. Provides user auth (register/login), JWTs, dataset upload + schema analysis (LLM-assisted), and a simple SQLAlchemy/Postgres setup for an MVP.

---

## Table of Contents
- Requirements
- Quick setup (Linux/macOS & Windows)
- Environment variables
- Database initialization
- Run the server
- API Reference
  - Auth
  - Datasets
- LLM notes
- Troubleshooting

---

## Requirements
- Python 3.10+ (3.13 recommended)
- PostgreSQL (shared dev DB)
- Git, a terminal

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Quick setup

Clone repo
```bash
git clone https://github.com/aayushxtech/ethoslens-backend.git
cd backend
```

Create & activate venv

Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

## Environment

Create a `.env` file at project root. Example:
```env
DATABASE_URL="postgresql://<username>:<password>@<host>/<database>?sslmode=require"
SECRET_KEY="replace_with_secure_random"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
GROQ_API_KEY="gsk_xxx"   # required if using LLM endpoints
```

.env is ignored by git.

Settings are read from `app/config.py` (Pydantic BaseSettings).

---

## Database initialization

Initialize the shared development DB (creates tables from SQLAlchemy models):

Linux / macOS
```bash
python3.13 -m app.init_db
```

Windows
```powershell
py -3.13 -m app.init_db
# or
python -m app.init_db
```

Alternative:
```bash
python migrate.py
```

For production use Alembic for versioned migrations.

---

## Run the server

Linux / macOS
```bash
uvicorn app.main:app --reload
```

Windows (PowerShell / cmd)
```powershell
uvicorn app.main:app --reload
```

Open API docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## API Reference

Base: http://127.0.0.1:8000

All dataset endpoints are prefixed with `/datasets` (router in `app/main.py`).

Schema models referenced below come from:
- `app/schemas/auth_schema.py`
- `app/schemas/dataset_schema.py`
- `app/schemas/column_schema.py`

### Auth

1) Register
- POST `/auth/register`
- Body (application/json):
```json
{
  "username": "test",
  "email": "test@gmail.com",
  "password": "test123@"
}
```
- Success: 200
```json
{ "message": "User registered successfully" }
```
- Errors:
  - 400 if email already exists
  - 422 if request body invalid

2) Login
- POST `/auth/login`
- Body:
```json
{
  "email": "test@gmail.com",
  "password": "test123@"
}
```
- Success: 200
```json
{
  "access_token": "<jwt_token>",
  "token_type": "bearer"
}
```
- Errors:
  - 401 invalid credentials
  - 422 validation error

### Datasets

1) Upload dataset
- POST `/datasets/upload`
- Form field: `file` (multipart/form-data)
- Supported types: csv, xlsx, json, parquet, zip (zip parsing not implemented)
- Response model: `DatasetResponse` (`app/schemas/dataset_schema.py`)
Example response (200):
```json
{
  "id": 1,
  "name": "housing.csv",
  "file_type": "csv",
  "file_path": "uploads/xxxx.csv",
  "row_count": 1000,
  "target_column": null,
  "sensitive_columns": [],
  "columns": [
    {
      "column_name": "Price",
      "column_type": "float64",
      "null_count": 0,
      "unique_count": 1000,
      "example_value": 1059033.558
    }
  ],
  "status": "pending"
}
```

2) Suggest target & sensitive columns (LLM)

- POST `/datasets/{dataset_id}/suggest_columns`
- No body required.
- Calls LLM to analyze dataset preview and returns `ColumnSuggestion`:

```json
{
  "target_column": "Price",
  "sensitive_columns": ["Address"]
}
```

- Behavior:
  - Persists suggestions to DB (`target_column`, `sensitive_columns`, status="suggested")
  - Requires `GROQ_API_KEY` configured for LLM (see LLM notes).
- Errors:
  - 404 dataset not found
  - 500 LLM / parsing errors (falls back to empty suggestion)

3) Update columns (explicit save)

- POST `/datasets/{dataset_id}/update_columns`
- Body (`ConfirmColumnsRequest`):

```json
{
  "target_column": "Price",
  "sensitive_columns": ["Address"]
}
```

- Validates columns exist in the uploaded file, persists them (`status="confirmed"`), returns `DatasetResponse`.
- Errors:
  - 400 column not present
  - 404 dataset not found

4) Confirm / review endpoint (GET/PUT behavior implemented as PUT)

- PUT `/datasets/{dataset_id}/confirm`
- If no body is provided: returns current suggestion and dataset schema (for UI preview).
- If body (ConfirmColumnsRequest) is provided: validates and persists corrections, returns updated `DatasetResponse`.
- Use this endpoint so users can correct LLM suggestions manually.

---

## LLM notes

- LLM client initialized in `app/utils/llm_utils.py` and reads `GROQ_API_KEY` via `app/config.py`.
- If GROQ_API_KEY missing, LLM calls will raise. For local testing you can:
  - Set GROQ_API_KEY in `.env`
  - Or mock `app.utils.llm_utils.call_llm` in tests

Prompt engineering and result parsing include robust JSON extraction; however, the LLM can still return unexpected formats — UI should present suggestions and allow user correction.

---

## Testing from Swagger UI

1. Start server: `uvicorn app.main:app --reload`
2. Open: http://127.0.0.1:8000/docs
3. Use `/datasets/upload` to upload a CSV.
4. Call `/datasets/{id}/suggest_columns` to get/save suggestions.
5. If suggestion is wrong, call `/datasets/{id}/confirm` (PUT) with corrected JSON.

---

## Troubleshooting

- Pydantic BaseSettings ValidationError: make sure keys in `.env` are declared in `app/config.py` or set `extra="allow"` in settings Config.
- 422 errors: request body doesn't match Pydantic schema — check field names and types.
- 405 Method Not Allowed: use correct HTTP method (e.g., suggest is POST).
- LLM errors: ensure `GROQ_API_KEY` present or mock calls during dev.
- DB connection errors: verify `DATABASE_URL` and that shared DB is reachable.

---

## Project structure (important files)

- app/main.py — FastAPI app and route registration
- app/routes/auth.py — auth endpoints
- app/routes/datasets.py — dataset upload / suggestion / confirm
- app/services/*.py — business logic
- app/models/*.py — SQLAlchemy models (Dataset, DatasetColumn, User)
- app/schemas/*.py — Pydantic schemas
- app/db/session.py — DB engine & session
- app/init_db.py / migrate.py — create tables
- app/utils/llm_utils.py — Groq LLM client wrapper
- app/utils/file_utils.py — file save / preview helpers
