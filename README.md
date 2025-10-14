# EthosLens Backend

Backend for EthosLens built with FastAPI. Provides user auth (register/login), JWTs, dataset upload + schema analysis (LLM-assisted), dataset evaluation, and a simple SQLAlchemy/Postgres setup for an MVP.

---

## Table of Contents
- Requirements
- Quick setup
- Environment variables
- Database initialization
- Run the server
- API Reference (all endpoints)
- Notes / Troubleshooting

---

## Quick setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Create & activate venv (Linux/macOS):
```bash
python3 -m venv venv
source venv/bin/activate
```

Run server:
```bash
uvicorn app.main:app --reload
```

Open API docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## Environment

Create a `.env` file at project root. Example:
```env
DATABASE_URL="postgresql://<username>:<password>@<host>/<database>?sslmode=require"
SECRET_KEY="replace_with_secure_random"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
GROQ_API_KEY="gsk_xxx"   # required for LLM suggestions
```

Settings are read from `app/config.py`.

---

## Database initialization

Create tables:
```bash
python -m app.init_db
# or
python migrate.py
```

---

## API Reference — Endpoints

Base: http://127.0.0.1:8000

All dataset endpoints are logically under `/datasets` (some route files include full paths). Authentication under `/auth`, posts under `/posts`.

Auth
- POST /auth/register  
  - Body (JSON): { "username": str, "email": str, "password": str }  
  - Success: 200 { "message": "User registered successfully" }

- POST /auth/login  
  - Body (JSON): { "email": str, "password": str }  
  - Success: 200 -> Token model: { "access_token": "<jwt>", "token_type": "bearer" }

Datasets
- POST /datasets/upload  
  - multipart/form-data: file (csv|xlsx|json|parquet|zip)  
  - Saves file, analyzes preview, creates Dataset row.  
  - Response: DatasetResponse (see schemas) with stored schema array.

- POST /datasets/{dataset_id}/suggest_columns  
  - No body. Calls LLM to suggest target and sensitive columns based on preview.  
  - Persists suggestions (status="suggested").  
  - Response: ColumnSuggestion { "target_column": str|null, "sensitive_columns": [str] }

- POST /datasets/{dataset_id}/update_columns  
  - Body (JSON): ConfirmColumnsRequest { "target_column": str, "sensitive_columns": [str] }  
  - Validates columns exist in uploaded file, persists as confirmed.  
  - Response: DatasetResponse (updated)

- PUT /datasets/{dataset_id}/confirm  
  - If no body: returns current suggestion + dataset schema (DatasetResponse).  
  - If body (ConfirmColumnsRequest): validate & persist corrections, return updated DatasetResponse.

- GET /datasets/{dataset_id}/evaluation  
  - Loads preview (up to 1000 rows), runs data quality + fairness tests and returns DatasetEvaluationResponse:  
    - dataset_id, dataset_name, total_rows, columns (ColumnResponse list)  
    - tests: List[TestResult] — includes missing_values, duplicate_rows, constant_columns, high_cardinality_columns, numeric_stats, skewness, kurtosis, fairness_disparity (when applicable).
  - Useful for UI to show per-column stats and quality/fairness warnings.

Posts
- POST /posts/  (create post)  
  - Body: PostCreate { title, content }  
  - Response: ResponsePost (created post)

- GET /posts/  (list posts)  
  - Response: list[ResponsePost]

- GET /posts/{post_id}  (single post)  
  - Response: ResponsePost

- PUT /posts/{post_id}  (update post)  
  - Body: PostUpdate (partial OK)  
  - Response: ResponsePost

- DELETE /posts/{post_id}  
  - Deletes post, returns 204 or success detail.

- POST /posts/{post_id}/upvote  
  - Increments upvotes, returns { upvotes, downvotes }

- POST /posts/{post_id}/downvote  
  - Increments downvotes, returns { upvotes, downvotes }

---

## Schemas (where to look)
- DatasetResponse, DatasetEvaluationResponse, TestResult: `app/schemas/dataset_schema.py`
- ColumnResponse / ColumnStats: `app/schemas/column_schema.py`
- Auth models: `app/schemas/auth_schema.py`
- Post models: `app/schemas/post_schema.py`

---

## What evaluations are run (run_data_quality_tests)
Implemented in `app/utils/eval_utils.py` — returns a list of TestResult entries:
- missing_values (per-column % missing; warning if >20%)
- duplicate_rows
- constant_columns
- high_cardinality_columns (unique > 100)
- numeric_stats (pandas .describe: count, mean, std, min, 25%, 50%, 75%, max)
- skewness (scipy.stats.skew)
- kurtosis (scipy.stats.kurtosis)
- fairness_disparity (requires target + sensitive column; computes group means and disparity_ratio)

---

## Notes / Troubleshooting
- Ensure `.env` contains DATABASE_URL and GROQ_API_KEY (if using LLM).  
- If you see SQLAlchemy mapper errors, make sure `import app.models` is executed (done in `app/main.py`).  
- For DB connection drops, enable pool_pre_ping in `app/db/session.py` (recommended) and check DB logs.  
- To include/exclude tests or change thresholds, edit `app/utils/eval_utils.py`.

---

## Development tips
- Use Swagger UI to test endpoints quickly.  
- Upload a CSV via `/datasets/upload`, then call `/datasets/{id}/suggest_columns` -> `/datasets/{id}/confirm` -> `/datasets/{id}/evaluation` to see full flow.

---

## Project structure (important files)
- app/main.py — FastAPI app and route registration  
- app/routes/*.py — API endpoints (auth, datasets, posts, evaluation)  
- app/services/*.py — business logic (upload processing)  
- app/models/*.py — SQLAlchemy models  
- app/schemas/*.py — Pydantic schemas  
- app/db/session.py — DB engine & session  
- app/utils/* — helpers (file handling, LLM, evaluation)

---

If you want, I can open and patch `README.md` in your workspace directly or add curl examples for each endpoint.
