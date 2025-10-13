# EthosLens Backend

Backend for EthosLens built with FastAPI. Provides user auth (register/login), JWTs, and a simple SQLAlchemy/Postgres setup for an MVP.

---

## Table of Contents
- Features
- Requirements
- Quick setup (Linux/macOS & Windows)
- Environment
- Database initialization / migrations
- Running the server
- API docs & endpoints
- Testing examples
- Project structure

---

## Features
- User registration and login with hashed passwords (argon2).
- JWT-based authentication.
- PostgreSQL database via SQLAlchemy.
- Small, clear code structure suitable for an MVP.

---

## Requirements
- Python 3.13 recommended (works with 3.10+)
- PostgreSQL (shared dev DB)
- Git, a terminal
- Windows users: use PowerShell or Command Prompt

---

## Quick setup

Clone repo
```bash
git clone https://github.com/aayushxtech/ethoslens-backend.git
cd backend
```

Create virtual environment

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

Windows (cmd.exe)
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Install dependencies
```bash
pip install -r requirements.txt
```

---

## Environment

Create a `.env` file in the project root (this repo's .gitignore already ignores `.env`):

Example `.env`
```env
DATABASE_URL="postgresql://<username>:<password>@<host>/<database>?sslmode=require"
SECRET_KEY="replace_with_secure_random"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

- Use your shared dev DB connection string for DATABASE_URL.
- Keep `.env` out of source control.

---

## Database initialization / migrations

Dev workflow uses the shared DB. Use the provided init module to initialize schema:

Linux / macOS
```bash
python3.13 -m app.init_db
```

Windows (if Python 3.13 installed as `py -3.13`)
```powershell
py -3.13 -m app.init_db
# or if `python` points to 3.13
python -m app.init_db
```

Alternative one-off migration script (creates tables using SQLAlchemy metadata):
```bash
python migrate.py
# Windows:
python migrate.py
```

Notes:
- Both commands only create tables from SQLAlchemy models (MVP). For production, use Alembic for versioned migrations.

---

## Running the server

Linux / macOS
```bash
uvicorn app.main:app --reload
```

Windows (PowerShell / cmd)
```powershell
uvicorn app.main:app --reload
```

Docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## API Endpoints (Auth)

Base path: /auth

1) Register
- POST /auth/register
- Body (application/json):
```json
{
  "username": "test",
  "email": "test@gmail.com",
  "password": "test123@"
}
```
- Success: 200 { "message": "User registered successfully" }
- Error: 400 if email already exists

2) Login
- POST /auth/login
- Body:
```json
{
  "email": "test@gmail.com",
  "password": "test123@"
}
```
- Success: 200 { "access_token": "<jwt>", "token_type": "bearer" }
- Error: 401 Invalid credentials

---

## Testing examples

Quick curl examples:

Register:
```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@gmail.com","password":"test123@"}'
```

Login:
```bash
curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@gmail.com","password":"test123@"}'
```

Python requests test script (file: test_auth_endpoints.py)
```python
import requests

BASE = "http://127.0.0.1:8000/auth"

r = requests.post(f"{BASE}/register", json={
    "username":"test", "email":"test@gmail.com", "password":"test123@"
})
print("register", r.status_code, r.json())

r = requests.post(f"{BASE}/login", json={
    "email":"test@gmail.com", "password":"test123@"
})
print("login", r.status_code, r.json())
```

Run:
```bash
python test_auth_endpoints.py
```

---

## Project structure (important files)
- app/main.py — FastAPI app and route registration
- app/routes/auth.py — auth endpoints (register, login)
- app/services/auth.py — business logic using SQLAlchemy Session
- app/models/user.py — SQLAlchemy user model
- app/schemas/auth_schema.py — Pydantic request/response schemas
- app/db/session.py — engine, SessionLocal, get_db()
- app/core/security.py — password hashing (argon2) and JWT helpers
- app/init_db.py — module used with `python3.13 -m app.init_db` to initialize DB
- migrate.py — simple script to create tables (alternative)

---

## Notes / Tips
- Ensure the shared DB is reachable from your machine and DATABASE_URL is correct.
- For development, use the shared DB carefully — avoid destructive changes without coordination.
- For production, adopt Alembic for migrations and a secure secret management solution.
- `.gitignore` already ignores `.env`. Do not commit secrets.

---

If you want, I can:
- Add a basic `test_auth_endpoints.py` file in the repo.
- Add a short troubleshooting section for the common Pydantic BaseSettings error.
