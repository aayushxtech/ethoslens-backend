# EthosLens Backend

This is the backend for the EthosLens project, built with FastAPI. It provides user authentication and other essential APIs.

---

## **Features**

- User registration and login with hashed passwords.
- JWT-based authentication.
- PostgreSQL database integration.
- Modular and scalable architecture.

---

## **Requirements**

- Python 3.10 or higher
- PostgreSQL
- Virtual environment (recommended)

---

## **Setup Instructions**

### 1. Clone the Repository

```bash
git clone https://github.com/aayushxtech/ethoslens-backend.git
cd backend
```

### 2. Create Virtual Env

```bash
python3 -m venv venv #Windows: python -m venv venv
source venv/bin/activate #Windows venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requriements.txt
```

### 4. Configure venv

```bash
DATABASE_URL="postgresql://<username>:<password>@<host>/<database>?sslmode=require"
```

### 5. Initialize the Database

The database tables will be automatically created when you run the application. Ensure your database is accessible.

---

### 1. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

### 2. Access the API Documentation

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## API Endpoints

### 1. User Registration

POST /auth/register

Request Body:

```request
{
  "username": "test",
  "email": "test@gmail.com",
  "password": "test123@"
}
```

### 2. 2. User Login

POST /auth/login

Request Body:

```request
{
  "email": "test@gmail.com",
  "password": "test123@"
}
```

---

## Development Notes

### Code Structure

- app/main.py: Application entry point.
- app/routes/auth.py: Authentication routes.
- app/services/auth.py: Business logic for authentication.
- app/models/user.py: SQLAlchemy model for the User table.
- app/schemas/auth_schema.py: Pydantic schemas for request/response validation.
- app/db/session.py: Database session and engine setup.
- app/core/security.py: Password hashing and JWT token generation.

### Database

- The database is managed using SQLAlchemy.
- The User table is defined in app/models/user.py.
