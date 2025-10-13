from sqlalchemy.orm import Session
from app.schemas.auth_schema import UserCreate, UserLogin, Token
from app.models.user import User
from app.core.security import hash_password, verify_password, create_access_token
from fastapi import HTTPException
import logging

logging.basicConfig(level=logging.DEBUG)


def register_user(user: UserCreate, db: Session):
    logging.debug(f"Registering user: {user}")
    # Check if the email already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(
            status_code=400, detail="User with this email already exists")

    # Hash the password and create a new user
    hashed_password = hash_password(user.password)
    new_user = User(email=user.email, username=user.username,
                    password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}


def login_user(user: UserLogin, db: Session):
    logging.debug(f"Logging in user: {user}")
    # Fetch the user from the database
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create a JWT token
    token = create_access_token({"sub": db_user.username})
    return Token(access_token=token)
