import logging
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from app.config import settings

# Use argon2 instead of bcrypt to avoid 72-byte password limitation
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Password hashing


def hash_password(password: str) -> str:
    logging.debug(f"Hashing password: {password}")
    # No truncation needed with argon2
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # No truncation needed with argon2
    return pwd_context.verify(plain_password, hashed_password)

# JWT token creation


def create_access_token(data: dict, expires_delta: int | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + \
        (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt 
