from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "EthosLens"
    SECRET_KEY: str = "your_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str  # Add this line to include the DATABASE_URL
    GROQ_API_KEY: str  # Add this line to include the GROQ_API_KEY

    # CORS: default allow local frontend on port 3000 (can be overridden via .env)
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    class Config:
        env_file = ".env"  # Load environment variables from the .env file


settings = Settings()
