from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "EthosLens"
    SECRET_KEY: str = "your_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str  # Add this line to include the DATABASE_URL

    class Config:
        env_file = ".env"  # Load environment variables from the .env file


settings = Settings()
