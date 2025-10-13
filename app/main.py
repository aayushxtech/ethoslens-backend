from fastapi import FastAPI
from app.routes import auth
from app.models.user import User  # Corrected import
from app.db.session import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router, prefix="/auth", tags=["auth"])


@app.get("/")
async def read_root():
    return {"message": "EthosLens API is running"}
