from fastapi import FastAPI
from app.routes import auth

app = FastAPI()

# Include authentication routes
app.include_router(auth.router, prefix="/auth", tags=["auth"])


@app.get("/")
async def read_root():
    return {"message": "EthosLens API is running"}
