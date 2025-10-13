from fastapi import FastAPI
from app.routes import auth, datasets

app = FastAPI()

# Include authentication routes
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])


@app.get("/")
async def read_root():
    return {"message": "EthosLens API is running"}
