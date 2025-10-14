from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth, datasets
from app.config import settings  # <- import settings
import app.models  # <- ensure all model modules are imported and mappers registered

app = FastAPI()

# Use configured origins (reads from app.config.settings)
origins = settings.ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])


@app.get("/")
async def read_root():
    return {"message": "EthosLens API is running"}
