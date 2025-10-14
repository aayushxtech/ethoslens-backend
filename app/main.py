from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth, datasets

app = FastAPI()

# Allow front-end running on port 3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

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
