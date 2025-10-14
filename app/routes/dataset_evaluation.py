from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.dataset import Dataset
from app.schemas.dataset_schema import DatasetEvaluationResponse


router = APIRouter()


@router.get("/datasets/{dataset_id}/evaluation", response_model=DatasetEvaluationResponse)
def get_dataset_evaluation(dataset_id: int, db: Session = Depends(get_db)):
    # Fetch Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # Load Dataset
    df = load_data
    # Run basic quality tests
    # Run fairness tests
