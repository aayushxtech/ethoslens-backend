from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.profile import Profile
from app.schemas.profile_schema import ProfileCreate, ProfileUpdate, ResponseProfile

router = APIRouter(prefix="/profiles", tags=["profiles"])

# Create a new profile

@router.post("/", response_model=ResponseProfile, status_code=status.HTTP_201_CREATED)

async def create_profile(profile: ProfileCreate, db: Session = Depends(get_db), user_id: int = None):
    existing = db.query(Profile).filter(Profile.user_id==user_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Profile already exists for this user")
    new_profile = Profile(user_id=user_id, name=profile.name, about=profile.about)
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    return new_profile

# Get profile by user_id

@router.get("/{user_id}", response_model=ResponseProfile)

async def get_profile(user_id: int, db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id==user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

# Update profile

@router.put("/{user_id}", response_model=ResponseProfile)

async def update_profile(user_id: int, profile_data: ProfileUpdate, db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id==user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    for key, value in profile_data.dict(exclude_unset=True).items():
        setattr(profile, key, value)
    db.commit()
    db.refresh(profile)
    return profile

# Delete profile

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)

async def delete_profile(user_id: int, db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id==user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    db.delete(profile)
    db.commit()
    return {"detail": "Profile deleted successfully"}