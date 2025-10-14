from pydantic import BaseModel
from typing import Any

class ProfileBase(BaseModel):
    name: str
    about: str | None = None
    records: dict[str, Any] | None = {}
    reward_points: int | None = 0
    num_posts: int | None = 0

class ProfileCreate(ProfileBase):
    pass

class ProfileUpdate(ProfileBase):
    name: str | None = None
    about: str | None = None
    records: dict[str, Any] | None = None
    reward_points: int | None = None
    num_posts: int | None = None

class ResponseProfile(ProfileBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True
