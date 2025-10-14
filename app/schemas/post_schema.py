from pydantic import BaseModel
from datetime import datetime

class PostBase(BaseModel):
    title: str
    content: str
class PostCreate(PostBase):
    pass

class PostUpdate(PostBase):
    title: str | None = None
    content: str | None = None

class ResponsePost(PostBase):
    id: int
    user_id: int | None
    created_at: datetime
    updated_at: datetime | None
    upvotes: int
    downvotes: int

    class Config:
        orm_mode = True
