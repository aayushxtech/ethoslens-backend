from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.post import Post
from app.schemas.post_schema import PostCreate, PostUpdate, ResponsePost

router = APIRouter(prefix="/posts", tags=["posts"])

# Create a new post

@router.post("/", response_model=ResponsePost, status_code=status.HTTP_201_CREATED)

async def create_post(post: PostCreate, db: Session = Depends(get_db), user_id: int = None):
    new_post = Post(title=post.title, content=post.content, user_id=user_id)
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return new_post

# Get all post

@router.get("/", response_model=list[ResponsePost])

async def get_all_posts(db: Session=Depends(get_db)):
    return db.query(Post).all()

# Get a post by id

@router.get("/{post_id}", response_model=ResponsePost)

async def get_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id== post_id).first()
    if not post:
        raise
    HTTPException(status_code=404, detail="Post not found")
    return post

# Update 

@router.put("/{post_id}", response_model= ResponsePost)

async def update_post(post_id: int, post_data: PostUpdate, db: Session= Depends(get_db)):
    post = db.query(Post).filter(Post.id==post_id).first()
    if not post:
        raise
    HTTPException(status_code=404, detail="Post not found")
    for key, value in post_data.dict(exclude_unset=True).items():
        setattr(post, key, value)
    db.commit()
    db.refresh(post)
    return post

# Delete 

@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)

async def delete_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id==post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    db.delete(post)
    db.commit()
    return {"detail": "Post deleted successfully"}

# Upvote

@router.post("/{post_id}/upvote")

async def upvote_post(post_id: int, db: Session= Depends(get_db)):
    post = db.query(Post).filter(Post.id==post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.upvotes += 1
    db.commit()
    db.refresh(post)
    return {"upvotes": post.upvotes, "downvotes": post.downvotes}

# Downvote

@router.post("/{post_id}/downvote")

async def downvote_post(post_id: int, db: Session= Depends(get_db)):
    post = db.query(Post).filter(Post.id==post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.downvotes += 1
    db.commit()
    db.refresh(post)
    return {"upvotes": post.upvotes, "downvotes": post.downvotes}


                        
