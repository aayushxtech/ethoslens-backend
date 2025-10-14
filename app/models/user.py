from sqlalchemy import Column, Integer, String, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    posts = relationship("Post", back_populates="user")
    profile = relationship("Profile", back_populates="user", uselist=False)
    models = relationship("Model", backref="creator")
   