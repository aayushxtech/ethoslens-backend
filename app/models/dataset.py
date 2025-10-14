from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    row_count = Column(Integer, nullable=True)
    schema = Column(JSON, nullable=True)
    target_column = Column(String, nullable=True)
    sensitive_columns = Column(JSON, nullable=True)  # list of sensitive fields
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    columns = relationship(
        "DatasetColumn", back_populates="dataset", cascade="all, delete-orphan")
