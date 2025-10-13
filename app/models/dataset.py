from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from app.db.session import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    # path for local storage or S3 URL
    file_path = Column(String, nullable=False)
    row_count = Column(Integer, nullable=True)
    schema = Column(JSON, nullable=True)  # store schema & metadata
    target_column = Column(String, nullable=True)
    sensitive_columns = Column(JSON, nullable=True)  # list of sensitive fields
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    version = Column(Integer, default=1)  # pending / evaluating / complete
    status = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
