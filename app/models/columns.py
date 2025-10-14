from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON, DateTime, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    column_type = Column(String, nullable=False)
    missing_count = Column(Integer, default=0)
    missing_percentage = Column(Float, default=0.0)
    unique_count = Column(Integer, default=0)
    stats = Column(JSON, default={})
    example_value = Column(String, nullable=True)
    top_values = Column(JSON, default=[])
    is_target = Column(Boolean, default=False)
    is_sensitive = Column(Boolean, default=False)
    reason = Column(String, nullable=True)  # reason for sensitive flag
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Add this reciprocal relationship so back_populates matches Dataset.columns
    dataset = relationship("Dataset", back_populates="columns")
