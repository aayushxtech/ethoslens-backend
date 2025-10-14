from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db.session import Base


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    # numeric, categorical, boolean, other
    column_type = Column(String, nullable=False)
    missing_count = Column(Integer, default=0)
    missing_percentage = Column(Float, default=0.0)
    unique_count = Column(Integer, default=0)
    stats = Column(JSON, default={})  # numeric stats, top/freq for categorical

    is_target = Column(Boolean, default=False)
    is_sensitive = Column(Boolean, default=False)

    dataset = relationship("Dataset", back_populates="columns")