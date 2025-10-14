from sqlalchemy import Column, Float, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db.session import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, unique=True, index=True, nullable=False)  # Unique model identifier
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    report = Column(JSON, default={})  # JSON report of model evaluation
    score = Column(Float, default=0)  # Overall model score
    suggestions = Column(String, default={})  # Suggestions for improvement

    jailbreak_events = relationship("JailbreakEvent", back_populates="model", cascade="all, delete-orphan")

    def update_score(self, delta: float):
        new_score = self.score + delta
        self.score = max(0, min(100, new_score))  # Keep score between 0 and 100

    def add_report_entry(self, key: str, value):
        if not self.report:
            self.report = {}
        self.report[key] = value




    # creator = relationship("User", back_populates="models")