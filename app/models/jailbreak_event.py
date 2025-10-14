from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from app.db.session import Base

class JailbreakEvent(Base):
    __tablename__ = "jailbreak_events"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    jailbreak_prompt = Column(Text, nullable=False)
    jailbreak_result = Column(JSON, nullable=True)  # store LLM result / structured payload
    proof_url = Column(String, nullable=True)  # URL / local path to stored proof
    proof_path = Column(String, nullable=True)  # optional local filesystem path
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("Model", back_populates="jailbreak_events")