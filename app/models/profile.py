from sqlalchemy import Column, Integer, String, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from app.db.session import Base


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    about = Column(Text, nullable=True)
    records = Column(JSON, default={})
    reward_points = Column(Integer, default=0)
    num_posts = Column(Integer, default=0)
    user = relationship("User", back_populates="profile")

    def add_post(self):
        self.num_posts += 1
        self.reward_points += 4  # Example: 4 points per post
    def remove_post(self):
        if self.num_posts > 0:
            self.num_posts -= 1
            self.reward_points = max(0, self.reward_points - 4)  # Deduct points but not below 0
    def update_records(self, key: str):
        rec = self.records or {}
        rec[key] = rec.get(key, 0) + 1
        self.records = rec

