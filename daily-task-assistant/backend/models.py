from sqlalchemy import Column, Integer, String, Boolean, Date, Time, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)

    items = relationship("Item", back_populates="user", cascade="all, delete-orphan")
    contexts = relationship("DayContext", back_populates="user", cascade="all, delete-orphan")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    priority = Column(String, default="medium")  # low/medium/high
    category = Column(String, default="general")
    active = Column(Boolean, default=True)

    user = relationship("User", back_populates="items")
    statuses = relationship("DailyItemStatus", back_populates="item", cascade="all, delete-orphan")


class DayContext(Base):
    __tablename__ = "day_contexts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(Date, nullable=False)
    weekday = Column(Integer, nullable=False)
    is_holiday = Column(Boolean, default=False)
    has_work_event = Column(Boolean, default=False)
    has_gym_event = Column(Boolean, default=False)
    cluster_label = Column(Integer, nullable=True)

    user = relationship("User", back_populates="contexts")
    item_statuses = relationship("DailyItemStatus", back_populates="context", cascade="all, delete-orphan")


class DailyItemStatus(Base):
    __tablename__ = "daily_item_status"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    context_id = Column(Integer, ForeignKey("day_contexts.id"), nullable=False)

    needed_label = Column(Boolean, nullable=True)  # Was the item actually needed?
    packed = Column(Boolean, default=False)        # Did user mark as packed?
    reminder_sent = Column(Boolean, default=False)
    feedback = Column(String, nullable=True)

    item = relationship("Item", back_populates="statuses")
    context = relationship("DayContext", back_populates="item_statuses")