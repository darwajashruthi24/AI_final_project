import datetime as dt
import os

import pandas as pd
from werkzeug.security import generate_password_hash

from database import SessionLocal, Base, engine
import models
from ml import train_models_for_user

"""
Seed from CSV + train models.

Run from backend folder:

    cd backend
    python seed_from_csv.py

This will:
- DROP and recreate all tables (⚠ wipes existing data).
- Read CSV with >600 records (see expected columns).
- Create users, items, day contexts, and daily item statuses.
- Train ML models for each user using the imported data.

Configure CSV_PATH below.
"""

# ---- CONFIGURE THIS PATH ----
# Relative to the backend folder. You can change this as needed.
CSV_PATH = os.path.join("data", "daily_task_history_rich_items.csv")


def reset_db():
    """Drop and recreate all DB tables."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def get_or_create_user(db, email: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if user:
        return user, False
    user = models.User(
        email=email,
        password_hash=generate_password_hash("demo123"),  # default password
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, True


def get_or_create_item(db, user_id: int, name: str, priority: str, category: str):
    item = (
        db.query(models.Item)
        .filter(
            models.Item.user_id == user_id,
            models.Item.name == name,
        )
        .first()
    )
    if item:
        return item, False
    item = models.Item(
        user_id=user_id,
        name=name,
        priority=priority or "medium",
        category=category or "general",
        active=True,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item, True


def get_or_create_context(
    db,
    user_id: int,
    date: dt.date,
    weekday: int,
    is_holiday: int,
    has_work_event: int,
    has_gym_event: int,
):
    ctx = (
        db.query(models.DayContext)
        .filter(
            models.DayContext.user_id == user_id,
            models.DayContext.date == date,
        )
        .first()
    )
    if ctx:
        return ctx, False

    ctx = models.DayContext(
        user_id=user_id,
        date=date,
        weekday=weekday,
        is_holiday=bool(is_holiday),
        has_work_event=bool(has_work_event),
        has_gym_event=bool(has_gym_event),
    )
    db.add(ctx)
    db.commit()
    db.refresh(ctx)
    return ctx, True


def seed_from_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV file not found at {CSV_PATH}. Please create it with your 600+ records."
        )

    print(f"Reading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Basic validation of columns
    required_cols = [
        "user_email",
        "date",
        "is_holiday",
        "has_work_event",
        "has_gym_event",
        "item_name",
        "item_priority",
        "item_category",
        "needed_label",
        "packed",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    # Parse dates
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # If weekday column missing or empty, compute from date
    if "weekday" not in df.columns:
        df["weekday"] = df["date"].apply(lambda d: d.weekday())
    else:
        # If present but has NaN, recompute those
        df["weekday"] = df.apply(
            lambda row: int(row["weekday"])
            if pd.notna(row["weekday"])
            else row["date"].weekday(),
            axis=1,
        )

    print(f"Loaded {len(df)} rows from CSV.")

    reset_db()
    db = SessionLocal()

    try:
        users_seen = {}

        # We'll create everything row by row
        for idx, row in df.iterrows():
            email = str(row["user_email"]).strip().lower()
            date = row["date"]
            weekday = int(row["weekday"])
            is_holiday = int(row["is_holiday"])
            has_work_event = int(row["has_work_event"])
            has_gym_event = int(row["has_gym_event"])

            item_name = str(row["item_name"]).strip()
            item_priority = str(row["item_priority"]).strip().lower() or "medium"
            item_category = str(row["item_category"]).strip().lower() or "general"

            needed_label = int(row["needed_label"])
            packed = int(row["packed"])

            # 1) User
            if email not in users_seen:
                user, _created = get_or_create_user(db, email)
                users_seen[email] = user
            else:
                user = users_seen[email]

            # 2) Item
            item, _ = get_or_create_item(db, user.id, item_name, item_priority, item_category)

            # 3) Context
            ctx, _ = get_or_create_context(
                db,
                user.id,
                date,
                weekday,
                is_holiday,
                has_work_event,
                has_gym_event,
            )

            # 4) DailyItemStatus
            existing = (
                db.query(models.DailyItemStatus)
                .filter(
                    models.DailyItemStatus.user_id == user.id,
                    models.DailyItemStatus.context_id == ctx.id,
                    models.DailyItemStatus.item_id == item.id,
                )
                .first()
            )
            if existing:
                # If already exists, you can choose to update or skip.
                existing.needed_label = bool(needed_label)
                existing.packed = bool(packed)
            else:
                st = models.DailyItemStatus(
                    user_id=user.id,
                    item_id=item.id,
                    context_id=ctx.id,
                    needed_label=bool(needed_label),
                    packed=bool(packed),
                )
                db.add(st)

            if (idx + 1) % 100 == 0:
                db.commit()
                print(f"... imported {idx + 1} rows")

        db.commit()
        print("Finished importing all rows from CSV.")

        # Train model for each user
        print("\nTraining models for users...")
        for email, user in users_seen.items():
            ok = train_models_for_user(db, user.id)
            if ok:
                print(f"  -> Model trained for {email}")
            else:
                print(f"  -> Not enough varied data to train model for {email}")

        print("\nSeed + training complete.")
        print("Demo passwords for all CSV users: demo123")
        print("You can now run: python app.py and log in using emails from the CSV.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_from_csv()