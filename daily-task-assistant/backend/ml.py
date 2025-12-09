import os
from pathlib import Path
from typing import List, Dict
import json

import pandas as pd
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

import models

MODEL_DIR = Path(__file__).parent / "models_store"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------- DATA LOADING ---------- #

def load_training_data(db: Session, user_id: int) -> pd.DataFrame:
    """
    Load labelled training data for a single user.
    """
    q = (
        db.query(models.DayContext, models.DailyItemStatus, models.Item)
        .join(models.DailyItemStatus, models.DayContext.id == models.DailyItemStatus.context_id)
        .join(models.Item, models.Item.id == models.DailyItemStatus.item_id)
        .filter(models.DayContext.user_id == user_id)
    )

    rows = []
    for ctx, status, item in q:
        if status.needed_label is None:
            continue
        rows.append(
            {
                "weekday": ctx.weekday,
                "is_holiday": int(ctx.is_holiday),
                "has_work_event": int(ctx.has_work_event),
                "has_gym_event": int(ctx.has_gym_event),
                "priority": {"low": 0, "medium": 1, "high": 2}.get(item.priority, 1),
                "needed_label": int(status.needed_label),
                "packed": int(status.packed),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_training_data_all_users(db: Session) -> pd.DataFrame:
    """
    Load labelled training data across ALL users.
    This will be used to train a GLOBAL model that powers new users.
    """
    q = (
        db.query(models.DayContext, models.DailyItemStatus, models.Item)
        .join(models.DailyItemStatus, models.DayContext.id == models.DailyItemStatus.context_id)
        .join(models.Item, models.Item.id == models.DailyItemStatus.item_id)
    )

    rows = []
    for ctx, status, item in q:
        if status.needed_label is None:
            continue
        rows.append(
            {
                "weekday": ctx.weekday,
                "is_holiday": int(ctx.is_holiday),
                "has_work_event": int(ctx.has_work_event),
                "has_gym_event": int(ctx.has_gym_event),
                "priority": {"low": 0, "medium": 1, "high": 2}.get(item.priority, 1),
                "needed_label": int(status.needed_label),
                "packed": int(status.packed),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------- METRICS STORAGE ---------- #

def _save_metrics(metrics_path: Path, y_true, y_pred):
    if len(y_true) == 0:
        return
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_model_metrics(user_id: int) -> Dict:
    """
    Load stored PERSONAL model metrics for a user.
    """
    metrics_path = MODEL_DIR / f"metrics_user_{user_id}.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_global_model_metrics() -> Dict:
    """
    Load GLOBAL model metrics (across all users).
    """
    metrics_path = MODEL_DIR / "metrics_global.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- TRAINING: PERSONAL MODEL ---------- #

def train_models_for_user(db: Session, user_id: int) -> bool:
    """
    Train a personal model for a specific user.
    """
    df = load_training_data(db, user_id)
    if df.empty or df["needed_label"].nunique() < 2:
        return False

    # Cluster “day types”
    day_features = df[["weekday", "is_holiday", "has_work_event", "has_gym_event"]].drop_duplicates()
    n_clusters = min(3, len(day_features))
    if n_clusters < 2:
        day_features["cluster_label"] = 0
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(day_features)
        day_features["cluster_label"] = kmeans.labels_

    df = df.merge(
        day_features,
        on=["weekday", "is_holiday", "has_work_event", "has_gym_event"],
        how="left",
    )

    X = df[["weekday", "is_holiday", "has_work_event", "has_gym_event", "priority", "cluster_label"]]
    y_needed = df["needed_label"]
    y_forget = ((df["needed_label"] == 1) & (df["packed"] == 0)).astype(int)

    routine_model = RandomForestClassifier(n_estimators=100, random_state=42)
    routine_model.fit(X, y_needed)

    y_pred = routine_model.predict(X)
    _save_metrics(MODEL_DIR / f"metrics_user_{user_id}.json", y_needed, y_pred)

    forget_model = LogisticRegression(max_iter=1000)
    forget_model.fit(X, y_forget)

    joblib.dump(day_features, MODEL_DIR / f"ctx_features_user_{user_id}.pkl")
    joblib.dump(routine_model, MODEL_DIR / f"routine_model_user_{user_id}.pkl")
    joblib.dump(forget_model, MODEL_DIR / f"forget_model_user_{user_id}.pkl")

    return True


# ---------- TRAINING: GLOBAL MODEL (ALL USERS) ---------- #

def train_global_models(db: Session) -> bool:
    """
    Train ONE global model across all users.
    New users (without personal model) will use this global model.
    """
    df = load_training_data_all_users(db)
    if df.empty or df["needed_label"].nunique() < 2:
        return False

    day_features = df[["weekday", "is_holiday", "has_work_event", "has_gym_event"]].drop_duplicates()
    n_clusters = min(3, len(day_features))
    if n_clusters < 2:
        day_features["cluster_label"] = 0
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(day_features)
        day_features["cluster_label"] = kmeans.labels_

    df = df.merge(
        day_features,
        on=["weekday", "is_holiday", "has_work_event", "has_gym_event"],
        how="left",
    )

    X = df[["weekday", "is_holiday", "has_work_event", "has_gym_event", "priority", "cluster_label"]]
    y_needed = df["needed_label"]
    y_forget = ((df["needed_label"] == 1) & (df["packed"] == 0)).astype(int)

    routine_model = RandomForestClassifier(n_estimators=120, random_state=42)
    routine_model.fit(X, y_needed)

    y_pred = routine_model.predict(X)
    _save_metrics(MODEL_DIR / "metrics_global.json", y_needed, y_pred)

    forget_model = LogisticRegression(max_iter=1000)
    forget_model.fit(X, y_forget)

    joblib.dump(day_features, MODEL_DIR / "global_ctx_features.pkl")
    joblib.dump(routine_model, MODEL_DIR / "global_routine_model.pkl")
    joblib.dump(forget_model, MODEL_DIR / "global_forget_model.pkl")

    return True


# ---------- PREDICTION (PERSONAL → GLOBAL → HEURISTIC) ---------- #

def _predict_with_models(context_features: Dict, items: List[Dict], day_features, routine_model, forget_model):
    """
    Shared logic: given context + items + loaded models, compute predictions.
    """
    ctx_df = pd.DataFrame([context_features])
    merged = ctx_df.merge(
        day_features,
        on=["weekday", "is_holiday", "has_work_event", "has_gym_event"],
        how="left",
    )
    if merged["cluster_label"].isna().all():
        merged["cluster_label"] = 0

    cluster_label = int(merged["cluster_label"].iloc[0])

    rows = []
    for it in items:
        rows.append(
            {
                "weekday": context_features["weekday"],
                "is_holiday": context_features["is_holiday"],
                "has_work_event": context_features["has_work_event"],
                "has_gym_event": context_features["has_gym_event"],
                "priority": {"low": 0, "medium": 1, "high": 2}.get(it["priority"], 1),
                "cluster_label": cluster_label,
            }
        )

    X = pd.DataFrame(rows)
    need_probs = routine_model.predict_proba(X)[:, 1]
    forget_probs = forget_model.predict_proba(X)[:, 1]

    results = []
    for it, p_need, p_forget in zip(items, need_probs, forget_probs):
        score = float(p_need * (0.7 + 0.3 * p_forget))
        results.append(
            {
                "item_id": it["id"],
                "name": it["name"],
                "need_probability": float(p_need),
                "forget_risk": float(p_forget),
                "score": score,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def predict_items_for_today(user_id: int, context_features: Dict, items: List[Dict]):
    """
    Prediction logic with 3 levels:

    1) If PERSONAL model exists for this user → use it.
    2) Else if GLOBAL model exists → use it.
    3) Else → use simple heuristic based on priority.
    """
    # --- Try personal model first ---
    ctx_path = MODEL_DIR / f"ctx_features_user_{user_id}.pkl"
    routine_path = MODEL_DIR / f"routine_model_user_{user_id}.pkl"
    forget_path = MODEL_DIR / f"forget_model_user_{user_id}.pkl"

    if ctx_path.exists() and routine_path.exists() and forget_path.exists():
        day_features = joblib.load(ctx_path)
        routine_model = joblib.load(routine_path)
        forget_model = joblib.load(forget_path)
        return _predict_with_models(context_features, items, day_features, routine_model, forget_model)

    # --- Otherwise, try GLOBAL model ---
    g_ctx_path = MODEL_DIR / "global_ctx_features.pkl"
    g_routine_path = MODEL_DIR / "global_routine_model.pkl"
    g_forget_path = MODEL_DIR / "global_forget_model.pkl"

    if g_ctx_path.exists() and g_routine_path.exists() and g_forget_path.exists():
        day_features = joblib.load(g_ctx_path)
        routine_model = joblib.load(g_routine_path)
        forget_model = joblib.load(g_forget_path)
        return _predict_with_models(context_features, items, day_features, routine_model, forget_model)

    # --- Fallback heuristic (if no models yet) ---
    results = []
    for it in items:
        base_prob = {"low": 0.3, "medium": 0.5, "high": 0.7}.get(it["priority"], 0.5)
        results.append(
            {
                "item_id": it["id"],
                "name": it["name"],
                "need_probability": base_prob,
                "forget_risk": 0.4,
                "score": base_prob,
            }
        )
    return results