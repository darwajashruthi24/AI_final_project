import datetime as dt
from pathlib import Path
from collections import defaultdict
import os
import smtplib
import hashlib
import atexit

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import (
    Flask, render_template, redirect, url_for, request, jsonify, flash
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler

from database import Base, engine, SessionLocal
import models
from ml import (
    train_models_for_user,
    train_global_models,
    predict_items_for_today,
    load_model_metrics,
    load_global_model_metrics,
)

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "frontend" / "templates"),
    static_folder=str(Path(__file__).parent.parent / "frontend" / "static"),
)
app.secret_key = "change-this-secret"  # <-- change for production

# Email configuration (use environment variables in real setup)
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587"))
EMAIL_USER = os.environ.get("EMAIL_USER")          # e.g. your Gmail address
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")  # Gmail app password
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")
TEST_EMAIL_TO = os.environ.get("TEST_EMAIL_TO")
# ---- Flask-Login Setup ----
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


class UserLogin(UserMixin):
    """Adapter for Flask-Login using our SQLAlchemy User model."""
    def __init__(self, user: models.User):
        self.id = user.id
        self.email = user.email


@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == int(user_id)).first()
        if user:
            return UserLogin(user)
        return None
    finally:
        db.close()


def get_session() -> Session:
    return SessionLocal()


# --------- DEFAULT ITEMS FOR ALL USERS --------- #

DEFAULT_ITEMS = [
    ("ID Card", "high", "work"),
    ("Access Card", "high", "work"),
    ("Laptop", "high", "work"),
    ("Charger", "medium", "work"),
    ("Power Bank", "medium", "work"),
    ("Lunch Box", "medium", "home"),
    ("Water Bottle", "medium", "personal"),
    ("Notebook", "medium", "work"),
    ("Pen", "low", "work"),
    ("Headphones", "low", "work"),
    ("Gym Shoes", "medium", "gym"),
    ("Gym Towel", "low", "gym"),
    ("Umbrella", "medium", "personal"),
]


def ensure_default_items(db: Session, user_id: int):
    """
    Make sure the given user has all DEFAULT_ITEMS.
    This is idempotent: no duplicates are created by name.
    """
    existing = (
        db.query(models.Item)
        .filter(models.Item.user_id == user_id)
        .all()
    )
    existing_names = {it.name for it in existing}

    created_any = False
    for name, priority, category in DEFAULT_ITEMS:
        if name not in existing_names:
            item = models.Item(
                user_id=user_id,
                name=name,
                priority=priority,
                category=category,
                active=True,
            )
            db.add(item)
            created_any = True

    if created_any:
        db.commit()

# ----------------- EMAIL HELPERS ----------------- #

def send_email(to_email: str, subject: str, html_body: str):
    # If TEST_EMAIL_TO is set, override recipient (good for demo/testing)
    real_to = TEST_EMAIL_TO or to_email

    if not (EMAIL_USER and EMAIL_PASSWORD):
        print(f"[EMAIL_DISABLED] Would send to {real_to} -> {subject}")
        print(html_body)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = real_to
    part_html = MIMEText(html_body, "html")
    msg.attach(part_html)

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        print(f"[EMAIL] Sent reminder to {real_to}")


def generate_email_token(user_id: int, date_str: str) -> str:
    raw = f"{user_id}:{date_str}:{app.secret_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_email_token(user_id: int, date_str: str, token: str) -> bool:
    return token == generate_email_token(user_id, date_str)

# ----------------- AUTH ROUTES ----------------- #

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        db = get_session()
        try:
            existing = db.query(models.User).filter(models.User.email == email).first()
            if existing:
                flash("Email already registered", "error")
                return redirect(url_for("register"))

            user = models.User(email=email, password_hash=generate_password_hash(password))
            db.add(user)
            db.commit()
            db.refresh(user)

            # 👇 NEW: give this user the default items
            ensure_default_items(db, user.id)

            flash("Registered successfully. Please log in.", "success")
            return redirect(url_for("login"))
        finally:
            db.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        db = get_session()
        try:
            user = db.query(models.User).filter(models.User.email == email).first()
            if not user or not check_password_hash(user.password_hash, password):
                flash("Invalid credentials", "error")
                return redirect(url_for("login"))
            ensure_default_items(db, user.id)
            login_user(UserLogin(user))
            return redirect(url_for("dashboard"))
        finally:
            db.close()

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ----------------- PAGES ----------------- #

@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/items", methods=["GET", "POST"])
@login_required
def items():
    db = get_session()
    try:
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            priority = request.form.get("priority", "medium")
            category = request.form.get("category", "general")
            if name:
                item = models.Item(
                    user_id=current_user.id,
                    name=name,
                    priority=priority,
                    category=category,
                )
                db.add(item)
                db.commit()
                flash("Item added.", "success")
        items_list = (
            db.query(models.Item)
            .filter(models.Item.user_id == current_user.id)
            .all()
        )
        return render_template("items.html", items=items_list)
    finally:
        db.close()


@app.route("/history")
@login_required
def history():
    db = get_session()
    try:
        today = dt.date.today()
        ctx = (
            db.query(models.DayContext)
            .filter(models.DayContext.user_id == current_user.id,
                    models.DayContext.date == today)
            .first()
        )
        statuses = []
        if ctx:
            sts = (
                db.query(models.DailyItemStatus, models.Item)
                .join(models.Item, models.Item.id == models.DailyItemStatus.item_id)
                .filter(models.DailyItemStatus.context_id == ctx.id)
                .all()
            )
            for st, item in sts:
                statuses.append(
                    {
                        "name": item.name,
                        "packed": st.packed,
                        "needed_label": st.needed_label,
                    }
                )
        return render_template("history.html", today=str(today), statuses=statuses)
    finally:
        db.close()


@app.route("/simulate")
@login_required
def simulate():
    return render_template("simulate.html")

# ----------------- CONTEXT HELPER ----------------- #

def get_or_create_today_context(db: Session, user_id: int) -> models.DayContext:
    today = dt.date.today()
    ctx = (
        db.query(models.DayContext)
        .filter(models.DayContext.user_id == user_id,
                models.DayContext.date == today)
        .first()
    )
    if ctx:
        return ctx

    ctx = models.DayContext(
        user_id=user_id,
        date=today,
        weekday=today.weekday(),
        is_holiday=False,
        has_work_event=False,
        has_gym_event=False,
    )
    db.add(ctx)
    db.commit()
    db.refresh(ctx)
    return ctx

# ----------------- API: ITEMS / CHECKLIST / ML ----------------- #

@app.route("/api/items")
@login_required
def api_items():
    db = get_session()
    try:
        items = (
            db.query(models.Item)
            .filter(models.Item.user_id == current_user.id,
                    models.Item.active == True)
            .all()
        )
        out = [
            {
                "id": it.id,
                "name": it.name,
                "priority": it.priority,
                "category": it.category,
                "active": it.active,
            }
            for it in items
        ]
        return jsonify(out)
    finally:
        db.close()


@app.route("/api/checklist_today")
@login_required
def api_checklist_today():
    db = get_session()
    try:
        ctx = get_or_create_today_context(db, current_user.id)
        items = (
            db.query(models.Item)
            .filter(models.Item.user_id == current_user.id,
                    models.Item.active == True)
            .all()
        )
        statuses = (
            db.query(models.DailyItemStatus)
            .filter(models.DailyItemStatus.user_id == current_user.id,
                    models.DailyItemStatus.context_id == ctx.id)
            .all()
        )
        status_map = {s.item_id: s for s in statuses}

        checklist = []
        for it in items:
            st = status_map.get(it.id)
            checklist.append(
                {
                    "item_id": it.id,
                    "name": it.name,
                    "priority": it.priority,
                    "packed": bool(st.packed) if st else False,
                    "needed_label": st.needed_label if st else None,
                }
            )
        return jsonify({
            "date": str(ctx.date),
            "weekday": ctx.weekday,
            "items": checklist,
        })
    finally:
        db.close()


@app.route("/api/checklist_update", methods=["POST"])
@login_required
def api_checklist_update():
    data = request.get_json()
    statuses = data.get("statuses", [])

    db = get_session()
    try:
        ctx = get_or_create_today_context(db, current_user.id)
        for st_json in statuses:
            item_id = st_json.get("item_id")
            packed = bool(st_json.get("packed", False))
            needed_label = st_json.get("needed_label")

            st = (
                db.query(models.DailyItemStatus)
                .filter(
                    models.DailyItemStatus.user_id == current_user.id,
                    models.DailyItemStatus.context_id == ctx.id,
                    models.DailyItemStatus.item_id == item_id,
                )
                .first()
            )
            if not st:
                st = models.DailyItemStatus(
                    user_id=current_user.id,
                    context_id=ctx.id,
                    item_id=item_id,
                )
                db.add(st)

            st.packed = packed
            if needed_label is not None:
                st.needed_label = bool(needed_label)

        db.commit()
        return jsonify({"status": "ok"})
    finally:
        db.close()


@app.route("/api/train_model", methods=["POST"])
@login_required
def api_train_model():
    db = get_session()
    try:
        ok = train_models_for_user(db, current_user.id)
        if not ok:
            return jsonify({"status": "error", "message": "Not enough data to train model"}), 400
        return jsonify({"status": "trained"})
    finally:
        db.close()


@app.route("/api/predict_today")
@login_required
def api_predict_today():
    db = get_session()
    try:
        today = dt.date.today()
        ctx = get_or_create_today_context(db, current_user.id)

        context_features = {
            "weekday": ctx.weekday,
            "is_holiday": int(ctx.is_holiday),
            "has_work_event": int(ctx.has_work_event),
            "has_gym_event": int(ctx.has_gym_event),
        }

        items = (
            db.query(models.Item)
            .filter(models.Item.user_id == current_user.id,
                    models.Item.active == True)
            .all()
        )
        item_dicts = [
            {"id": it.id, "name": it.name, "priority": it.priority}
            for it in items
        ]

        preds = predict_items_for_today(current_user.id, context_features, item_dicts)
        return jsonify({"date": str(today), "predictions": preds})
    finally:
        db.close()


@app.route("/api/simulate_predict", methods=["POST"])
@login_required
def api_simulate_predict():
    data = request.get_json()
    try:
        weekday = int(data.get("weekday", 0))
        is_holiday = 1 if data.get("is_holiday") else 0
        has_work_event = 1 if data.get("has_work_event") else 0
        has_gym_event = 1 if data.get("has_gym_event") else 0
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid payload"}), 400

    db = get_session()
    try:
        items = (
            db.query(models.Item)
            .filter(models.Item.user_id == current_user.id,
                    models.Item.active == True)
            .all()
        )
        item_dicts = [
            {"id": it.id, "name": it.name, "priority": it.priority}
            for it in items
        ]

        context_features = {
            "weekday": weekday,
            "is_holiday": is_holiday,
            "has_work_event": has_work_event,
            "has_gym_event": has_gym_event,
        }

        preds = predict_items_for_today(current_user.id, context_features, item_dicts)
        return jsonify({"predictions": preds})
    finally:
        db.close()

# ----------------- AI INSIGHTS ----------------- #

@app.route("/api/insights")
@login_required
def api_insights():
    db = get_session()
    try:
        q = (
            db.query(models.DailyItemStatus, models.Item, models.DayContext)
            .join(models.Item, models.Item.id == models.DailyItemStatus.item_id)
            .join(models.DayContext, models.DayContext.id == models.DailyItemStatus.context_id)
            .filter(models.DailyItemStatus.user_id == current_user.id)
        )

        stats = defaultdict(lambda: {
            "item_id": None,
            "name": "",
            "needed_days": 0,
            "packed_when_needed": 0,
            "forgotten_days": 0,
            "total_days": 0,
        })

        for st, item, ctx in q:
            entry = stats[item.id]
            entry["item_id"] = item.id
            entry["name"] = item.name

            if st.needed_label is not None:
                entry["total_days"] += 1
                if st.needed_label:
                    entry["needed_days"] += 1
                    if st.packed:
                        entry["packed_when_needed"] += 1
                    else:
                        entry["forgotten_days"] += 1

        per_item_stats = []
        for item_id, entry in stats.items():
            if entry["needed_days"] > 0:
                forget_rate = entry["forgotten_days"] / entry["needed_days"]
            else:
                forget_rate = 0.0
            entry["forget_rate"] = forget_rate
            per_item_stats.append(entry)

        top_forgotten = sorted(per_item_stats, key=lambda x: x["forget_rate"], reverse=True)
        metrics = load_model_metrics(current_user.id)

        today = dt.date.today()
        ctx = (
            db.query(models.DayContext)
            .filter(models.DayContext.user_id == current_user.id,
                    models.DayContext.date == today)
            .first()
        )
        today_ctx = None
        if ctx:
            day_type = "Weekend" if ctx.weekday >= 5 else "Weekday"
            today_ctx = {
                "date": str(ctx.date),
                "weekday": ctx.weekday,
                "day_type": day_type,
                "has_work_event": bool(ctx.has_work_event),
                "has_gym_event": bool(ctx.has_gym_event),
            }

        return jsonify({
            "per_item_stats": per_item_stats,
            "top_forgotten": top_forgotten,
            "model_metrics": metrics,
            "today_context": today_ctx,
        })
    finally:
        db.close()

@app.route("/api/train_global", methods=["POST"])
def api_train_global():
    """
    Train a GLOBAL model across all users.
    Use this after seeding/importing data or periodically.
    """
    db = SessionLocal()
    try:
        ok = train_global_models(db)
        if not ok:
            return jsonify({"status": "error", "message": "Not enough global data to train model"}), 400
        return jsonify({"status": "trained"})
    finally:
        db.close()

# ----------------- EMAIL: ONE-CLICK PACKED ----------------- #

@app.route("/email/mark_packed")
def email_mark_packed():
    """
    Called from email link:
      /email/mark_packed?user=<id>&date=<YYYY-MM-DD>&token=<hash>
    Marks all today's items as packed for that user & date.
    """
    user_id = request.args.get("user", type=int)
    date_str = request.args.get("date")
    token = request.args.get("token")

    if not (user_id and date_str and token and verify_email_token(user_id, date_str, token)):
        flash("Invalid or expired email link.", "error")
        return redirect(url_for("login"))

    try:
        date = dt.date.fromisoformat(date_str)
    except ValueError:
        flash("Invalid date in email link.", "error")
        return redirect(url_for("login"))

    db = get_session()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            flash("User not found for email link.", "error")
            return redirect(url_for("login"))

        ctx = (
            db.query(models.DayContext)
            .filter(models.DayContext.user_id == user_id,
                    models.DayContext.date == date)
            .first()
        )
        if not ctx:
            ctx = models.DayContext(
                user_id=user_id,
                date=date,
                weekday=date.weekday(),
                is_holiday=False,
                has_work_event=False,
                has_gym_event=False,
            )
            db.add(ctx)
            db.commit()
            db.refresh(ctx)

        items = (
            db.query(models.Item)
            .filter(models.Item.user_id == user_id,
                    models.Item.active == True)
            .all()
        )

        for it in items:
            st = (
                db.query(models.DailyItemStatus)
                .filter(
                    models.DailyItemStatus.user_id == user_id,
                    models.DailyItemStatus.context_id == ctx.id,
                    models.DailyItemStatus.item_id == it.id,
                )
                .first()
            )
            if not st:
                st = models.DailyItemStatus(
                    user_id=user_id,
                    item_id=it.id,
                    context_id=ctx.id,
                )
                db.add(st)
            st.packed = True
            if st.needed_label is None:
                st.needed_label = True

        db.commit()
        flash("Marked today's items as packed from your email reminder.", "success")
    finally:
        db.close()

    return redirect(url_for("login"))

# ----------------- DAILY EMAIL SCHEDULER ----------------- #

def send_daily_reminders():
    """
    Runs at 8 AM daily (server time).
    For each user:
      - compute today's predictions
      - send email with summary and one-click packed link
    """
    db = SessionLocal()
    
    try:
        # OPTIONAL: keep the global model up-to-date once per day
        try:
            trained = train_global_models(db)
            if trained:
                print("[SCHEDULER] Global model retrained successfully.")
            else:
                print("[SCHEDULER] Not enough global data to retrain model yet.")
        except Exception as e:
            print(f"[SCHEDULER] Global training failed: {e}")

        today = dt.date.today()
        users = db.query(models.User).all()
        if not users:
            print("[SCHEDULER] No users yet; skipping reminder.")
            return

        for user in users:
            ctx = (
                db.query(models.DayContext)
                .filter(models.DayContext.user_id == user.id,
                        models.DayContext.date == today)
                .first()
            )
            if not ctx:
                ctx = models.DayContext(
                    user_id=user.id,
                    date=today,
                    weekday=today.weekday(),
                    is_holiday=False,
                    has_work_event=False,
                    has_gym_event=False,
                )
                db.add(ctx)
                db.commit()
                db.refresh(ctx)

            context_features = {
                "weekday": ctx.weekday,
                "is_holiday": int(ctx.is_holiday),
                "has_work_event": int(ctx.has_work_event),
                "has_gym_event": int(ctx.has_gym_event),
            }

            items = (
                db.query(models.Item)
                .filter(models.Item.user_id == user.id,
                        models.Item.active == True)
                .all()
            )
            item_dicts = [
                {"id": it.id, "name": it.name, "priority": it.priority}
                for it in items
            ]

            preds = predict_items_for_today(user.id, context_features, item_dicts)
            # keep top 5 most important
            top_preds = [p for p in preds if p["need_probability"] > 0.5][:5]

            date_str = today.isoformat()
            token = generate_email_token(user.id, date_str)
            link = f"{APP_BASE_URL}/email/mark_packed?user={user.id}&date={date_str}&token={token}"

            items_html = ""
            if top_preds:
                items_html = "<ul>"
                for p in top_preds:
                    items_html += (
                        f"<li><strong>{p['name']}</strong> "
                        f"(Need: {p['need_probability']*100:.0f}%, "
                        f"Forget risk: {p['forget_risk']*100:.0f}%)</li>"
                    )
                items_html += "</ul>"
            else:
                items_html = "<p>No specific items predicted today. Check your checklist in the app.</p>"

            html_body = f"""
            <html>
            <body>
              <p>Good morning 👋,</p>
              <p>Here are the top items you usually need today ({date_str}):</p>
              {items_html}
              <p>
                <a href="{APP_BASE_URL}" style="padding:8px 14px; background:#2563eb; color:#fff; text-decoration:none; border-radius:4px;">
                  Open Daily Task Memory Assistant
                </a>
              </p>
              <p>
                If you've already packed, you can mark everything as packed for today with one click:
              </p>
              <p>
                <a href="{link}" style="padding:8px 14px; background:#16a34a; color:#fff; text-decoration:none; border-radius:4px;">
                  Yes, I’ve packed everything ✅
                </a>
              </p>
              <p style="font-size:12px; color:#6b7280;">
                This link is unique for today and will update your packing history.
              </p>
            </body>
            </html>
            """

            send_email(user.email, "Your daily packing reminder", html_body)

    except Exception as e:
        print(f"[SCHEDULER ERROR] {e}")
    finally:
        db.close()


scheduler = BackgroundScheduler()
# runs daily at 8:00
scheduler.add_job(send_daily_reminders, "cron", hour=8, minute=0)
#runs every 2mins
# runs daily at 8:00
# scheduler.add_job(send_daily_reminders, "interval", minutes=5)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    # run without debug reloader so scheduler doesn't double-run
    app.run(debug=False)