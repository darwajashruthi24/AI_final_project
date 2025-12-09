# 📦 Daily Task Memory Assistant – AI-Powered Personal Reminder System

A Machine Learning + Flask + Bootstrap Web Application  
**Team Size:** 3 Members

---

## 📘 Overview

The **Daily Task Memory Assistant** is an AI-powered smart reminder system that learns a user's daily routine using machine learning models and automatically predicts which essential items they may forget before leaving home. It also provides:

- ✔️ **Personalized item predictions** based on user behavior
- ✔️ **Forget-probability estimation** with confidence scores
- ✔️ **Daily 8AM reminder emails** (configurable to 2 minutes for testing)
- ✔️ **One-click "Mark Packed"** from email links
- ✔️ **AI insights** on user-specific forget patterns
- ✔️ **What-if Simulation** to test different day types
- ✔️ **Global learning model** → New users benefit from all users' data
- ✔️ **Bootstrap-based responsive UI** for modern, clean design
- ✔️ **Background scheduler** that retrains models daily
- ✔️ **Performance metrics** (Accuracy, Precision, Recall, F1 Score)

This system demonstrates a **full AI pipeline** combined with a **full-stack web application**, suitable for a Master's project.

---

## 🏗 Architecture

```
         ┌────────────────────────────────────────────────┐
         │                CSV Dataset (180+ days)          │
         │            7000+ rows with patterns             │
         └────────────────────────────────────────────────┘
                            │
                            ▼
         ┌───────────────────────────┐
         │     FastAPI / Flask API    │
         │      Backend Server        │
         └───────────────────────────┘
          │        │         │
          ├────────┼─────────┤
          ▼        ▼         ▼
     ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
     │  Personal Model  │  │  Global Model    │  │ Background Scheduler │
     │ (User-specific)  │  │ (All-user learn) │  │ (Emails + Re-train)  │
     │ RandomForest +   │  │ RandomForest +   │  │ Runs every 2 mins    │
     │ Logistic Reg.    │  │ Logistic Reg.    │  └─────────────────────┘
     └─────────────────┘  └──────────────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
         ┌───────────────────┐
         │ Prediction Engine │
         └───────────────────┘
                │
                ▼
     ┌──────────────────┐
     │   Web Frontend   │
     │ HTML + CSS + JS  │
     │ Bootstrap + Icons│
     └──────────────────┘
```

---

## ✨ Key Features

### 🔮 Smart AI Predictions

The system learns user behavior from a rich **7000+ row dataset** using:

**Models Used:**
- **Random Forest Classifier** → Predicts if item is needed today
- **Logistic Regression** → Predicts forget probability
- **K-Means Clustering** → Automatic day-type learning

**Computed Metrics:**
- Need Probability
- Forget Probability
- Final Priority Score = `need_probability × (0.7 + 0.3 × forget_risk)`

### 🌍 Global Model for New Users

**Problem:** New users have zero personal history → no training data.

**Solution:** We implemented a **GLOBAL MODEL TRAINED ON ALL USERS**

- Used when users have zero personal history
- Predictions follow population-level learned behavior
- **System auto-learns each day**: When real users generate new data, the scheduler re-trains the global model automatically

### 📧 Daily Email at 8 AM (Configurable)

Each user receives:
- Top required predicted items
- Forget risk scores
- Button to open the app
- **One-click "Mark Packed"** link with secure token

**Testing Configuration:** Scheduler runs every 2 minutes instead of 8AM:
```python
scheduler.add_job(send_daily_reminders, "interval", minutes=2)
```

### 📝 One-Click "Mark Packed" via Email

Email includes secure link:
```
/email/mark_packed?user=<user_id>&date=<today>&token=<sha256_hash>
```
Users can update the system without logging in.

### 📊 AI Insights Dashboard

Users get:
- **Most forgotten items** (histogram)
- **Pack vs forget rate** (percentage)
- **Daily context patterns**
- **Model performance metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 🧪 What-If Simulation

Change:
- Weekday
- Holiday status
- Work day
- Gym day

→ The system simulates predictions using the same ML engine.

---

## 🗂 Dataset

We generated a rich **7,000+ row dataset** with realistic patterns:

| Column | Description |
|--------|-------------|
| `user_email` | Demo users (5-10 users) |
| `date` | 180 days of history |
| `weekday` | 0–6 (Monday–Sunday) |
| `is_holiday` | Weekend logic |
| `has_work_event` | Work day indicator |
| `has_gym_event` | Gym schedule per user |
| `item_name` | 13 essential items |
| `needed_label` | 1 if required, 0 otherwise |
| `packed` | 1 if user packed it, 0 otherwise |

### Items Tracked (13 Essential Items):
1. ID Card
2. Access Card
3. Laptop
4. Charger
5. Power Bank
6. Notebook
7. Lunch Box
8. Water Bottle
9. Pen
10. Headphones
11. Gym Shoes
12. Gym Towel
13. Umbrella

---

## ⚙️ Installation & Setup

### 1. Clone Project
```bash
git clone <repo-url>
cd daily-task-assistant
```

### 2. Setup Virtual Environment
```bash
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the `backend` folder:

```env
EMAIL_USER=yourgmail@gmail.com
EMAIL_PASSWORD=your_app_password
APP_BASE_URL=http://localhost:5000

# Optional: send all emails to one inbox for testing
TEST_EMAIL_TO=yourtest@gmail.com
```

### 5. Enable Gmail App Password

Gmail Setup:
1. Enable **2-factor authentication** on your Gmail account
2. Create an **App Password** for the application
3. Use the generated password in `EMAIL_PASSWORD`

[Gmail App Passwords Guide](https://support.google.com/accounts/answer/185833)

### 6. Seed Database (Optional)

To populate the database with sample data:

```bash
cd backend
python seed_from_csv.py
```

This will:
- Drop and recreate all DB tables
- Read CSV with 600+ records
- Create users, items, day contexts, and daily item statuses
- Train ML models for each user

**CSV Location:** `data/daily_task_history.csv`

---

## 🧠 Machine Learning Workflow

### 1. Training Phase

Two ML models are trained per user:

| Model | Purpose |
|-------|---------|
| **RandomForest** | Predict if item is needed today |
| **Logistic Regression** | Predict forget risk |

### 2. Global Model Strategy

- Trained from **all users' data combined**
- Used for:
  - New users without personal history
  - Cold-start prediction scenarios
- Automatically retrained daily by the scheduler

### 3. Prediction Formula

```
final_score = need_probability × (0.7 + 0.3 × forget_risk)
```

Items are sorted by score and displayed to the user.

### 4. Daily Auto-Learning (Scheduler)

**BackgroundScheduler** handles:
- ✔️ Sends reminder emails
- ✔️ Trains/updates global model
- ✔️ Ensures system improves automatically

**Default testing interval:** Every 2 minutes
```python
scheduler.add_job(send_daily_reminders, "interval", minutes=2)
```

---

## 🖥 Frontend

The UI is fully redesigned with **Bootstrap 5** + **Bootstrap Icons**:

### Pages:
- **Login / Register** - User authentication
- **Dashboard** - Daily checklist and quick predictions
- **Items Management** - Add, edit, delete items
- **History + AI Insights** - Historical data and performance metrics
- **Simulation Page** - What-if analysis tool
- **Profile / Settings** - User configuration

All pages feature:
- Modern card-based UI
- Responsive design
- Bootstrap icons
- Clean, intuitive navigation

---

## 🔗 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/items` | GET | List all items |
| `/api/checklist_today` | GET | Get today's checklist |
| `/api/checklist_update` | POST | Update packed status |
| `/api/train_model` | POST | Train personal model |
| `/api/train_global` | POST | Train global model |
| `/api/predict_today` | GET | Get predictions for today |
| `/api/insights` | GET | Get AI insights |
| `/api/simulate` | POST | Run what-if simulation |
| `/email/mark_packed` | GET | One-click email action |

---

## 🚀 Running the Application

### Start Backend Server

```bash
cd backend
python app.py
```

**Backend runs on:** `http://localhost:5000`

### Open in Browser

Navigate to:
```
http://localhost:5000
```

You'll see the full UI with:
- Login page
- Dashboard
- Items management
- AI insights
- All features working

---

## 🧪 Testing Email Feature

### Send Emails to Test Inbox

Set the test email environment variable:

```bash
# Windows (PowerShell):
$env:TEST_EMAIL_TO="yourtest@gmail.com"

# Windows (CMD):
set TEST_EMAIL_TO=yourtest@gmail.com

# macOS/Linux:
export EMAIL_USER=taskmemory.assistant@gmail.com
export EMAIL_PASSWORD=edtklslcbbiusbrz
export APP_BASE_URL=http://localhost:5000
export TEST_EMAIL_TO=yourEmail@gmail.com

```

Now all user emails will be redirected to your test inbox instead of individual user emails.

### Verify Scheduler

Check logs to confirm emails are being sent:
```
INFO: Sending reminder email to user_id=1
INFO: Training global model...
```

---

## 🧩 Team Distribution (Suggested for 3 Members)

### Member 1 — Backend + ML
- FastAPI/Flask API implementation
- Model training & prediction engine
- Global model logic & updates
- Email scheduler configuration
- Database schema & ORM

### Member 2 — Frontend UI
- HTML/CSS/JavaScript development
- Bootstrap 5 components & customization
- Dashboard, Insights, Simulation pages
- Responsive design & testing
- User experience refinement

### Member 3 — Data & Testing
- Dataset creation (7000+ rows)
- Seeding scripts & CSV generation
- API endpoint testing
- Model evaluation & metrics
- Performance benchmarking

---

## 📈 Results Summary (Example)

| Metric | Global Model | Personal Model |
|--------|--------------|----------------|
| **Accuracy** | 0.82 | 0.85+ |
| **Precision** | 0.79 | 0.84+ |
| **Recall** | 0.81 | 0.82+ |
| **F1 Score** | 0.80 | 0.83+ |

*These values populate dynamically based on your real dataset.*

---

## 📁 Project Structure

```
daily-task-assistant/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── database.py            # SQLAlchemy setup
│   ├── models.py              # SQLAlchemy models
│   ├── ml.py                  # ML training & prediction
│   ├── seed_from_csv.py       # Database seeding script
│   ├── requirements.txt        # Python dependencies
│   ├── venv/                  # Virtual environment
│   └── __init__.py            # Package initialization
│
├── frontend/
│   ├── templates/             # HTML templates
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── items.html
│   │   ├── history.html
│   │   └── simulate.html
│   ├── static/                # CSS, JS, images
│   │   ├── style.css
│   │   ├── dashboard.js
│   │   └── icons/
│   └── index.html             # Entry point
│
├── data/
│   └── daily_task_history.csv # Dataset (7000+ rows)
│
├── README.md                  # This file
└── .env                       # Environment variables (create manually)
```

---

## 🔐 Security Considerations

1. **Change Secret Key** in `app.py` for production
2. **Use HTTPS** in production environments
3. **Email Token** is SHA256-hashed for secure one-click updates
4. **Password Storage** uses werkzeug hashing
5. **CSRF Protection** implemented on all forms

---

## 🐛 Troubleshooting

### "CSV file not found"
- Ensure `data/daily_task_history.csv` exists in the project root
- Run: `python seed_from_csv.py`

### "Email not sending"
- Verify Gmail credentials in `.env`
- Check 2FA is enabled and App Password is generated
- Check `TEST_EMAIL_TO` is set if testing

### "Model training fails"
- Ensure enough data in database (minimum 50 rows per user)
- Check CSV format matches expected columns
- Verify pandas and scikit-learn are installed

### "Port 5000 already in use"
- Kill existing process: `lsof -ti:5000 | xargs kill -9` (macOS/Linux)
- Or use different port: `python app.py --port 5001`

---

## 📚 Dependencies

### Backend (`requirements.txt`)
```
Flask
Flask-Login
SQLAlchemy
pandas
scikit-learn
python-jose[cryptography]
APScheduler
python-dotenv
werkzeug
```

### Frontend
- Bootstrap 5
- Bootstrap Icons
- Vanilla JavaScript
- HTML5 + CSS3

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✔️ **Real-world ML engineering** - Model training, evaluation, deployment
- ✔️ **Full-stack development** - Backend API + Frontend UI
- ✔️ **Automated learning systems** - Scheduler & continuous improvement
- ✔️ **Intelligent reminder system** - Predictive analytics in action
- ✔️ **Email automation** - Integration with external services
- ✔️ **UX design** - Modern, responsive user interfaces
- ✔️ **Database design** - Relational schema with proper normalization
- ✔️ **API design** - RESTful endpoints with clear documentation

---

## 📝 License

This project is provided as-is for educational purposes.

---

## 👥 Contributors

- **Team Member 1** - Backend & ML
- **Team Member 2** - Frontend & UI
- **Team Member 3** - Data & Testing

---

## 📧 Support & Questions

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review **API Endpoints** documentation
3. Inspect **logs** in the terminal
4. Check `.env` configuration

---

## 🎯 Conclusion

The **Daily Task Memory Assistant** is a complete, production-ready AI application. It showcases:

- Practical machine learning implementation
- Full-stack web development
- Automated systems design
- Real-world problem solving
- Professional software engineering practices

