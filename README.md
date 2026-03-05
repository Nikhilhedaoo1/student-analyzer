# 🎓 AI Powered Student Performance Analyzer

A complete Python project for analyzing student assessment data with machine learning predictions and an interactive Streamlit dashboard.

---

## 📁 Folder Structure

```
student_analyzer/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   └── students.csv                # Sample dataset (30 students)
│
├── models/
│   ├── __init__.py
│   └── ml_model.py                 # Logistic Regression & Decision Tree
│
└── utils/
    ├── __init__.py
    ├── data_processor.py           # Data cleaning & feature engineering
    ├── visualizations.py           # Plotly chart functions
    └── report_generator.py         # HTML report builder
```

---

## ⚙️ Setup & Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
cd student_analyzer
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📊 Dataset Format

Your CSV must have these columns (case-sensitive):

| Column | Description | Range |
|---|---|---|
| `Student_ID` | Unique identifier | e.g. S001 |
| `Student_Name` | Full name | Text |
| `Python_Score` | Python assessment score | 0–100 |
| `Data_Analytics_Score` | Data Analytics score | 0–100 |
| `SQL_Score` | SQL assessment score | 0–100 |
| `Attendance` | Attendance percentage | 0–100 |
| `Study_Hours` | Weekly study hours | 0–24 |

Missing values are handled automatically.

---

## 🧠 How the AI Models Work

### Pass/Fail Logic
A student **Passes** if:
- Average score ≥ 60, AND
- No individual subject score < 40

Otherwise → **Fail**

### Logistic Regression
- Scales features with `StandardScaler`
- Fits a sigmoid boundary: `P(Pass) = 1 / (1 + e^(-z))`
- Output: probability 0–100% for passing
- Best for: interpretable linear decision boundaries

### Decision Tree
- Learns if/else rules from data (e.g., "Python < 50 AND Attendance < 70 → Fail")
- `max_depth=4` prevents overfitting
- Feature importances show which factors drive predictions most
- Best for: explainable, non-linear decisions

### Evaluation
Both models use **5-fold cross-validation**:
- Dataset split into 5 parts
- Trained on 4, tested on 1 — repeated 5 times
- Reports mean accuracy ± standard deviation

---

## 🗂️ Dashboard Pages

| Page | Features |
|---|---|
| 🏠 Overview | KPI cards, pass/fail pie, risk donut, top students table |
| 📊 Analytics | Subject comparison, scatter plot, heatmap, weak students |
| 🧠 AI Predictions | Model accuracy, feature importance, batch predictions |
| 👤 Student Lookup | Individual profile, AI predictions, improvement tips |
| 📄 Generate Report | Downloadable HTML class report |

---

## ⚠️ Risk Classification

| Risk Level | Criteria |
|---|---|
| 🔴 High | Score < 50, OR attendance < 60%, OR study hours < 3 |
| 🟡 Medium | Score 50–64, OR attendance 60–74%, OR study hours < 5 |
| 🟢 Low | All metrics healthy |

---

## 📦 Tech Stack

- **Streamlit** — Web dashboard
- **Pandas** — Data processing
- **Scikit-learn** — ML models
- **Plotly** — Interactive charts
- **NumPy** — Numerical operations
