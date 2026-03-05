"""
AI Powered Student Performance Analyzer
Main Streamlit Application
Run with: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import io

from utils.data_processor import load_and_clean, engineer_features, subject_summary, get_recommendations
from models.ml_model import train_models, predict_student, batch_predict
from utils.visualizations import (
    bar_top10, subject_comparison, pass_fail_pie, risk_donut,
    scatter_attendance_score, heatmap_scores, weak_students_bar,
    feature_importance_bar,
)
from utils.report_generator import generate_html_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Student Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f0e17; }
    [data-testid="stSidebar"] { background: #1e1b4b; }
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border-radius: 14px; padding: 1.2rem 1.5rem;
        border: 1px solid #4338ca; margin-bottom: .5rem;
    }
    .metric-val { font-size: 2.2rem; font-weight: 800; color: #a5b4fc; }
    .metric-lbl { font-size: .85rem; color: #94a3b8; margin-top: 2px; }
    .risk-high  { color: #ef4444; font-weight: 700; }
    .risk-med   { color: #f59e0b; font-weight: 700; }
    .risk-low   { color: #10b981; font-weight: 700; }
    .section-header { color: #a5b4fc; font-size: 1.3rem; font-weight: 700;
                       border-left: 4px solid #6366f1; padding-left: .7rem; margin: 1.5rem 0 .8rem; }
    .rec-box { background: #1e1b4b; border-radius: 10px; padding: 1rem 1.2rem;
                border-left: 4px solid #6366f1; margin-bottom: .5rem; }
    .pred-pass { background: #064e3b; border-radius: 10px; padding: 1rem; border: 1px solid #10b981; }
    .pred-fail { background: #450a0a; border-radius: 10px; padding: 1rem; border: 1px solid #ef4444; }
    div[data-testid="stDataFrame"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 AI Student Analyzer")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
    st.markdown("---")
    page = st.radio("📑 Navigate", [
        "🏠 Overview",
        "📊 Analytics Dashboard",
        "🧠 AI Predictions",
        "👤 Student Lookup",
        "📄 Generate Report",
    ])
    st.markdown("---")
    st.markdown("**About**")
    st.caption("Analyzes student performance using ML models (Logistic Regression & Decision Tree) to predict pass/fail outcomes.")


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data(file_bytes: bytes) -> tuple:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    cleaned = load_and_clean(raw)
    processed = engineer_features(cleaned)
    subj = subject_summary(processed)
    return raw, processed, subj


@st.cache_data
def get_models(file_bytes: bytes):
    _, processed, _ = get_data(file_bytes)
    return train_models(processed)


if uploaded:
    file_bytes = uploaded.read()
    raw_df, df, subj_df = get_data(file_bytes)
    models = get_models(file_bytes)
    df_pred = batch_predict(models, df)
else:
    # Load sample data by default
    sample_path = os.path.join(os.path.dirname(__file__), "data", "students.csv")
    with open(sample_path, "rb") as f:
        file_bytes = f.read()
    raw_df, df, subj_df = get_data(file_bytes)
    models = get_models(file_bytes)
    df_pred = batch_predict(models, df)
    st.sidebar.info("📌 Using sample dataset. Upload your own CSV above.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🎓 AI Powered Student Performance Analyzer")
    st.markdown("##### Real-time analytics & ML-powered predictions for student success")
    st.markdown("---")

    total = len(df)
    passed = (df["Pass_Fail"] == "Pass").sum()
    failed = total - passed
    pass_rate = round(passed / total * 100, 1)
    avg_score = round(df["Average_Score"].mean(), 2)
    high_risk = (df["Risk_Level"] == "High").sum()
    top_student = df.nsmallest(1, "Rank").iloc[0]["Student_Name"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, str(total), "Total Students", "👥"),
        (c2, str(passed), "Passed", "✅"),
        (c3, str(failed), "Failed", "❌"),
        (c4, f"{pass_rate}%", "Pass Rate", "📈"),
        (c5, str(avg_score), "Class Avg", "🎯"),
        (c6, str(high_risk), "At Risk", "⚠️"),
    ]
    for col, val, lbl, icon in metrics:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{icon} {val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(pass_fail_pie(df), use_container_width=True)
    with col_r:
        st.plotly_chart(risk_donut(df), use_container_width=True)

    st.markdown(f'<div class="section-header">🏆 Top Performer: {top_student}</div>', unsafe_allow_html=True)
    st.plotly_chart(bar_top10(df), use_container_width=True)

    st.markdown('<div class="section-header">📋 Full Student Table</div>', unsafe_allow_html=True)
    display_cols = ["Rank", "Student_Name", "Average_Score", "Pass_Fail", "Risk_Level", "Performance_Category"]
    st.dataframe(df[display_cols].set_index("Rank"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Analytics Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    st.plotly_chart(subject_comparison(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(scatter_attendance_score(df), use_container_width=True)
    with c2:
        st.plotly_chart(weak_students_bar(df), use_container_width=True)

    st.plotly_chart(heatmap_scores(df), use_container_width=True)

    st.markdown('<div class="section-header">📚 Subject-wise Statistics</div>', unsafe_allow_html=True)
    st.dataframe(subj_df.set_index("Subject"), use_container_width=True)

    st.markdown('<div class="section-header">🚨 High-Risk Students</div>', unsafe_allow_html=True)
    risk_cols = ["Student_Name", "Average_Score", "Attendance", "Study_Hours", "Pass_Fail"]
    at_risk = df[df["Risk_Level"] == "High"][risk_cols]
    if at_risk.empty:
        st.success("🎉 No high-risk students found!")
    else:
        st.dataframe(at_risk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI Predictions
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 AI Predictions":
    st.title("🧠 AI Prediction Dashboard")
    st.markdown("Predicts whether each student will pass or fail their **next assessment**.")
    st.markdown("---")

    if "error" in models:
        st.error(f"Model error: {models['error']}")
    else:
        # Model accuracy cards
        lr = models.get("logistic_regression", {})
        dt = models.get("decision_tree", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">📐 {lr.get('cv_accuracy', 'N/A')}%</div>
                <div class="metric-lbl">Logistic Regression — CV Accuracy (±{lr.get('cv_std', '')}%)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">🌳 {dt.get('cv_accuracy', 'N/A')}%</div>
                <div class="metric-lbl">Decision Tree — CV Accuracy (±{dt.get('cv_std', '')}%)</div>
            </div>""", unsafe_allow_html=True)

        # Feature importance
        if "importances" in dt:
            st.plotly_chart(feature_importance_bar(dt["importances"]), use_container_width=True)

        # Prediction table
        st.markdown('<div class="section-header">📋 Batch Predictions — All Students</div>', unsafe_allow_html=True)
        pred_cols = ["Student_Name", "Average_Score", "Pass_Fail"]
        if "LR_Prediction" in df_pred.columns:
            pred_cols += ["LR_Pass_Prob_%", "LR_Prediction"]
        if "DT_Prediction" in df_pred.columns:
            pred_cols += ["DT_Pass_Prob_%", "DT_Prediction"]
        pred_cols.append("Risk_Level")
        st.dataframe(df_pred[pred_cols], use_container_width=True)

        # How it works
        with st.expander("🔍 How the AI Models Work"):
            st.markdown("""
**Logistic Regression**
- A statistical model that estimates the probability of Pass/Fail using a sigmoid function.
- Learns a weighted combination of features (scores, attendance, study hours).
- Great for interpretable, linear decision boundaries.
- Output: probability score between 0–100% for passing.

**Decision Tree**
- Learns a series of if/else rules from training data.
- More intuitive — "If Python score < 50 AND Attendance < 70% → likely Fail"
- Feature importances show which factors matter most.
- `max_depth=4` prevents overfitting.

**Features used:**
- Python Score, Data Analytics Score, SQL Score
- Attendance percentage
- Weekly study hours

**Evaluation:** 5-fold cross-validation — the dataset is split 5 times, ensuring accuracy is measured on unseen data.
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Student Lookup
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Student Lookup":
    st.title("👤 Student Lookup & Profile")
    st.markdown("---")

    names = df["Student_Name"].tolist()
    selected = st.selectbox("Select a student:", names)
    student = df[df["Student_Name"] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    score_cols = [c for c in ["Python_Score", "Data_Analytics_Score", "SQL_Score"] if c in df.columns]
    colors_metric = ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"]
    for i, col in enumerate(score_cols + ["Average_Score", "Attendance"]):
        label = col.replace("_Score", "").replace("_", " ")
        val = student[col]
        [c1, c2, c3, c4][i % 4].metric(label, f"{val}{'%' if col == 'Attendance' else ''}")

    st.markdown("---")
    c_left, c_right = st.columns(2)

    risk = student["Risk_Level"]
    risk_class = {"High": "risk-high", "Medium": "risk-med", "Low": "risk-low"}.get(risk, "")
    with c_left:
        st.markdown(f"""
        **Rank:** #{student['Rank']} out of {len(df)}  
        **Status:** {'✅ Pass' if student['Pass_Fail'] == 'Pass' else '❌ Fail'}  
        **Risk Level:** <span class="{risk_class}">{risk}</span>  
        **Performance:** {student['Performance_Category']}  
        **Study Hours/week:** {student.get('Study_Hours', 'N/A')}h
        """, unsafe_allow_html=True)

    with c_right:
        if "error" not in models:
            preds = predict_student(models, student)
            for model_name, pred in preds.items():
                label = "Logistic Regression" if "logistic" in model_name else "Decision Tree"
                is_pass = pred["prediction"] == "Pass"
                box_class = "pred-pass" if is_pass else "pred-fail"
                icon = "✅" if is_pass else "❌"
                st.markdown(f"""<div class="{box_class}">
                    <strong>{label}</strong><br>
                    {icon} <strong>{pred['prediction']}</strong> &nbsp;|&nbsp;
                    Pass prob: <strong>{pred['pass_probability']}%</strong>
                </div><br>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
    recs = get_recommendations(student)
    for rec in recs:
        st.markdown(f'<div class="rec-box">💡 {rec}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Generate Report
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Generate Report":
    st.title("📄 Performance Report Generator")
    st.markdown("---")

    st.markdown("Generate a complete HTML performance report for the class.")

    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("Building report..."):
            html = generate_html_report(df, subj_df)

        st.success("✅ Report generated!")
        st.download_button(
            label="⬇️ Download HTML Report",
            data=html,
            file_name="student_performance_report.html",
            mime="text/html",
        )

        st.markdown("---")
        st.markdown("**Preview:**")
        st.components.v1.html(html, height=700, scrolling=True)
