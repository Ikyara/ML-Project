import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stApp"], [data-testid="stBottom"], .main, .block-container,
    [data-testid="stMainBlockContainer"], [data-testid="stVerticalBlock"],
    [data-testid="stAppViewBlockContainer"], .appview-container,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], div[data-testid="collapsedControl"],
    .stDeployButton, [data-testid="baseButton-header"],
    [data-testid="stToolbarActions"], header[data-testid="stHeader"],
    div[data-testid="stSidebarCollapsedControl"] {
        background-color: #0e0b1e !important;
        color-scheme: dark !important;
    }
    /* Force overscroll color */
    html {
        overscroll-behavior: none;
        scrollbar-color: #2d2654 #0e0b1e;
        background: #0e0b1e !important;
    }
    body {
        background: #0e0b1e !important;
        overflow-y: auto;
    }
    body::before, body::after {
        background-color: #0e0b1e !important;
    }
    /* Kill the top colored decoration bar */
    [data-testid="stDecoration"] {
        display: none !important;
        background: none !important;
        height: 0 !important;
    }
    /* Header bar background */
    header[data-testid="stHeader"] {
        background: #0e0b1e !important;
        backdrop-filter: none !important;
    }
    /* Top header bar */
    [data-testid="stHeader"] {
        background: #0e0b1e !important;
        backdrop-filter: none !important;
    }
    .stApp {
        background: #0e0b1e;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #140f2d 0%, #0e0b1e 100%);
        border-right: 1px solid rgba(168, 85, 247, 0.15);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: rgba(168, 85, 247, 0.05);
        border-radius: 14px;
        padding: 4px;
        border: 1px solid rgba(168, 85, 247, 0.12);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        color: #8b7fb8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #a855f7, #ec4899) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.12), rgba(236, 72, 153, 0.08), rgba(168, 85, 247, 0.05));
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 20px;
        padding: 35px 40px;
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a855f7, #ec4899, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #9d8ec7;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.05));
        border: 1px solid rgba(168, 85, 247, 0.18);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .metric-card:hover {
        border-color: rgba(168, 85, 247, 0.35);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0;
    }
    .metric-label {
        color: #7a6fa5;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* Grade display */
    .grade-container {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(236, 72, 153, 0.08));
        border: 2px solid rgba(168, 85, 247, 0.3);
        border-radius: 24px;
        padding: 45px 30px;
        text-align: center;
    }
    .grade-number {
        font-size: 5.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a855f7, #ec4899, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .grade-subtitle {
        color: #7a6fa5;
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 8px;
    }

    /* Status */
    .status-excellent {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05));
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
        padding: 14px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin-top: 15px;
    }
    .status-good {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(168, 85, 247, 0.05));
        border: 1px solid rgba(168, 85, 247, 0.3);
        color: #c084fc;
        padding: 14px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin-top: 15px;
    }
    .status-risk {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.15), rgba(236, 72, 153, 0.05));
        border: 1px solid rgba(236, 72, 153, 0.3);
        color: #f472b6;
        padding: 14px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin-top: 15px;
    }

    /* Info cards */
    .info-card {
        background: rgba(168, 85, 247, 0.06);
        border: 1px solid rgba(168, 85, 247, 0.12);
        border-radius: 14px;
        padding: 22px;
        margin: 10px 0;
        height: 100%;
    }
    .info-card-title {
        color: #c084fc;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }
    .info-card-text {
        color: #b0a3d4;
        font-size: 0.88rem;
        line-height: 1.65;
    }

    /* Stat boxes */
    .stat-box {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.08), rgba(236, 72, 153, 0.04));
        border: 1px solid rgba(168, 85, 247, 0.15);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #c084fc;
        margin: 6px 0;
    }
    .stat-label {
        color: #7a6fa5;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Section headers */
    .section-head {
        color: #e0d4f7;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(168, 85, 247, 0.15);
    }
    .section-desc {
        color: #8b7fb8;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    /* Sidebar */
    .sidebar-section {
        color: #c084fc;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 18px 0 8px 0;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #ec4899);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 40px;
        font-size: 1.05rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #b866ff, #f05aa8);
        box-shadow: 0 6px 25px rgba(168, 85, 247, 0.35);
        transform: translateY(-1px);
    }

    /* Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }
    .styled-table th {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        padding: 12px 16px;
        text-align: left;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid rgba(168, 85, 247, 0.2);
    }
    .styled-table td {
        padding: 11px 16px;
        color: #b0a3d4;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(168, 85, 247, 0.06);
    }
    .styled-table tr:hover td {
        background: rgba(168, 85, 247, 0.04);
    }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Train model if needed ────────────────────────────────────────────────────
def train_and_save():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    por = pd.read_csv('student-por.csv', sep=';')
    mat = pd.read_csv('student-mat.csv', sep=';')
    df = pd.concat([por, mat], ignore_index=True)
    df['higher_yes'] = (df['higher'] == 'yes').astype(int)

    FEATURES = ['G1', 'G2', 'absences', 'studytime', 'failures', 'higher_yes']
    X = df[FEATURES].copy()
    y = df['G3'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)

    # Save evaluation metrics too
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rf_pred = rf_model.predict(X_test_scaled)
    metrics = {
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'r2': r2_score(y_test, rf_pred),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'y_test': y_test.values.tolist(),
        'y_pred': rf_pred.tolist(),
    }
    n = X_test.shape[0]
    p = X_test.shape[1]
    metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)

    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(FEATURES, f)
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

if not all(os.path.exists(f) for f in ['random_forest_model.pkl', 'scaler.pkl', 'feature_columns.pkl', 'metrics.pkl']):
    with st.spinner('🔧 Training model for the first time...'):
        train_and_save()

# ── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    with open('metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return model, scaler, feature_columns, metrics

model, scaler, feature_columns, metrics = load_artifacts()

# ── Load raw data for visualizations ─────────────────────────────────────────
@st.cache_data
def load_raw_data():
    por = pd.read_csv('student-por.csv', sep=';')
    mat = pd.read_csv('student-mat.csv', sep=';')
    por['subject'] = 'Portuguese'
    mat['subject'] = 'Math'
    df = pd.concat([por, mat], ignore_index=True)
    return df

df_raw = load_raw_data()

# ── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 5px 0;'>
        <span style='font-size: 2.2rem;'>🎓</span>
        <h2 style='background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin: 5px 0;'>Grade Predictor</h2>
        <p style='color: #7a6fa5; font-size: 0.82rem;'>Adjust student parameters below</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='sidebar-section'>📊 Academic Grades</div>", unsafe_allow_html=True)
    G1 = st.slider('First Period Grade (G1)', 0, 20, 10)
    G2 = st.slider('Second Period Grade (G2)', 0, 20, 10)

    st.markdown("<div class='sidebar-section'>📚 Habits & Attendance</div>", unsafe_allow_html=True)
    studytime = st.slider('Weekly Study Time', 1, 4, 2, help="1: <2hrs · 2: 2-5hrs · 3: 5-10hrs · 4: >10hrs")
    absences = st.number_input('Number of Absences', 0, 30, 5)

    st.markdown("<div class='sidebar-section'>📋 Background</div>", unsafe_allow_html=True)
    failures = st.slider('Past Class Failures', 0, 4, 0)
    higher_val = st.radio('Wants Higher Education?', ['Yes', 'No'], index=0)

    st.markdown("---")
    predict_btn = st.button('🔮 Predict Final Grade', use_container_width=True)

# ── Helper ───────────────────────────────────────────────────────────────────
def create_input_df(G1, G2, absences, studytime, failures, higher_val, feature_columns):
    input_data = {
        'G1': G1, 'G2': G2, 'absences': absences,
        'studytime': studytime, 'failures': failures,
        'higher_yes': 1 if higher_val == 'Yes' else 0,
    }
    return pd.DataFrame([input_data])[feature_columns]

input_df = create_input_df(G1, G2, absences, studytime, failures, higher_val, feature_columns)

plotly_layout = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#9d8ec7', family='Inter'),
    margin=dict(l=40, r=30, t=40, b=40),
)

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🎓 Student Final Grade Predictor</div>
    <p class='hero-subtitle'>
        This application uses a <strong>Random Forest Regressor</strong> trained on the 
        <strong>UCI Student Performance Dataset</strong> to predict a student's final grade (G3) on a 0–20 scale.
        The model analyzes academic scores, study habits, attendance, and background factors to generate predictions.
        Use the sidebar to input student details and explore the tabs below for data insights, model performance, and predictions.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Dataset Explorer", "🧠 Model Performance", "📖 About"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    # Input summary
    st.markdown("<div class='section-head'>📋 Input Summary</div>", unsafe_allow_html=True)
    st.markdown("<p class='section-desc'>These are the current values you've set in the sidebar. Click <strong>Predict Final Grade</strong> to see the result.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        ("G1", G1, "Period 1"), ("G2", G2, "Period 2"), ("Study", studytime, "hrs/week"),
        ("Absences", absences, "total"), ("Failures", failures, "past"),
        ("Higher Ed", "✓" if higher_val == "Yes" else "✗", higher_val)
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4, c5, c6], cards):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    if predict_btn:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        predicted_grade = float(np.clip(prediction[0], 0, 20))
        percentage = (predicted_grade / 20) * 100

        if predicted_grade >= 16: letter = "A"
        elif predicted_grade >= 14: letter = "B"
        elif predicted_grade >= 12: letter = "C"
        elif predicted_grade >= 10: letter = "D"
        else: letter = "F"

        st.markdown("---")

        # Grade + Charts side by side
        grade_col, chart_col = st.columns([1, 1.5])

        with grade_col:
            st.markdown(f"""<div class='grade-container'>
                <div class='grade-subtitle'>Predicted Final Grade (G3)</div>
                <div class='grade-number'>{predicted_grade:.1f}</div>
                <div class='grade-subtitle'>out of 20 · {percentage:.0f}% · Grade {letter}</div>
            </div>""", unsafe_allow_html=True)

            if predicted_grade >= 16:
                st.markdown("<div class='status-excellent'>🌟 Excellent — Outstanding performance predicted</div>", unsafe_allow_html=True)
            elif predicted_grade >= 14:
                st.markdown("<div class='status-excellent'>✅ Very Good — Strong academic trajectory</div>", unsafe_allow_html=True)
            elif predicted_grade >= 10:
                st.markdown("<div class='status-good'>📘 Satisfactory — On track to pass</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-risk'>⚠️ At Risk — Intervention recommended</div>", unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_grade,
                number=dict(suffix="/20", font=dict(size=28, color='#c084fc')),
                gauge=dict(
                    axis=dict(range=[0, 20], tickcolor='#7a6fa5', tickfont=dict(color='#7a6fa5')),
                    bar=dict(color='#a855f7'),
                    bgcolor='rgba(168, 85, 247, 0.05)',
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 10], color='rgba(236, 72, 153, 0.12)'),
                        dict(range=[10, 14], color='rgba(168, 85, 247, 0.12)'),
                        dict(range=[14, 20], color='rgba(34, 197, 94, 0.12)'),
                    ],
                    threshold=dict(line=dict(color='#ec4899', width=3), thickness=0.8, value=predicted_grade),
                ),
            ))
            gauge_layout = {k: v for k, v in plotly_layout.items() if k != 'margin'}
            fig_gauge.update_layout(**gauge_layout, height=250, margin=dict(l=30, r=30, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_col:
            # Grade progression bar chart
            st.markdown("<div class='section-head'>📈 Grade Progression</div>", unsafe_allow_html=True)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=['G1 (Period 1)', 'G2 (Period 2)', 'G3 (Predicted)'],
                y=[G1, G2, predicted_grade],
                marker=dict(
                    color=['#a855f7', '#c084fc', '#ec4899'],
                    line=dict(width=0),
                ),
                text=[f"{G1}", f"{G2}", f"{predicted_grade:.1f}"],
                textposition='outside',
                textfont=dict(color='#e0d4f7', size=15, family='Inter'),
            ))
            fig_bar.update_layout(
                **plotly_layout, height=280,
                yaxis=dict(range=[0, 22], gridcolor='rgba(168,85,247,0.08)', title='Grade'),
                xaxis=dict(title=''),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Feature importance
            st.markdown("<div class='section-head'>🧠 Feature Impact</div>", unsafe_allow_html=True)
            importances = model.feature_importances_
            feat_labels = ['G1', 'G2', 'Absences', 'Study Time', 'Failures', 'Higher Ed']
            feat_df = pd.DataFrame({'Feature': feat_labels, 'Importance': importances}).sort_values('Importance', ascending=True)

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=feat_df['Importance'], y=feat_df['Feature'], orientation='h',
                marker=dict(color=feat_df['Importance'], colorscale=[[0, '#ec4899'], [0.5, '#a855f7'], [1, '#c084fc']]),
                text=[f"{v:.1%}" for v in feat_df['Importance']],
                textposition='outside',
                textfont=dict(color='#e0d4f7', size=12),
            ))
            fig_imp.update_layout(
                **plotly_layout, height=250,
                xaxis=dict(gridcolor='rgba(168,85,247,0.08)', title='Importance', range=[0, max(importances) * 1.35]),
                yaxis=dict(title=''),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        # Insights row
        st.markdown("<div class='section-head'>💡 Detailed Insights</div>", unsafe_allow_html=True)

        i1, i2, i3 = st.columns(3)
        trend = predicted_grade - G2
        trend_icon = "📈" if trend > 0.5 else ("📉" if trend < -0.5 else "➡️")
        trend_word = f"improving by {trend:.1f} points" if trend > 0.5 else (f"declining by {abs(trend):.1f} points" if trend < -0.5 else "holding steady")

        with i1:
            st.markdown(f"""<div class='info-card'>
                <div class='info-card-title'>{trend_icon} Grade Trajectory</div>
                <div class='info-card-text'>
                    The student's grades are <strong>{trend_word}</strong> from G2 to the predicted G3.
                    The G1→G2 shift was <strong>{'+' if G2 - G1 >= 0 else ''}{G2 - G1} points</strong>.
                    {'The upward trend suggests growing understanding of the material.' if trend > 0.5 else ('A downward trend may indicate disengagement or increased difficulty.' if trend < -0.5 else 'Consistent performance across periods.')}
                </div>
            </div>""", unsafe_allow_html=True)

        abs_icon = "✅" if absences <= 5 else ("⚠️" if absences <= 15 else "🚨")
        with i2:
            avg_abs = df_raw['absences'].mean()
            st.markdown(f"""<div class='info-card'>
                <div class='info-card-title'>{abs_icon} Attendance Analysis</div>
                <div class='info-card-text'>
                    <strong>{absences} absences</strong> recorded (dataset average: {avg_abs:.1f}).
                    {'This is well below average — great attendance!' if absences <= 5 else ('This is around the average. Reducing absences could help improve performance.' if absences <= 15 else f'This is {absences - avg_abs:.0f} above average and is likely a significant drag on academic performance. Each missed class is a missed learning opportunity.')}
                </div>
            </div>""", unsafe_allow_html=True)

        study_labels = {1: "less than 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "over 10 hours"}
        with i3:
            st.markdown(f"""<div class='info-card'>
                <div class='info-card-title'>📚 Study & Background</div>
                <div class='info-card-text'>
                    Studying <strong>{study_labels[studytime]}</strong> per week.
                    {'Excellent study commitment — this strongly supports good outcomes.' if studytime >= 3 else ('Moderate effort. Increasing study hours is one of the most effective ways to boost grades.' if studytime == 2 else 'Minimal study time — this is the biggest area for improvement.')}
                    {' No past failures — clean record.' if failures == 0 else f' {failures} past failure(s) may affect confidence.'} 
                    {'Motivation for higher education is a strong positive signal.' if higher_val == 'Yes' else ''}
                </div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align: center; padding: 80px 20px; color: #7a6fa5;'>
            <p style='font-size: 3.5rem; margin-bottom: 10px;'>🔮</p>
            <p style='font-size: 1.3rem; font-weight: 700; color: #c084fc;'>Ready to predict</p>
            <p style='font-size: 0.95rem; margin-top: 8px;'>Adjust the student parameters in the sidebar, then click <strong>Predict Final Grade</strong></p>
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: DATASET EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("<div class='section-head'>📊 About the Dataset</div>", unsafe_allow_html=True)
    st.markdown("""<p class='section-desc'>
        The <strong>UCI Student Performance Dataset</strong> contains data from two Portuguese secondary schools, 
        covering students in Math and Portuguese language courses. It was collected via school reports and questionnaires 
        and includes demographic, social, and academic attributes. The dataset captures 33 features per student 
        including family background, lifestyle habits, and three grading periods (G1, G2, G3). 
        G3 (the final grade) is our prediction target, scored on a 0–20 scale where 10 is passing.
    </p>""", unsafe_allow_html=True)

    # Dataset stats
    s1, s2, s3, s4, s5 = st.columns(5)
    stat_items = [
        ("Total Students", len(df_raw)),
        ("Features", "33"),
        ("Math Students", len(df_raw[df_raw['subject'] == 'Math'])),
        ("Portuguese", len(df_raw[df_raw['subject'] == 'Portuguese'])),
        ("Avg Final Grade", f"{df_raw['G3'].mean():.1f}"),
    ]
    for col, (label, val) in zip([s1, s2, s3, s4, s5], stat_items):
        with col:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-label'>{label}</div>
                <div class='stat-value'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Charts row 1
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("<div class='section-head'>🎯 Final Grade (G3) Distribution</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>How final grades are distributed across all students. The dashed line marks the passing threshold (10/20).</p>", unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_raw['G3'], nbinsx=21,
            marker=dict(color='#a855f7', line=dict(color='#c084fc', width=1)),
            opacity=0.85,
        ))
        fig_hist.add_vline(x=10, line_dash="dash", line_color="#ec4899", annotation_text="Passing (10)",
                           annotation_font_color="#ec4899")
        fig_hist.update_layout(**plotly_layout, height=350, xaxis_title='Final Grade (G3)', yaxis_title='Number of Students')
        fig_hist.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig_hist.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig_hist, use_container_width=True)

    with ch2:
        st.markdown("<div class='section-head'>📚 Grade by Subject</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>Comparing final grade distributions between Math and Portuguese courses. Portuguese students tend to score slightly higher on average.</p>", unsafe_allow_html=True)
        fig_box = go.Figure()
        for subj, color in [('Math', '#ec4899'), ('Portuguese', '#a855f7')]:
            fig_box.add_trace(go.Box(
                y=df_raw[df_raw['subject'] == subj]['G3'], name=subj,
                marker_color=color, line_color=color,
                boxmean=True,
            ))
        fig_box.update_layout(**plotly_layout, height=350, yaxis_title='Final Grade (G3)', showlegend=True,
                              legend=dict(font=dict(color='#9d8ec7')))
        fig_box.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig_box, use_container_width=True)

    # Charts row 2
    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("<div class='section-head'>📈 G1 vs G2 vs G3 Correlation</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>The three grading periods are highly correlated. G2 is the strongest predictor of G3, which is why it dominates the model's feature importance.</p>", unsafe_allow_html=True)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=df_raw['G2'], y=df_raw['G3'], mode='markers',
            marker=dict(color='#a855f7', opacity=0.4, size=6),
            name='G2 vs G3'
        ))
        fig_scatter.add_trace(go.Scatter(
            x=[0, 20], y=[0, 20], mode='lines',
            line=dict(color='#ec4899', dash='dash', width=2),
            name='Perfect correlation'
        ))
        fig_scatter.update_layout(**plotly_layout, height=350, xaxis_title='G2 (Period 2)', yaxis_title='G3 (Final)',
                                  legend=dict(font=dict(color='#9d8ec7')))
        fig_scatter.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig_scatter.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with ch4:
        st.markdown("<div class='section-head'>🏫 Absences vs Final Grade</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>While the relationship isn't perfectly linear, students with very high absences tend to perform worse. Many high-performing students have low absence counts.</p>", unsafe_allow_html=True)
        fig_abs = go.Figure()
        fig_abs.add_trace(go.Scatter(
            x=df_raw['absences'], y=df_raw['G3'], mode='markers',
            marker=dict(color='#ec4899', opacity=0.35, size=6),
        ))
        fig_abs.update_layout(**plotly_layout, height=350, xaxis_title='Absences', yaxis_title='Final Grade (G3)',
                              showlegend=False)
        fig_abs.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig_abs.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig_abs, use_container_width=True)

    # Charts row 3
    ch5, ch6 = st.columns(2)

    with ch5:
        st.markdown("<div class='section-head'>⏱️ Study Time vs Final Grade</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>Students who study more tend to get better grades on average, though there's significant variance at every level.</p>", unsafe_allow_html=True)
        study_avg = df_raw.groupby('studytime')['G3'].agg(['mean', 'std', 'count']).reset_index()
        study_labels_map = {1: '<2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '>10 hrs'}
        study_avg['label'] = study_avg['studytime'].map(study_labels_map)

        fig_study = go.Figure()
        fig_study.add_trace(go.Bar(
            x=study_avg['label'], y=study_avg['mean'],
            marker=dict(color=['#ec4899', '#c084fc', '#a855f7', '#7c3aed']),
            text=[f"{v:.1f}" for v in study_avg['mean']],
            textposition='outside',
            textfont=dict(color='#e0d4f7', size=13),
            error_y=dict(type='data', array=study_avg['std'].tolist(), color='#7a6fa5', thickness=1.5),
        ))
        fig_study.update_layout(**plotly_layout, height=350, yaxis_title='Average G3', xaxis_title='Weekly Study Time',
                                yaxis=dict(range=[0, 18], gridcolor='rgba(168,85,247,0.08)'))
        st.plotly_chart(fig_study, use_container_width=True)

    with ch6:
        st.markdown("<div class='section-head'>❌ Failures vs Final Grade</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>Past failures have a notable negative relationship with final grades. Students with 0 failures average significantly higher.</p>", unsafe_allow_html=True)
        fail_avg = df_raw.groupby('failures')['G3'].agg(['mean', 'count']).reset_index()

        fig_fail = go.Figure()
        fig_fail.add_trace(go.Bar(
            x=[str(f) for f in fail_avg['failures']], y=fail_avg['mean'],
            marker=dict(color=['#22c55e', '#c084fc', '#ec4899', '#f43f5e', '#dc2626'][:len(fail_avg)]),
            text=[f"{v:.1f} (n={c})" for v, c in zip(fail_avg['mean'], fail_avg['count'])],
            textposition='outside',
            textfont=dict(color='#e0d4f7', size=12),
        ))
        fig_fail.update_layout(**plotly_layout, height=350, yaxis_title='Average G3', xaxis_title='Number of Past Failures',
                               yaxis=dict(range=[0, 16], gridcolor='rgba(168,85,247,0.08)'))
        st.plotly_chart(fig_fail, use_container_width=True)

    # Feature description table
    st.markdown("<div class='section-head'>📝 Feature Descriptions</div>", unsafe_allow_html=True)
    st.markdown("<p class='section-desc'>Complete description of the 6 features used by our prediction model.</p>", unsafe_allow_html=True)
    st.markdown("""
    <table class='styled-table'>
        <tr><th>Feature</th><th>Description</th><th>Range</th><th>Type</th></tr>
        <tr><td><strong>G1</strong></td><td>First period grade — the earliest academic indicator</td><td>0 – 20</td><td>Numeric</td></tr>
        <tr><td><strong>G2</strong></td><td>Second period grade — strongest predictor of final performance</td><td>0 – 20</td><td>Numeric</td></tr>
        <tr><td><strong>Absences</strong></td><td>Total number of school absences throughout the year</td><td>0 – 30</td><td>Numeric</td></tr>
        <tr><td><strong>Study Time</strong></td><td>Weekly study time: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)</td><td>1 – 4</td><td>Ordinal</td></tr>
        <tr><td><strong>Failures</strong></td><td>Number of past class failures</td><td>0 – 4</td><td>Numeric</td></tr>
        <tr><td><strong>Higher Education</strong></td><td>Whether the student aspires to pursue higher education</td><td>Yes / No</td><td>Binary</td></tr>
    </table>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("<div class='section-head'>🧠 Model Overview</div>", unsafe_allow_html=True)
    st.markdown("""<p class='section-desc'>
        We use a <strong>Random Forest Regressor</strong> with 200 trees and max depth of 10.
        The model was trained on 80% of the data and evaluated on the remaining 20%.
        Below are the key performance metrics and visualizations showing how well the model generalizes.
    </p>""", unsafe_allow_html=True)

    # Metrics cards
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_items = [
        ("MAE", f"{metrics['mae']:.3f}", "Mean Abs Error"),
        ("RMSE", f"{metrics['rmse']:.3f}", "Root Mean Sq Err"),
        ("R² Score", f"{metrics['r2']:.3f}", "Variance Explained"),
        ("Adj. R²", f"{metrics['adj_r2']:.3f}", "Adjusted"),
        ("Test Size", f"{metrics['n_test']}", "Samples"),
    ]
    for col, (label, val, sub) in zip([m1, m2, m3, m4, m5], metric_items):
        with col:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-label'>{label}</div>
                <div class='stat-value'>{val}</div>
                <div class='stat-label'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Metric explanations
    st.markdown("<div class='section-head'>📖 What Do These Metrics Mean?</div>", unsafe_allow_html=True)
    me1, me2 = st.columns(2)
    with me1:
        st.markdown(f"""<div class='info-card'>
            <div class='info-card-title'>📏 MAE (Mean Absolute Error) = {metrics['mae']:.3f}</div>
            <div class='info-card-text'>
                On average, the model's predictions are off by about <strong>{metrics['mae']:.2f} points</strong> on a 0–20 scale.
                This means if the model predicts a 14, the actual grade is typically between {14 - metrics['mae']:.1f} and {14 + metrics['mae']:.1f}.
                Lower is better — and this is quite good for student performance prediction.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class='info-card'>
            <div class='info-card-title'>📐 RMSE (Root Mean Squared Error) = {metrics['rmse']:.3f}</div>
            <div class='info-card-text'>
                Similar to MAE but penalizes larger errors more heavily. An RMSE of {metrics['rmse']:.2f} means the model 
                doesn't make many big mistakes. The closer RMSE is to MAE, the fewer outlier predictions the model makes.
                The gap here is {metrics['rmse'] - metrics['mae']:.2f}, which is small — indicating consistent predictions.
            </div>
        </div>""", unsafe_allow_html=True)

    with me2:
        st.markdown(f"""<div class='info-card'>
            <div class='info-card-title'>📊 R² Score = {metrics['r2']:.3f}</div>
            <div class='info-card-text'>
                The model explains <strong>{metrics['r2'] * 100:.1f}%</strong> of the variance in final grades.
                An R² of 1.0 would mean perfect prediction. Our score of {metrics['r2']:.3f} is {'excellent' if metrics['r2'] > 0.85 else ('very good' if metrics['r2'] > 0.7 else 'decent')} — 
                especially considering we're only using 6 features out of 33 available in the dataset.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class='info-card'>
            <div class='info-card-title'>🔧 Adjusted R² = {metrics['adj_r2']:.3f}</div>
            <div class='info-card-text'>
                This is R² adjusted for the number of features used. It penalizes adding useless features. 
                The fact that Adj. R² ({metrics['adj_r2']:.3f}) is very close to R² ({metrics['r2']:.3f}) confirms that 
                all 6 features are contributing meaningfully — none are redundant or hurting the model.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Charts
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown("<div class='section-head'>🎯 Predicted vs Actual</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>Each dot represents a test student. Points close to the diagonal line indicate accurate predictions. The tighter the cluster around the line, the better.</p>", unsafe_allow_html=True)
        y_test = metrics['y_test']
        y_pred = metrics['y_pred']

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(color='#a855f7', opacity=0.5, size=7),
            name='Predictions'
        ))
        fig_pred.add_trace(go.Scatter(
            x=[0, 20], y=[0, 20], mode='lines',
            line=dict(color='#ec4899', dash='dash', width=2),
            name='Perfect prediction'
        ))
        fig_pred.update_layout(**plotly_layout, height=400, xaxis_title='Actual G3', yaxis_title='Predicted G3',
                               legend=dict(font=dict(color='#9d8ec7')))
        fig_pred.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig_pred.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig_pred, use_container_width=True)

    with pc2:
        st.markdown("<div class='section-head'>📉 Prediction Error Distribution</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>The residuals (prediction errors) show how much the model over- or under-predicts. A bell curve centered at 0 means the model isn't systematically biased in either direction.</p>", unsafe_allow_html=True)
        errors = [p - a for p, a in zip(y_pred, y_test)]

        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=errors, nbinsx=30,
            marker=dict(color='#ec4899', line=dict(color='#f472b6', width=1)),
            opacity=0.8,
        ))
        fig_err.add_vline(x=0, line_dash="dash", line_color="#a855f7", line_width=2)
        fig_err.update_layout(**plotly_layout, height=400, xaxis_title='Prediction Error (Predicted - Actual)',
                              yaxis_title='Count')
        fig_err.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig_err.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig_err, use_container_width=True)

    # Model config table
    st.markdown("<div class='section-head'>⚙️ Model Configuration</div>", unsafe_allow_html=True)
    st.markdown("""
    <table class='styled-table'>
        <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        <tr><td>Algorithm</td><td><strong>Random Forest Regressor</strong></td><td>Ensemble of decision trees that averages predictions for robust results</td></tr>
        <tr><td>n_estimators</td><td>200</td><td>Number of decision trees in the forest</td></tr>
        <tr><td>max_depth</td><td>10</td><td>Maximum depth of each tree — prevents overfitting</td></tr>
        <tr><td>random_state</td><td>42</td><td>Seed for reproducibility</td></tr>
        <tr><td>Scaler</td><td>StandardScaler</td><td>Features normalized to zero mean and unit variance</td></tr>
        <tr><td>Train/Test Split</td><td>80% / 20%</td><td>""" + f"{metrics['n_train']} training samples, {metrics['n_test']} test samples" + """</td></tr>
        <tr><td>Features Used</td><td>6</td><td>G1, G2, absences, studytime, failures, higher_yes</td></tr>
    </table>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: ABOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("<div class='section-head'>📖 About This Project</div>", unsafe_allow_html=True)

    st.markdown("""<div class='info-card'>
        <div class='info-card-title'>🎯 Project Goal</div>
        <div class='info-card-text'>
            The aim of this project is to predict a student's <strong>final grade (G3)</strong> in secondary school
            using machine learning. By analyzing patterns in academic performance, study habits, and background factors, 
            the model provides educators and students with early insights into expected outcomes — enabling timely 
            intervention for at-risk students.
        </div>
    </div>""", unsafe_allow_html=True)

    # Team credits
    st.markdown("<div class='section-head'>👩‍💻 Project Team</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    team = [
        ("Iqra Ansari", "03", "🎓"),
        ("Shiwani Pandey", "31", "🎓"),
        ("Sonal Sharma", "37", "🎓"),
    ]
    for col, (name, roll, icon) in zip([t1, t2, t3], team):
        with col:
            st.markdown(f"""<div class='stat-box'>
                <div style='font-size: 2rem; margin-bottom: 5px;'>{icon}</div>
                <div class='stat-value' style='font-size: 1.3rem;'>{name}</div>
                <div class='stat-label'>Roll No. {roll}</div>
                <div class='stat-label' style='margin-top: 4px;'>SY BSc Data Science</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class='info-card' style='margin-top: 15px;'>
        <div class='info-card-title'>📚 Course Details</div>
        <div class='info-card-text'>
            <strong>Subject:</strong> Machine Learning<br>
            <strong>Program:</strong> SY BSc Data Science<br>
            <strong>Project:</strong> Student Final Grade Prediction using Random Forest Regression
        </div>
    </div>""", unsafe_allow_html=True)

    ab1, ab2 = st.columns(2)

    with ab1:
        st.markdown("""<div class='info-card'>
            <div class='info-card-title'>📊 Dataset Source</div>
            <div class='info-card-text'>
                <strong>UCI Machine Learning Repository — Student Performance Dataset</strong><br><br>
                Collected from two Portuguese secondary schools (Gabriel Pereira and Mousinho da Silveira), 
                this dataset contains records from students in Math and Portuguese language courses. 
                Data was gathered via school reports and student questionnaires during the 2005-2006 academic year.<br><br>
                <strong>Citation:</strong> P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." 
                In A. Brito and J. Teixeira (Eds.), EUROSIS, 2008.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-card'>
            <div class='info-card-title'>🔬 Methodology</div>
            <div class='info-card-text'>
                <strong>1. Data Collection & Merging:</strong> Combined Math (395 students) and Portuguese (649 students) datasets into a single dataset of 1,044 records.<br><br>
                <strong>2. Feature Selection:</strong> Selected 6 most impactful features from the original 33 to ensure clean deployment without train-serve skew.<br><br>
                <strong>3. Preprocessing:</strong> Applied StandardScaler normalization to ensure all features contribute equally regardless of their natural scale.<br><br>
                <strong>4. Model Training:</strong> Trained a Random Forest Regressor (200 trees, max depth 10) on 80% of the data.<br><br>
                <strong>5. Evaluation:</strong> Validated on the remaining 20% using MAE, RMSE, R², and Adjusted R² metrics.<br><br>
                <strong>6. Deployment:</strong> Deployed as an interactive web application using Streamlit.
            </div>
        </div>""", unsafe_allow_html=True)

    with ab2:
        st.markdown("""<div class='info-card'>
            <div class='info-card-title'>🤖 Why Random Forest?</div>
            <div class='info-card-text'>
                Random Forest was chosen over Linear Regression for several reasons:<br><br>
                <strong>• Non-linear relationships:</strong> Student performance is influenced by complex, non-linear interactions between features. A student with high G2 but many absences behaves differently than linear models assume.<br><br>
                <strong>• Robustness:</strong> By averaging predictions from 200 decision trees, Random Forest is resistant to noise and outliers in the data.<br><br>
                <strong>• Feature importance:</strong> The model naturally provides feature importance scores, helping us understand which factors matter most.<br><br>
                <strong>• No overfitting:</strong> With max_depth=10 and enough trees, the model generalizes well to unseen students without memorizing the training data.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-card'>
            <div class='info-card-title'>🛠️ Tech Stack</div>
            <div class='info-card-text'>
                <strong>• Python</strong> — Core programming language<br>
                <strong>• Scikit-learn</strong> — Model training, preprocessing, evaluation<br>
                <strong>• Pandas & NumPy</strong> — Data manipulation and processing<br>
                <strong>• Plotly</strong> — Interactive charts and visualizations<br>
                <strong>• Streamlit</strong> — Web application framework and deployment<br>
                <strong>• Matplotlib</strong> — Additional plotting in the notebook<br>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-card'>
            <div class='info-card-title'>⚠️ Limitations</div>
            <div class='info-card-text'>
                <strong>• Data age:</strong> The dataset is from 2005-2006 Portuguese schools. Student behavior and grading standards may differ today or in other countries.<br><br>
                <strong>• Feature subset:</strong> We use 6 of 33 available features for deployment simplicity. Including more features (family education, health, social factors) could improve accuracy.<br><br>
                <strong>• G2 dominance:</strong> G2 heavily influences predictions. Without G2, the model would be less accurate but might reveal more about behavioral factors.
            </div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 40px 0 20px 0; color: #4a4270; font-size: 0.8rem;'>
    Built with Streamlit & Scikit-learn · Random Forest Regressor · UCI Student Performance Dataset
</div>
""", unsafe_allow_html=True)
