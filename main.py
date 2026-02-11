import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Dark & Modern Theme ───────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid rgba(108, 92, 231, 0.3);
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.15), rgba(72, 52, 212, 0.08));
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    .metric-label {
        color: #a0a0b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Grade display */
    .grade-display {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.2), rgba(72, 52, 212, 0.1));
        border: 2px solid rgba(108, 92, 231, 0.4);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    .grade-number {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe, #fd79a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .grade-label {
        color: #a0a0b8;
        font-size: 1.1rem;
        margin-top: 8px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Status badges */
    .status-excellent {
        background: linear-gradient(135deg, rgba(0, 206, 158, 0.2), rgba(0, 206, 158, 0.05));
        border: 1px solid rgba(0, 206, 158, 0.4);
        color: #00ce9e;
        padding: 12px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
    }
    .status-good {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.2), rgba(108, 92, 231, 0.05));
        border: 1px solid rgba(108, 92, 231, 0.4);
        color: #a29bfe;
        padding: 12px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
    }
    .status-risk {
        background: linear-gradient(135deg, rgba(253, 121, 168, 0.2), rgba(253, 121, 168, 0.05));
        border: 1px solid rgba(253, 121, 168, 0.4);
        color: #fd79a8;
        padding: 12px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Insight cards */
    .insight-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .insight-title {
        color: #a29bfe;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .insight-text {
        color: #d0d0e0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Section headers */
    .section-header {
        color: #e0e0f0;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(108, 92, 231, 0.3);
    }

    /* Sidebar styling */
    .sidebar-header {
        color: #a29bfe;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 20px 0 10px 0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #6c5ce7;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6c5ce7, #4834d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 40px;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c6cf7, #5844e4);
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.4);
        transform: translateY(-1px);
    }

    /* Radio buttons */
    .stRadio > div {
        display: flex;
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Train model if pkl files don't exist ─────────────────────────────────────
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

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)

    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(FEATURES, f)

if not all(os.path.exists(f) for f in ['random_forest_model.pkl', 'scaler.pkl', 'feature_columns.pkl']):
    with st.spinner('🔧 Training model for the first time...'):
        train_and_save()

# ── Load model artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Grade Predictor")
    st.markdown("<p style='color: #a0a0b8; font-size: 0.85rem;'>Predict a student's final grade using machine learning</p>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div class='sidebar-header'>📊 Academic Scores</div>", unsafe_allow_html=True)
    G1 = st.slider('First Period Grade (G1)', 0, 20, 10)
    G2 = st.slider('Second Period Grade (G2)', 0, 20, 10)

    st.markdown("<div class='sidebar-header'>📚 Study Habits</div>", unsafe_allow_html=True)
    studytime = st.slider('Weekly Study Time', 1, 4, 2, help="1 = <2hrs, 2 = 2-5hrs, 3 = 5-10hrs, 4 = >10hrs")
    absences = st.number_input('School Absences', 0, 93, 5)

    st.markdown("<div class='sidebar-header'>📋 Background</div>", unsafe_allow_html=True)
    failures = st.slider('Past Class Failures', 0, 4, 0)
    higher_val = st.radio('Wants Higher Education?', ['Yes', 'No'], index=0)

    st.markdown("---")

    predict_btn = st.button('🔮 Predict Grade', use_container_width=True)

# ── Build input DataFrame ────────────────────────────────────────────────────
def create_input_df(G1, G2, absences, studytime, failures, higher_val, feature_columns):
    input_data = {
        'G1': G1,
        'G2': G2,
        'absences': absences,
        'studytime': studytime,
        'failures': failures,
        'higher_yes': 1 if higher_val == 'Yes' else 0,
    }
    return pd.DataFrame([input_data])[feature_columns]

input_df = create_input_df(G1, G2, absences, studytime, failures, higher_val, feature_columns)

# ── Main Content Area ────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='text-align: center; padding: 20px 0 10px 0;'>
    <h1 style='font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #6c5ce7, #a29bfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px;'>
        Student Grade Predictor
    </h1>
    <p style='color: #a0a0b8; font-size: 1.05rem;'>AI-powered final grade prediction using Random Forest</p>
</div>
""", unsafe_allow_html=True)

# Input summary cards
st.markdown("<div class='section-header'>📋 Current Input Summary</div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>G1</div>
        <div class='metric-value'>{G1}</div>
        <div class='metric-label'>Period 1</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>G2</div>
        <div class='metric-value'>{G2}</div>
        <div class='metric-label'>Period 2</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Study</div>
        <div class='metric-value'>{studytime}</div>
        <div class='metric-label'>Weekly</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Absences</div>
        <div class='metric-value'>{absences}</div>
        <div class='metric-label'>Total</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Failures</div>
        <div class='metric-value'>{failures}</div>
        <div class='metric-label'>Past</div>
    </div>""", unsafe_allow_html=True)
with c6:
    higher_icon = "✓" if higher_val == "Yes" else "✗"
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Higher Ed</div>
        <div class='metric-value'>{higher_icon}</div>
        <div class='metric-label'>{'Yes' if higher_val == 'Yes' else 'No'}</div>
    </div>""", unsafe_allow_html=True)

# ── Prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_grade = float(np.clip(prediction[0], 0, 20))

    st.markdown("---")

    # Grade display
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(f"""<div class='grade-display'>
            <div class='grade-label'>Predicted Final Grade</div>
            <div class='grade-number'>{predicted_grade:.1f}</div>
            <div class='grade-label'>out of 20</div>
        </div>""", unsafe_allow_html=True)

        # Status badge
        if predicted_grade >= 16:
            st.markdown("<div class='status-excellent'>🌟 Excellent — Top performer</div>", unsafe_allow_html=True)
        elif predicted_grade >= 14:
            st.markdown("<div class='status-excellent'>✅ Very Good — Strong performance expected</div>", unsafe_allow_html=True)
        elif predicted_grade >= 10:
            st.markdown("<div class='status-good'>📘 Satisfactory — Likely to pass</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-risk'>⚠️ At Risk — May need additional support</div>", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts & Insights ────────────────────────────────────────────────────
    chart_col, insight_col = st.columns([1.2, 1])

    with chart_col:
        st.markdown("<div class='section-header'>📊 Grade Comparison</div>", unsafe_allow_html=True)

        # Bar chart: G1 vs G2 vs Predicted G3
        chart_data = pd.DataFrame({
            'Period': ['G1 (Period 1)', 'G2 (Period 2)', 'G3 (Predicted)'],
            'Grade': [G1, G2, predicted_grade]
        })
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = ['#6c5ce7', '#a29bfe', '#fd79a8']
        fig.add_trace(go.Bar(
            x=chart_data['Period'],
            y=chart_data['Grade'],
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}" for v in chart_data['Grade']],
            textposition='outside',
            textfont=dict(color='#e0e0f0', size=14, family='Arial Black'),
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0a0b8'),
            yaxis=dict(
                range=[0, 22],
                gridcolor='rgba(108, 92, 231, 0.1)',
                title='Grade',
            ),
            xaxis=dict(title=''),
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance chart
        st.markdown("<div class='section-header'>🧠 What Matters Most</div>", unsafe_allow_html=True)
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': ['G1', 'G2', 'Absences', 'Study Time', 'Failures', 'Higher Ed'],
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=feat_df['Importance'],
            y=feat_df['Feature'],
            orientation='h',
            marker=dict(
                color=feat_df['Importance'],
                colorscale=[[0, '#4834d4'], [0.5, '#6c5ce7'], [1, '#a29bfe']],
                line=dict(width=0),
            ),
            text=[f"{v:.1%}" for v in feat_df['Importance']],
            textposition='outside',
            textfont=dict(color='#e0e0f0', size=12),
        ))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0a0b8'),
            xaxis=dict(
                gridcolor='rgba(108, 92, 231, 0.1)',
                title='Importance',
                range=[0, max(importances) * 1.3],
            ),
            yaxis=dict(title=''),
            height=300,
            margin=dict(l=20, r=60, t=10, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with insight_col:
        st.markdown("<div class='section-header'>💡 Grade Breakdown & Insights</div>", unsafe_allow_html=True)

        # Grade trend
        trend = predicted_grade - G2
        trend_icon = "📈" if trend > 0.5 else ("📉" if trend < -0.5 else "➡️")
        trend_text = f"up {trend:.1f} pts from G2" if trend > 0.5 else (f"down {abs(trend):.1f} pts from G2" if trend < -0.5 else "stable from G2")

        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>{trend_icon} Grade Trend</div>
            <div class='insight-text'>Predicted G3 is <strong>{trend_text}</strong>. 
            G1→G2 change was <strong>{'+' if G2-G1 >= 0 else ''}{G2-G1} pts</strong>.</div>
        </div>""", unsafe_allow_html=True)

        # Absences impact
        abs_icon = "✅" if absences <= 5 else ("⚠️" if absences <= 15 else "🚨")
        abs_text = "Low absences — good attendance!" if absences <= 5 else ("Moderate absences — could affect performance." if absences <= 15 else "High absences — this is likely dragging the grade down significantly.")

        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>{abs_icon} Attendance</div>
            <div class='insight-text'>{absences} absences recorded. {abs_text}</div>
        </div>""", unsafe_allow_html=True)

        # Study time
        study_labels = {1: "< 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "> 10 hours"}
        study_icon = "📚" if studytime >= 3 else ("📖" if studytime == 2 else "⏰")
        study_tip = "Excellent study commitment!" if studytime >= 3 else ("Average study time. Increasing could help." if studytime == 2 else "Minimal study time — this is a key area for improvement.")

        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>{study_icon} Study Habits</div>
            <div class='insight-text'>Studying {study_labels[studytime]}/week. {study_tip}</div>
        </div>""", unsafe_allow_html=True)

        # Failures
        if failures == 0:
            fail_text = "No past failures — clean academic record."
            fail_icon = "🏆"
        elif failures <= 2:
            fail_text = f"{failures} past failure(s). Previous struggles may impact confidence."
            fail_icon = "📋"
        else:
            fail_text = f"{failures} past failures. Significant academic challenges detected."
            fail_icon = "🔴"

        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>{fail_icon} Academic History</div>
            <div class='insight-text'>{fail_text}</div>
        </div>""", unsafe_allow_html=True)

        # Higher education
        higher_text = "Motivation for higher education is a positive indicator for academic effort." if higher_val == "Yes" else "Not pursuing higher education may reduce academic motivation."
        higher_icon = "🎯" if higher_val == "Yes" else "💭"

        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>{higher_icon} Future Goals</div>
            <div class='insight-text'>{higher_text}</div>
        </div>""", unsafe_allow_html=True)

        # Percentage & letter grade
        percentage = (predicted_grade / 20) * 100
        if predicted_grade >= 16:
            letter = "A"
        elif predicted_grade >= 14:
            letter = "B"
        elif predicted_grade >= 12:
            letter = "C"
        elif predicted_grade >= 10:
            letter = "D"
        else:
            letter = "F"

        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>📊 Grade Summary</div>
            <div class='insight-text'>
                Score: <strong>{predicted_grade:.1f}/20</strong> ({percentage:.0f}%)<br>
                Equivalent Letter Grade: <strong>{letter}</strong>
            </div>
        </div>""", unsafe_allow_html=True)

else:
    # Default state before prediction
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px; color: #a0a0b8;'>
        <p style='font-size: 3rem; margin-bottom: 10px;'>🔮</p>
        <p style='font-size: 1.2rem; font-weight: 600;'>Adjust parameters in the sidebar</p>
        <p style='font-size: 0.95rem;'>Then click <strong style='color: #a29bfe;'>Predict Grade</strong> to see results</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 40px 0 20px 0; color: #505070; font-size: 0.8rem;'>
    Built with Streamlit & Scikit-learn · Random Forest Regressor · UCI Student Performance Dataset
</div>
""", unsafe_allow_html=True)
