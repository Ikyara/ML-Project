import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import time
import base64
from datetime import datetime

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stApp"], [data-testid="stBottom"], .main, .block-container,
    [data-testid="stMainBlockContainer"], .appview-container,
    header[data-testid="stHeader"] { background-color: #08060f !important; }
    html { overscroll-behavior: none; background: #08060f !important; }
    [data-testid="stDecoration"] { display: none !important; }
    header[data-testid="stHeader"] { background: #08060f !important; }
    [data-testid="stSidebar"] { display: none !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: #08060f; font-family: 'Inter', sans-serif; color: #c4b5d4; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap:0; background: rgba(255,255,255,0.02); border-radius:10px; padding:3px;
        border:1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px; padding:9px 20px; color:#6b5f85; font-weight:600; font-size:0.85rem;
        letter-spacing: 0.3px; text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6, #d946ef) !important; color:white !important;
    }

    /* Glass card */
    .glass {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    .glass-accent {
        background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(217,70,239,0.04));
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 14px;
        padding: 20px;
    }

    /* Grade display */
    .grade-container {
        background: linear-gradient(135deg, rgba(139,92,246,0.1), rgba(217,70,239,0.05));
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 20px;
        padding: 40px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .grade-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(139,92,246,0.06), transparent 60%);
        pointer-events: none;
    }
    .grade-number {
        font-size: 5rem; font-weight: 900; line-height: 1;
        background: linear-gradient(135deg, #a78bfa, #e879f9, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        position: relative;
    }
    .grade-label {
        color: #6b5f85; font-size: 0.85rem; letter-spacing: 2px;
        text-transform: uppercase; margin-top: 8px; font-weight: 500;
    }

    .grade-sm {
        background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(217,70,239,0.04));
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 16px; padding: 28px 16px; text-align: center;
    }
    .grade-num-sm {
        font-size: 3.2rem; font-weight: 900; line-height: 1;
        background: linear-gradient(135deg, #a78bfa, #e879f9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* Status pills */
    .pill-good { background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.2); color:#6ee7b7; padding:10px 20px; border-radius:10px; text-align:center; font-weight:600; margin-top:12px; font-size:0.85rem; letter-spacing:0.3px; }
    .pill-ok { background:rgba(139,92,246,0.08); border:1px solid rgba(139,92,246,0.2); color:#a78bfa; padding:10px 20px; border-radius:10px; text-align:center; font-weight:600; margin-top:12px; font-size:0.85rem; letter-spacing:0.3px; }
    .pill-risk { background:rgba(244,114,182,0.08); border:1px solid rgba(244,114,182,0.2); color:#f472b6; padding:10px 20px; border-radius:10px; text-align:center; font-weight:600; margin-top:12px; font-size:0.85rem; letter-spacing:0.3px; }

    /* Info cards */
    .icard {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 18px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }
    .icard:hover { border-color: rgba(139,92,246,0.2); }
    .icard-t { color:#a78bfa; font-weight:700; font-size:0.85rem; margin-bottom:6px; letter-spacing:0.3px; }
    .icard-b { color:#8b7fa8; font-size:0.82rem; line-height:1.6; }

    /* Section */
    .sh { color:#ddd0f0; font-size:1.1rem; font-weight:700; margin:22px 0 10px 0; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.05); letter-spacing:0.3px; }
    .sd { color:#6b5f85; font-size:0.82rem; line-height:1.5; margin-bottom:14px; }

    /* Stat */
    .sbox {
        background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);
        border-radius:12px; padding:18px; text-align:center;
        transition: border-color 0.2s;
    }
    .sbox:hover { border-color: rgba(139,92,246,0.2); }
    .sv { font-size:1.5rem; font-weight:800; color:#a78bfa; }
    .sl { color:#6b5f85; font-size:0.68rem; text-transform:uppercase; letter-spacing:1.2px; font-weight:600; }

    /* Table */
    .tb { width:100%; border-collapse:collapse; }
    .tb th { background:rgba(139,92,246,0.06); color:#a78bfa; padding:10px 14px; text-align:left; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid rgba(255,255,255,0.06); }
    .tb td { padding:10px 14px; color:#8b7fa8; font-size:0.82rem; border-bottom:1px solid rgba(255,255,255,0.03); }

    /* Button */
    .stButton > button {
        background:linear-gradient(135deg,#8b5cf6,#d946ef); color:white; border:none;
        border-radius:10px; padding:12px 35px; font-size:0.95rem; font-weight:700; width:100%;
        letter-spacing:0.5px; text-transform:uppercase; transition: all 0.3s;
    }
    .stButton > button:hover { box-shadow:0 0 30px rgba(139,92,246,0.25); transform:translateY(-1px); }

    /* VS badge */
    .vs-badge {
        background:linear-gradient(135deg,#8b5cf6,#d946ef); color:white;
        font-weight:900; font-size:1rem; padding:8px 14px; border-radius:50%;
        display:inline-block; letter-spacing:1px;
    }

    /* Download */
    .dl-btn {
        display:inline-block; background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.2);
        color:#a78bfa !important; padding:10px 24px; border-radius:10px; text-decoration:none;
        font-weight:600; font-size:0.85rem; margin-top:12px; text-align:center; letter-spacing:0.3px;
        transition: all 0.2s;
    }
    .dl-btn:hover { background:rgba(139,92,246,0.15); border-color:rgba(139,92,246,0.35); color:#c4b5fd !important; }

    /* Where you stand marker */
    .marker-line {
        width:3px; height:100%; background:linear-gradient(to bottom,#d946ef,#8b5cf6);
        position:absolute; border-radius:2px;
    }

    /* Slider label color */
    .stSlider label, .stNumberInput label, .stRadio label { color: #9b8fbf !important; font-weight: 500 !important; }

    /* Progress bar */
    .stProgress > div > div > div { background: linear-gradient(90deg, #8b5cf6, #d946ef) !important; }
</style>
""", unsafe_allow_html=True)

# ── Train ────────────────────────────────────────────────────────────────────
def train_and_save():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    por = pd.read_csv('student-por.csv', sep=';')
    mat = pd.read_csv('student-mat.csv', sep=';')
    df = pd.concat([por, mat], ignore_index=True)
    df['higher_yes'] = (df['higher'] == 'yes').astype(int)
    FEATURES = ['G1', 'G2', 'absences', 'studytime', 'failures', 'higher_yes']
    X, y = df[FEATURES].copy(), df['G3'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train); Xte = sc.transform(X_test)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    rf.fit(Xtr, y_train)
    preds = rf.predict(Xte)
    m = {'mae': mean_absolute_error(y_test, preds), 'rmse': np.sqrt(mean_squared_error(y_test, preds)),
         'r2': r2_score(y_test, preds), 'n_train': len(X_train), 'n_test': len(X_test),
         'y_test': y_test.values.tolist(), 'y_pred': preds.tolist()}
    n, p = X_test.shape
    m['adj_r2'] = 1 - (1 - m['r2']) * (n - 1) / (n - p - 1)
    for name, obj in [('random_forest_model.pkl', rf), ('scaler.pkl', sc), ('feature_columns.pkl', FEATURES), ('metrics.pkl', m)]:
        with open(name, 'wb') as f: pickle.dump(obj, f)

if not all(os.path.exists(f) for f in ['random_forest_model.pkl', 'scaler.pkl', 'feature_columns.pkl', 'metrics.pkl']):
    with st.spinner('Initializing model...'): train_and_save()

@st.cache_resource
def load_all():
    objs = []
    for name in ['random_forest_model.pkl', 'scaler.pkl', 'feature_columns.pkl', 'metrics.pkl']:
        with open(name, 'rb') as f: objs.append(pickle.load(f))
    return objs
model, scaler, feature_columns, metrics = load_all()

@st.cache_data
def load_data():
    por = pd.read_csv('student-por.csv', sep=';')
    mat = pd.read_csv('student-mat.csv', sep=';')
    por['subject'], mat['subject'] = 'Portuguese', 'Math'
    return pd.concat([por, mat], ignore_index=True)
df_raw = load_data()

PL = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
          font=dict(color='#6b5f85', family='Inter', size=12), margin=dict(l=40, r=30, t=35, b=40))
GRID = 'rgba(139,92,246,0.06)'

def predict_grade(g1, g2, abs_, study, fail, higher):
    inp = pd.DataFrame([{'G1':g1,'G2':g2,'absences':abs_,'studytime':study,'failures':fail,
                          'higher_yes':1 if higher=='Yes' else 0}])[feature_columns]
    return float(np.clip(model.predict(scaler.transform(inp))[0], 0, 20))

def get_letter(g):
    return "A" if g>=16 else ("B" if g>=14 else ("C" if g>=12 else ("D" if g>=10 else "F")))

def status_html(pred):
    if pred >= 14: return "<div class='pill-good'>Strong performance predicted</div>"
    elif pred >= 10: return "<div class='pill-ok'>On track to pass</div>"
    return "<div class='pill-risk'>At risk — may need support</div>"

def generate_report(g1, g2, abs_, study, fail, higher, pred, letter, pct):
    sl = {1:"<2 hours",2:"2-5 hours",3:"5-10 hours",4:">10 hours"}
    return f"""<html><head><style>
        body{{font-family:Inter,Arial,sans-serif;padding:40px;color:#1a1a2e;background:#faf8ff;}}
        h1{{color:#7c3aed;border-bottom:3px solid #d946ef;padding-bottom:12px;font-size:24px;}}
        h2{{color:#8b5cf6;margin-top:28px;font-size:18px;}}
        .grade{{font-size:64px;font-weight:900;color:#7c3aed;text-align:center;margin:20px 0;}}
        .grade-info{{text-align:center;color:#666;font-size:16px;}}
        table{{width:100%;border-collapse:collapse;margin:16px 0;}}
        th{{background:#f3e8ff;color:#7c3aed;padding:10px 14px;text-align:left;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;}}
        td{{padding:10px 14px;border-bottom:1px solid #f0e8ff;font-size:14px;}}
        .pill{{text-align:center;padding:12px;border-radius:8px;font-weight:600;font-size:14px;margin:12px 0;}}
        .pill-g{{background:#dcfce7;color:#16a34a;}}.pill-o{{background:#f3e8ff;color:#7c3aed;}}.pill-r{{background:#fce7f3;color:#db2777;}}
        .footer{{margin-top:40px;padding-top:16px;border-top:1px solid #e8e0f0;color:#999;font-size:11px;text-align:center;}}
    </style></head><body>
        <h1>Student Grade Prediction Report</h1>
        <p style="color:#888;font-size:13px;">Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        <h2>Predicted Final Grade</h2>
        <div class="grade">{pred:.1f}</div>
        <div class="grade-info">out of 20 &middot; {pct:.0f}% &middot; Grade {letter}</div>
        <div class="pill {'pill-g' if pred>=14 else ('pill-o' if pred>=10 else 'pill-r')}">
            {'Strong performance predicted' if pred>=14 else ('On track to pass' if pred>=10 else 'At risk — may need support')}</div>
        <h2>Input Parameters</h2>
        <table><tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        <tr><td>G1</td><td>{g1}</td><td>First period grade</td></tr>
        <tr><td>G2</td><td>{g2}</td><td>Second period grade</td></tr>
        <tr><td>Absences</td><td>{abs_}</td><td>School absences</td></tr>
        <tr><td>Study Time</td><td>{study} ({sl[study]})</td><td>Weekly study hours</td></tr>
        <tr><td>Failures</td><td>{fail}</td><td>Past class failures</td></tr>
        <tr><td>Higher Ed</td><td>{higher}</td><td>Pursing higher education</td></tr></table>
        <h2>Analysis</h2>
        <table><tr><th>Aspect</th><th>Assessment</th></tr>
        <tr><td>Grade Trend</td><td>G1 to G2: {'+' if g2-g1>=0 else ''}{g2-g1} &middot; G2 to G3: {'+' if pred-g2>=0 else ''}{pred-g2:.1f}</td></tr>
        <tr><td>Attendance</td><td>{abs_} absences — {'Good' if abs_<=5 else ('Average' if abs_<=15 else 'Poor')}</td></tr>
        <tr><td>Study Habits</td><td>{sl[study]}/week — {'Strong' if study>=3 else ('Moderate' if study==2 else 'Needs improvement')}</td></tr>
        <tr><td>History</td><td>{fail} past failures</td></tr></table>
        <h2>Model</h2>
        <p style="font-size:13px;color:#666;">Random Forest Regressor (200 trees, depth 10) · UCI Student Performance Dataset · 1,044 students</p>
        <div class="footer">Iqra Ansari (03) &middot; Shiwani Pandey (31) &middot; Sonal Sharma (37) &middot; SY BSc Data Science</div>
    </body></html>"""

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 8px 0;'>
    <div style='font-size:2rem; font-weight:900; background:linear-gradient(135deg,#a78bfa,#e879f9);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.5px;'>
        Student Grade Predictor
    </div>
    <div style='color:#6b5f85; font-size:0.82rem; margin-top:4px; letter-spacing:0.5px;'>
        Random Forest Regression &middot; UCI Student Performance Dataset
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["PREDICT", "COMPARE", "DATASET", "MODEL", "ABOUT"])

# ━━━━━━━━━━━ TAB 1: PREDICT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("<div class='sh'>Input Parameters</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: G1 = st.slider('G1 — Period 1', 0, 20, 10, help="First period grade (0–20)")
    with c2: G2 = st.slider('G2 — Period 2', 0, 20, 10, help="Second period grade — strongest predictor")
    with c3: studytime = st.slider('Study Time', 1, 4, 2, help="1: <2h · 2: 2-5h · 3: 5-10h · 4: >10h weekly")
    with c4: absences = st.number_input('Absences', 0, 30, 5, help="School absences (dataset avg: ~6)")
    with c5: failures = st.slider('Past Failures', 0, 4, 0, help="Previous class failures")
    with c6: higher_val = st.radio('Higher Ed?', ['Yes','No'], horizontal=True, help="Pursuing higher education?")

    predict_btn = st.button('PREDICT FINAL GRADE', use_container_width=True)

    if predict_btn:
        with st.spinner(''):
            bar = st.progress(0)
            for i in range(100): time.sleep(0.006); bar.progress(i+1)
            bar.empty()

        pred = predict_grade(G1, G2, absences, studytime, failures, higher_val)
        pct = pred/20*100; letter = get_letter(pred)

        st.markdown("---")
        left, mid, right = st.columns([1, 1.2, 1])

        with left:
            st.markdown(f"""<div class='grade-container'>
                <div class='grade-label'>Predicted Final Grade</div>
                <div class='grade-number'>{pred:.1f}</div>
                <div class='grade-label'>{pct:.0f}% &middot; Grade {letter} &middot; out of 20</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(status_html(pred), unsafe_allow_html=True)

            # Where you stand
            rank = sum(1 for g in df_raw['G3'] if g < pred) / len(df_raw) * 100
            st.markdown(f"""<div class='icard' style='margin-top:12px;'>
                <div class='icard-t'>Percentile Rank</div>
                <div class='icard-b'>This prediction is higher than <strong>{rank:.0f}%</strong> of all 1,044 students in the dataset.</div>
            </div>""", unsafe_allow_html=True)

            report = generate_report(G1, G2, absences, studytime, failures, higher_val, pred, letter, pct)
            b64 = base64.b64encode(report.encode()).decode()
            st.markdown(f'<a class="dl-btn" href="data:text/html;base64,{b64}" download="grade_report.html">Download Report</a>', unsafe_allow_html=True)

        with mid:
            fig = go.Figure(go.Bar(
                x=['G1','G2','G3 (predicted)'], y=[G1, G2, pred],
                marker=dict(color=['#8b5cf6','#a78bfa','#d946ef'], line=dict(width=0)),
                text=[f"{G1}",f"{G2}",f"{pred:.1f}"], textposition='outside',
                textfont=dict(color='#c4b5d4', size=14, family='Inter'),
            ))
            fig.update_layout(**PL, height=320, yaxis=dict(range=[0,22], gridcolor=GRID, title='Grade'),
                              xaxis_title='', title=dict(text='Grade Progression', font=dict(size=14, color='#9b8fbf')))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            imp = model.feature_importances_
            fdf = pd.DataFrame({'F':['G1','G2','Absences','Study Time','Failures','Higher Ed'],'I':imp}).sort_values('I')
            fig2 = go.Figure(go.Bar(
                x=fdf['I'], y=fdf['F'], orientation='h',
                marker=dict(color=fdf['I'], colorscale=[[0,'#d946ef'],[0.5,'#a78bfa'],[1,'#8b5cf6']]),
                text=[f"{v:.0%}" for v in fdf['I']], textposition='outside',
                textfont=dict(color='#c4b5d4', size=11),
            ))
            fig2.update_layout(**PL, height=320, xaxis=dict(gridcolor=GRID, range=[0,max(imp)*1.35], title='Weight'),
                              yaxis_title='', title=dict(text='Feature Importance', font=dict(size=14, color='#9b8fbf')))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='sh'>Analysis</div>", unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        trend = pred - G2
        with i1:
            t = f"improving by {trend:.1f} pts" if trend>0.5 else (f"declining by {abs(trend):.1f} pts" if trend<-0.5 else "holding steady")
            st.markdown(f"""<div class='icard'><div class='icard-t'>Grade Trajectory</div>
                <div class='icard-b'>G3 is {t} from G2. The G1 to G2 shift was {'+' if G2-G1>=0 else ''}{G2-G1} points.</div></div>""", unsafe_allow_html=True)
        with i2:
            avg = df_raw['absences'].mean()
            st.markdown(f"""<div class='icard'><div class='icard-t'>Attendance</div>
                <div class='icard-b'>{absences} absences recorded (dataset avg: {avg:.0f}). {'Below average — consistent attendance.' if absences<=5 else ('Near average. Reducing absences could help.' if absences<=15 else 'Well above average — likely impacting performance.')}</div></div>""", unsafe_allow_html=True)
        with i3:
            sl={1:"under 2 hours",2:"2–5 hours",3:"5–10 hours",4:"over 10 hours"}
            st.markdown(f"""<div class='icard'><div class='icard-t'>Study Habits & Background</div>
                <div class='icard-b'>Studying {sl[studytime]} per week. {failures} past failure(s). {'Pursuing higher education.' if higher_val=='Yes' else 'Not pursuing higher education.'}</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='text-align:center; padding:60px 0; color:#6b5f85;'>
            <div style='font-size:1.1rem; color:#a78bfa; font-weight:600;'>Set parameters above and click Predict</div>
            <div style='font-size:0.82rem; margin-top:6px;'>The model will analyze 6 features to estimate the final grade</div>
        </div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━ TAB 2: COMPARE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("<div class='sh'>Side-by-Side Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='sd'>Enter details for two students and compare their predicted outcomes.</div>", unsafe_allow_html=True)

    col_a, col_vs, col_b = st.columns([5,1,5])
    with col_a:
        st.markdown("<div class='icard-t' style='font-size:0.95rem; text-align:center; margin-bottom:8px;'>Student A</div>", unsafe_allow_html=True)
        a1,a2 = st.columns(2)
        with a1: a_g1=st.slider('G1',0,20,12,key='a_g1'); a_g2=st.slider('G2',0,20,13,key='a_g2'); a_study=st.slider('Study',1,4,3,key='a_st')
        with a2: a_abs=st.number_input('Absences',0,30,2,key='a_ab'); a_fail=st.slider('Failures',0,4,0,key='a_fa'); a_hi=st.radio('Higher Ed?',['Yes','No'],key='a_hi',horizontal=True)
    with col_vs:
        st.markdown("<div style='text-align:center;padding-top:120px;'><div class='vs-badge'>VS</div></div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<div class='icard-t' style='font-size:0.95rem; text-align:center; margin-bottom:8px;'>Student B</div>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        with b1: b_g1=st.slider('G1',0,20,7,key='b_g1'); b_g2=st.slider('G2',0,20,6,key='b_g2'); b_study=st.slider('Study',1,4,1,key='b_st')
        with b2: b_abs=st.number_input('Absences',0,30,18,key='b_ab'); b_fail=st.slider('Failures',0,4,2,key='b_fa'); b_hi=st.radio('Higher Ed?',['Yes','No'],key='b_hi',horizontal=True,index=1)

    cmp_btn = st.button('COMPARE STUDENTS', use_container_width=True)

    if cmp_btn:
        with st.spinner(''):
            bar=st.progress(0)
            for i in range(100): time.sleep(0.005); bar.progress(i+1)
            bar.empty()
        pa = predict_grade(a_g1,a_g2,a_abs,a_study,a_fail,a_hi)
        pb = predict_grade(b_g1,b_g2,b_abs,b_study,b_fail,b_hi)
        la,lb = get_letter(pa),get_letter(pb)

        st.markdown("---")
        r1,r_vs,r2 = st.columns([2,3,2])
        with r1:
            st.markdown(f"""<div class='grade-sm'><div class='grade-label'>Student A</div>
                <div class='grade-num-sm'>{pa:.1f}</div><div class='grade-label'>{pa/20*100:.0f}% · {la}</div></div>""", unsafe_allow_html=True)
            st.markdown(status_html(pa), unsafe_allow_html=True)
        with r_vs:
            fig=go.Figure()
            feats=['G1','G2','Study','Absences','Failures']
            va=[a_g1,a_g2,a_study,a_abs,a_fail]; vb=[b_g1,b_g2,b_study,b_abs,b_fail]
            fig.add_trace(go.Bar(name='Student A',x=feats,y=va,marker_color='#8b5cf6',text=[str(v) for v in va],textposition='outside',textfont=dict(color='#a78bfa',size=11)))
            fig.add_trace(go.Bar(name='Student B',x=feats,y=vb,marker_color='#d946ef',text=[str(v) for v in vb],textposition='outside',textfont=dict(color='#e879f9',size=11)))
            fig.update_layout(**PL,height=300,barmode='group',legend=dict(font=dict(color='#6b5f85'),orientation='h',yanchor='bottom',y=1.02,xanchor='center',x=0.5),yaxis=dict(gridcolor=GRID))
            st.plotly_chart(fig,use_container_width=True)
        with r2:
            st.markdown(f"""<div class='grade-sm'><div class='grade-label'>Student B</div>
                <div class='grade-num-sm'>{pb:.1f}</div><div class='grade-label'>{pb/20*100:.0f}% · {lb}</div></div>""", unsafe_allow_html=True)
            st.markdown(status_html(pb), unsafe_allow_html=True)

        diff=pa-pb; winner="Student A" if diff>0 else ("Student B" if diff<0 else "Tied")
        st.markdown(f"""<div class='glass-accent' style='text-align:center;margin-top:16px;'>
            <div class='icard-t' style='font-size:1rem;'>{winner} {'scores higher' if diff!=0 else ''} by {abs(diff):.1f} points</div>
            <div class='icard-b'>A: <strong>{pa:.1f}/20 ({la})</strong> &middot; B: <strong>{pb:.1f}/20 ({lb})</strong></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sh'>Key Differences</div>", unsafe_allow_html=True)
        d1,d2,d3 = st.columns(3)
        with d1:
            gd=a_g2-b_g2
            st.markdown(f"""<div class='icard'><div class='icard-t'>G2 Gap: {'+' if gd>=0 else ''}{gd}</div>
                <div class='icard-b'>{'Student A has the stronger recent performance.' if gd>0 else ('Student B has the edge.' if gd<0 else 'Equal.')}</div></div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class='icard'><div class='icard-t'>Absences: {a_abs} vs {b_abs}</div>
                <div class='icard-b'>{'A has better attendance.' if a_abs<b_abs else ('B has better attendance.' if b_abs<a_abs else 'Equal.')}</div></div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class='icard'><div class='icard-t'>Effort: {a_study} vs {b_study} study · {a_fail} vs {b_fail} failures</div>
                <div class='icard-b'>{'A shows stronger academic habits.' if a_study>b_study else ('B studies more.' if b_study>a_study else 'Equal study time.')}</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='text-align:center;padding:50px 0;color:#6b5f85;'>
            <div style='font-size:1.1rem;color:#a78bfa;font-weight:600;'>Configure both students and click Compare</div>
        </div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━ TAB 3: DATASET ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("<div class='sh'>Dataset Overview</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sd'>The <strong>UCI Student Performance Dataset</strong> contains 1,044 records from two Portuguese secondary schools
        (2005-2006). 33 features per student — we use 6 for prediction. G3 (final grade, 0–20) is the target.</div>""", unsafe_allow_html=True)

    s1,s2,s3,s4,s5 = st.columns(5)
    for col,(l,v) in zip([s1,s2,s3,s4,s5],[("Students",len(df_raw)),("Features","33"),("Math",len(df_raw[df_raw['subject']=='Math'])),("Portuguese",len(df_raw[df_raw['subject']=='Portuguese'])),("Avg Grade",f"{df_raw['G3'].mean():.1f}")]):
        with col: st.markdown(f"<div class='sbox'><div class='sl'>{l}</div><div class='sv'>{v}</div></div>",unsafe_allow_html=True)

    st.markdown("")
    ch1,ch2 = st.columns(2)
    with ch1:
        st.markdown("<div class='sh'>Grade Distribution</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Spike at 0 represents dropouts. Dashed line marks the passing threshold at 10.</div>",unsafe_allow_html=True)
        fig=go.Figure(go.Histogram(x=df_raw['G3'],nbinsx=21,marker=dict(color='#8b5cf6',line=dict(color='#a78bfa',width=1)),opacity=0.85))
        fig.add_vline(x=10,line_dash="dash",line_color="#d946ef",annotation_text="Pass",annotation_font_color="#d946ef")
        fig.update_layout(**PL,height=320,xaxis_title='G3',yaxis_title='Count'); fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig,use_container_width=True)
    with ch2:
        st.markdown("<div class='sh'>Grade by Subject</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Portuguese students average 11.9 vs Math 10.4. Math shows wider variance.</div>",unsafe_allow_html=True)
        fig=go.Figure()
        for s,c in [('Math','#d946ef'),('Portuguese','#8b5cf6')]:
            fig.add_trace(go.Box(y=df_raw[df_raw['subject']==s]['G3'],name=s,marker_color=c,line_color=c,boxmean=True))
        fig.update_layout(**PL,height=320,yaxis_title='G3',legend=dict(font=dict(color='#6b5f85'))); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig,use_container_width=True)

    ch3,ch4 = st.columns(2)
    with ch3:
        st.markdown("<div class='sh'>G2 vs G3 Correlation</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Strong positive correlation — explains why G2 dominates the model.</div>",unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df_raw['G2'],y=df_raw['G3'],mode='markers',marker=dict(color='#8b5cf6',opacity=0.35,size=5),name='Students'))
        fig.add_trace(go.Scatter(x=[0,20],y=[0,20],mode='lines',line=dict(color='#d946ef',dash='dash',width=2),name='Perfect'))
        fig.update_layout(**PL,height=320,xaxis_title='G2',yaxis_title='G3',legend=dict(font=dict(color='#6b5f85')))
        fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig,use_container_width=True)
    with ch4:
        st.markdown("<div class='sh'>Absences vs Grade</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Weak negative trend. Very high absences are rarely associated with strong grades.</div>",unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(x=df_raw['absences'],y=df_raw['G3'],mode='markers',marker=dict(color='#d946ef',opacity=0.3,size=5)))
        fig.update_layout(**PL,height=320,xaxis_title='Absences',yaxis_title='G3',showlegend=False)
        fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig,use_container_width=True)

    ch5,ch6 = st.columns(2)
    with ch5:
        st.markdown("<div class='sh'>Study Time vs Grade</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Positive trend with significant individual variation at every level.</div>",unsafe_allow_html=True)
        sa=df_raw.groupby('studytime')['G3'].agg(['mean','std']).reset_index()
        sl={1:'<2h',2:'2-5h',3:'5-10h',4:'>10h'}
        fig=go.Figure(go.Bar(x=[sl[s] for s in sa['studytime']],y=sa['mean'],marker_color=['#d946ef','#a78bfa','#8b5cf6','#7c3aed'],
            text=[f"{v:.1f}" for v in sa['mean']],textposition='outside',textfont=dict(color='#c4b5d4',size=13),
            error_y=dict(type='data',array=sa['std'].tolist(),color='#6b5f85',thickness=1.5)))
        fig.update_layout(**PL,height=320,yaxis=dict(range=[0,18],gridcolor=GRID),yaxis_title='Avg G3',xaxis_title='Study Time')
        st.plotly_chart(fig,use_container_width=True)
    with ch6:
        st.markdown("<div class='sh'>Failures vs Grade</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Strong negative pattern. Zero failures: avg 12.1 (n=861). Three failures: avg 6.8 (n=23).</div>",unsafe_allow_html=True)
        fa=df_raw.groupby('failures')['G3'].agg(['mean','count']).reset_index()
        fig=go.Figure(go.Bar(x=[str(f) for f in fa['failures']],y=fa['mean'],
            marker_color=['#34d399','#a78bfa','#d946ef','#f472b6','#ef4444'][:len(fa)],
            text=[f"{v:.1f} (n={c})" for v,c in zip(fa['mean'],fa['count'])],textposition='outside',textfont=dict(color='#c4b5d4',size=12)))
        fig.update_layout(**PL,height=320,yaxis=dict(range=[0,16],gridcolor=GRID),yaxis_title='Avg G3',xaxis_title='Failures')
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("<div class='sh'>Feature Reference</div>",unsafe_allow_html=True)
    st.markdown("""<table class='tb'><tr><th>Feature</th><th>Description</th><th>Range</th><th>Type</th></tr>
        <tr><td>G1</td><td>First period grade</td><td>0–20</td><td>Numeric</td></tr>
        <tr><td>G2</td><td>Second period grade</td><td>0–20</td><td>Numeric</td></tr>
        <tr><td>Absences</td><td>School absences</td><td>0–30</td><td>Numeric</td></tr>
        <tr><td>Study Time</td><td>1: <2h &middot; 2: 2-5h &middot; 3: 5-10h &middot; 4: >10h</td><td>1–4</td><td>Ordinal</td></tr>
        <tr><td>Failures</td><td>Past class failures</td><td>0–4</td><td>Numeric</td></tr>
        <tr><td>Higher Ed</td><td>Pursuing higher education</td><td>Yes/No</td><td>Binary</td></tr></table>""",unsafe_allow_html=True)

# ━━━━━━━━━━━ TAB 4: MODEL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("<div class='sh'>Performance Metrics</div>",unsafe_allow_html=True)
    st.markdown("<div class='sd'>Random Forest Regressor &middot; 200 trees &middot; max depth 10 &middot; 80/20 split</div>",unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    for col,(l,v,s) in zip([m1,m2,m3,m4],[("MAE",f"{metrics['mae']:.3f}","avg error"),("RMSE",f"{metrics['rmse']:.3f}","penalizes outliers"),("R²",f"{metrics['r2']:.3f}","variance explained"),("Adj R²",f"{metrics['adj_r2']:.3f}","adjusted for features")]):
        with col: st.markdown(f"<div class='sbox'><div class='sl'>{l}</div><div class='sv'>{v}</div><div class='sl'>{s}</div></div>",unsafe_allow_html=True)

    st.markdown("")
    e1,e2 = st.columns(2)
    with e1:
        st.markdown(f"""<div class='icard'><div class='icard-t'>Mean Absolute Error: {metrics['mae']:.3f}</div>
            <div class='icard-b'>On average, predictions deviate by {metrics['mae']:.1f} points. A prediction of 14 implies the actual grade falls between {14-metrics['mae']:.1f} and {14+metrics['mae']:.1f}.</div></div>""",unsafe_allow_html=True)
        st.markdown(f"""<div class='icard'><div class='icard-t'>Root Mean Squared Error: {metrics['rmse']:.3f}</div>
            <div class='icard-b'>The RMSE–MAE gap of {metrics['rmse']-metrics['mae']:.2f} indicates consistent predictions without large outlier errors.</div></div>""",unsafe_allow_html=True)
    with e2:
        st.markdown(f"""<div class='icard'><div class='icard-t'>R² Score: {metrics['r2']:.3f}</div>
            <div class='icard-b'>The model explains {metrics['r2']*100:.1f}% of the variance in final grades using only 6 of 33 available features.</div></div>""",unsafe_allow_html=True)
        st.markdown(f"""<div class='icard'><div class='icard-t'>Adjusted R²: {metrics['adj_r2']:.3f}</div>
            <div class='icard-b'>Nearly identical to R², confirming all 6 features contribute meaningful predictive signal.</div></div>""",unsafe_allow_html=True)

    st.markdown("")
    p1,p2 = st.columns(2)
    with p1:
        st.markdown("<div class='sh'>Predicted vs Actual</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Proximity to the diagonal indicates accuracy. The model struggles with dropout cases (G3=0).</div>",unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=metrics['y_test'],y=metrics['y_pred'],mode='markers',marker=dict(color='#8b5cf6',opacity=0.5,size=6),name='Predictions'))
        fig.add_trace(go.Scatter(x=[0,20],y=[0,20],mode='lines',line=dict(color='#d946ef',dash='dash',width=2),name='Perfect'))
        fig.update_layout(**PL,height=350,xaxis_title='Actual',yaxis_title='Predicted',legend=dict(font=dict(color='#6b5f85')))
        fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig,use_container_width=True)
    with p2:
        st.markdown("<div class='sh'>Residual Distribution</div>",unsafe_allow_html=True)
        st.markdown("<div class='sd'>Approximately normal and centered at zero — the model shows no systematic bias.</div>",unsafe_allow_html=True)
        errs=[p-a for p,a in zip(metrics['y_pred'],metrics['y_test'])]
        fig=go.Figure(go.Histogram(x=errs,nbinsx=30,marker=dict(color='#d946ef',line=dict(color='#e879f9',width=1)),opacity=0.8))
        fig.add_vline(x=0,line_dash="dash",line_color="#8b5cf6",line_width=2)
        fig.update_layout(**PL,height=350,xaxis_title='Prediction Error',yaxis_title='Frequency')
        fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("<div class='sh'>Configuration</div>",unsafe_allow_html=True)
    st.markdown(f"""<table class='tb'><tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        <tr><td>Algorithm</td><td>Random Forest Regressor</td><td>Ensemble of decision trees</td></tr>
        <tr><td>n_estimators</td><td>200</td><td>Number of trees in the forest</td></tr>
        <tr><td>max_depth</td><td>10</td><td>Controls tree complexity</td></tr>
        <tr><td>Scaler</td><td>StandardScaler</td><td>Zero mean, unit variance normalization</td></tr>
        <tr><td>Train/Test</td><td>80% / 20%</td><td>{metrics['n_train']} training &middot; {metrics['n_test']} test samples</td></tr>
        <tr><td>Features</td><td>6</td><td>G1, G2, absences, studytime, failures, higher_yes</td></tr></table>""",unsafe_allow_html=True)

# ━━━━━━━━━━━ TAB 5: ABOUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("<div class='sh'>Project Overview</div>",unsafe_allow_html=True)
    st.markdown("""<div class='glass-accent'><div class='icard-b' style='font-size:0.88rem;'>
        This application predicts a student's final grade (G3) using a Random Forest model trained on the UCI Student Performance Dataset.
        By analyzing academic scores, study habits, attendance, and background, it enables early identification of at-risk students.
    </div></div>""",unsafe_allow_html=True)

    st.markdown("<div class='sh'>Team</div>",unsafe_allow_html=True)
    t1,t2,t3 = st.columns(3)
    for col,(name,roll) in zip([t1,t2,t3],[("Iqra Ansari","03"),("Shiwani Pandey","31"),("Sonal Sharma","37")]):
        with col:
            st.markdown(f"""<div class='sbox'>
                <div class='sv' style='font-size:1.1rem;'>{name}</div>
                <div class='sl'>Roll No. {roll}</div>
                <div class='sl' style='margin-top:2px;'>SY BSc Data Science</div>
            </div>""",unsafe_allow_html=True)

    st.markdown("""<div class='icard' style='margin-top:12px;'><div class='icard-t'>Course</div>
        <div class='icard-b'><strong>Subject:</strong> Machine Learning &middot; <strong>Program:</strong> SY BSc Data Science</div></div>""",unsafe_allow_html=True)

    a1,a2 = st.columns(2)
    with a1:
        st.markdown("""<div class='icard'><div class='icard-t'>Dataset</div>
            <div class='icard-b'><strong>UCI Student Performance Dataset</strong> — 1,044 records from Portuguese secondary schools (2005-2006). 395 Math + 649 Portuguese students with 33 attributes each.<br><br>
            <em>P. Cortez & A. Silva, "Using Data Mining to Predict Secondary School Student Performance," EUROSIS, 2008.</em></div></div>""",unsafe_allow_html=True)
        st.markdown("""<div class='icard'><div class='icard-t'>Methodology</div>
            <div class='icard-b'><strong>1.</strong> Merged Math and Portuguese datasets (1,044 records)<br>
            <strong>2.</strong> Selected 6 features from 33 available<br>
            <strong>3.</strong> Applied StandardScaler normalization<br>
            <strong>4.</strong> Trained Random Forest (200 trees, depth 10)<br>
            <strong>5.</strong> Evaluated on 20% holdout set<br>
            <strong>6.</strong> Deployed as interactive Streamlit application</div></div>""",unsafe_allow_html=True)
    with a2:
        st.markdown("""<div class='icard'><div class='icard-t'>Why Random Forest?</div>
            <div class='icard-b'><strong>Non-linear modeling</strong> — captures complex feature interactions that linear methods miss<br><br>
            <strong>Ensemble robustness</strong> — averages 200 trees, reducing sensitivity to noise<br><br>
            <strong>Built-in interpretability</strong> — provides feature importance rankings<br><br>
            <strong>Controlled complexity</strong> — max_depth of 10 prevents overfitting</div></div>""",unsafe_allow_html=True)
        st.markdown("""<div class='icard'><div class='icard-t'>Technology</div>
            <div class='icard-b'>Python &middot; Scikit-learn &middot; Pandas &middot; NumPy &middot; Plotly &middot; Streamlit &middot; Matplotlib</div></div>""",unsafe_allow_html=True)
        st.markdown("""<div class='icard'><div class='icard-t'>Known Limitations</div>
            <div class='icard-b'><strong>Temporal gap:</strong> Data from 2005-2006 Portuguese schools<br>
            <strong>Feature reduction:</strong> 6 of 33 features used for deployment simplicity<br>
            <strong>G2 dominance:</strong> Second period grade accounts for ~65-70% of model weight</div></div>""",unsafe_allow_html=True)

st.markdown("<div style='text-align:center;padding:30px 0 15px;color:#3d3555;font-size:0.72rem;letter-spacing:0.5px;'>Random Forest Regression &middot; UCI Student Performance Dataset &middot; Scikit-learn + Streamlit</div>",unsafe_allow_html=True)
