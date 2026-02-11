import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stApp"], [data-testid="stBottom"], .main, .block-container,
    [data-testid="stMainBlockContainer"], .appview-container,
    header[data-testid="stHeader"] {
        background-color: #0e0b1e !important;
    }
    html {
        overscroll-behavior: none;
        background: #0e0b1e !important;
    }
    [data-testid="stDecoration"] { display: none !important; }
    header[data-testid="stHeader"] { background: #0e0b1e !important; }
    [data-testid="stSidebar"] { display: none !important; }
    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background: #0e0b1e;
        font-family: 'Inter', sans-serif;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(168,85,247,0.05);
        border-radius: 12px;
        padding: 3px;
        border: 1px solid rgba(168,85,247,0.12);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 18px;
        color: #8b7fb8;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #a855f7, #ec4899) !important;
        color: white !important;
    }

    /* Cards */
    .card {
        background: rgba(168,85,247,0.07);
        border: 1px solid rgba(168,85,247,0.15);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .card-val {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .card-lbl {
        color: #7a6fa5;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* Grade result */
    .grade-box {
        background: linear-gradient(135deg, rgba(168,85,247,0.12), rgba(236,72,153,0.06));
        border: 2px solid rgba(168,85,247,0.25);
        border-radius: 20px;
        padding: 35px 20px;
        text-align: center;
    }
    .grade-num {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .grade-sub {
        color: #7a6fa5;
        font-size: 0.9rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* Status */
    .status-good {
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        color: #4ade80;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin-top: 12px;
        font-size: 0.9rem;
    }
    .status-ok {
        background: rgba(168,85,247,0.1);
        border: 1px solid rgba(168,85,247,0.25);
        color: #c084fc;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin-top: 12px;
        font-size: 0.9rem;
    }
    .status-risk {
        background: rgba(236,72,153,0.1);
        border: 1px solid rgba(236,72,153,0.25);
        color: #f472b6;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin-top: 12px;
        font-size: 0.9rem;
    }

    /* Info box */
    .info-box {
        background: rgba(168,85,247,0.05);
        border: 1px solid rgba(168,85,247,0.1);
        border-radius: 12px;
        padding: 18px;
        margin: 8px 0;
    }
    .info-title { color: #c084fc; font-weight: 700; font-size: 0.9rem; margin-bottom: 6px; }
    .info-text { color: #b0a3d4; font-size: 0.85rem; line-height: 1.6; }

    /* Section */
    .sec-head {
        color: #e0d4f7;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(168,85,247,0.12);
    }
    .sec-desc { color: #8b7fb8; font-size: 0.85rem; line-height: 1.5; margin-bottom: 15px; }

    /* Stat */
    .stat-box {
        background: rgba(168,85,247,0.06);
        border: 1px solid rgba(168,85,247,0.12);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stat-val { font-size: 1.5rem; font-weight: 800; color: #c084fc; }
    .stat-lbl { color: #7a6fa5; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

    /* Table */
    .s-table { width: 100%; border-collapse: collapse; }
    .s-table th { background: rgba(168,85,247,0.1); color: #c084fc; padding: 10px 14px; text-align: left; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(168,85,247,0.15); }
    .s-table td { padding: 10px 14px; color: #b0a3d4; font-size: 0.85rem; border-bottom: 1px solid rgba(168,85,247,0.05); }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #ec4899);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 35px;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(168,85,247,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ── Train if needed ──────────────────────────────────────────────────────────
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    rf.fit(X_train_scaled, y_train)
    preds = rf.predict(X_test_scaled)

    m = {
        'mae': mean_absolute_error(y_test, preds),
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'r2': r2_score(y_test, preds),
        'n_train': len(X_train), 'n_test': len(X_test),
        'y_test': y_test.values.tolist(), 'y_pred': preds.tolist(),
    }
    n, p = X_test.shape
    m['adj_r2'] = 1 - (1 - m['r2']) * (n - 1) / (n - p - 1)

    for name, obj in [('random_forest_model.pkl', rf), ('scaler.pkl', scaler),
                       ('feature_columns.pkl', FEATURES), ('metrics.pkl', m)]:
        with open(name, 'wb') as f: pickle.dump(obj, f)

if not all(os.path.exists(f) for f in ['random_forest_model.pkl', 'scaler.pkl', 'feature_columns.pkl', 'metrics.pkl']):
    with st.spinner('Training model...'): train_and_save()

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

# ── Plotly defaults ──────────────────────────────────────────────────────────
PL = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
          font=dict(color='#9d8ec7', family='Inter'), margin=dict(l=40, r=30, t=35, b=40))

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 15px 0 5px 0;'>
    <span style='font-size: 2rem;'>🎓</span>
    <span style='font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'> Student Grade Predictor</span>
    <p style='color: #7a6fa5; font-size: 0.88rem; margin-top: 4px;'>
        Predict final grades using Random Forest · UCI Student Performance Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Dataset", "🧠 Model", "📖 About"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    # Inputs — all on main page, compact row
    st.markdown("<div class='sec-head'>Enter Student Details</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: G1 = st.slider('G1', 0, 20, 10)
    with c2: G2 = st.slider('G2', 0, 20, 10)
    with c3: studytime = st.slider('Study Time', 1, 4, 2)
    with c4: absences = st.number_input('Absences', 0, 30, 5)
    with c5: failures = st.slider('Failures', 0, 4, 0)
    with c6: higher_val = st.radio('Higher Ed?', ['Yes', 'No'], horizontal=True)

    predict_btn = st.button('🔮 Predict Final Grade', use_container_width=True)

    if predict_btn:
        inp = pd.DataFrame([{
            'G1': G1, 'G2': G2, 'absences': absences,
            'studytime': studytime, 'failures': failures,
            'higher_yes': 1 if higher_val == 'Yes' else 0,
        }])[feature_columns]
        pred = float(np.clip(model.predict(scaler.transform(inp))[0], 0, 20))
        pct = pred / 20 * 100
        letter = "A" if pred >= 16 else ("B" if pred >= 14 else ("C" if pred >= 12 else ("D" if pred >= 10 else "F")))

        st.markdown("---")

        # Result row
        left, mid, right = st.columns([1, 1.2, 1])

        with left:
            st.markdown(f"""<div class='grade-box'>
                <div class='grade-sub'>Predicted Grade</div>
                <div class='grade-num'>{pred:.1f}</div>
                <div class='grade-sub'>out of 20 · {pct:.0f}% · {letter}</div>
            </div>""", unsafe_allow_html=True)
            if pred >= 14:
                st.markdown("<div class='status-good'>✅ Strong performance predicted</div>", unsafe_allow_html=True)
            elif pred >= 10:
                st.markdown("<div class='status-ok'>📘 On track to pass</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-risk'>⚠️ At risk — may need support</div>", unsafe_allow_html=True)

        with mid:
            # Grade progression
            fig = go.Figure(go.Bar(
                x=['G1', 'G2', 'G3 (pred)'], y=[G1, G2, pred],
                marker_color=['#a855f7', '#c084fc', '#ec4899'],
                text=[f"{G1}", f"{G2}", f"{pred:.1f}"],
                textposition='outside', textfont=dict(color='#e0d4f7', size=14),
            ))
            fig.update_layout(**PL, height=300, yaxis=dict(range=[0, 22], gridcolor='rgba(168,85,247,0.08)'),
                              xaxis_title='', yaxis_title='Grade')
            st.plotly_chart(fig, use_container_width=True)

        with right:
            # Feature importance
            imp = model.feature_importances_
            fdf = pd.DataFrame({'F': ['G1','G2','Absences','Study','Failures','Higher Ed'], 'I': imp}).sort_values('I')
            fig2 = go.Figure(go.Bar(
                x=fdf['I'], y=fdf['F'], orientation='h',
                marker=dict(color=fdf['I'], colorscale=[[0,'#ec4899'],[1,'#c084fc']]),
                text=[f"{v:.0%}" for v in fdf['I']], textposition='outside',
                textfont=dict(color='#e0d4f7', size=11),
            ))
            fig2.update_layout(**PL, height=300, xaxis=dict(gridcolor='rgba(168,85,247,0.08)',
                               range=[0, max(imp)*1.35], title='Importance'), yaxis_title='')
            st.plotly_chart(fig2, use_container_width=True)

        # Insights
        st.markdown("<div class='sec-head'>💡 Insights</div>", unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        trend = pred - G2
        with i1:
            t = f"up {trend:.1f}" if trend > 0.5 else (f"down {abs(trend):.1f}" if trend < -0.5 else "stable")
            st.markdown(f"""<div class='info-box'>
                <div class='info-title'>📈 Trend</div>
                <div class='info-text'>G3 is <strong>{t}</strong> from G2. G1→G2 was {'+' if G2-G1>=0 else ''}{G2-G1} pts.</div>
            </div>""", unsafe_allow_html=True)
        with i2:
            avg = df_raw['absences'].mean()
            st.markdown(f"""<div class='info-box'>
                <div class='info-title'>🏫 Attendance</div>
                <div class='info-text'>{absences} absences (avg: {avg:.0f}). {'Good attendance!' if absences <= 5 else ('Above average — could improve.' if absences <= 15 else 'High absences — likely hurting grades.')}</div>
            </div>""", unsafe_allow_html=True)
        with i3:
            sl = {1:"<2hrs",2:"2-5hrs",3:"5-10hrs",4:">10hrs"}
            st.markdown(f"""<div class='info-box'>
                <div class='info-title'>📚 Study & Background</div>
                <div class='info-text'>Studying {sl[studytime]}/week. {failures} past failure(s). {'Wants higher ed ✓' if higher_val=='Yes' else 'No higher ed plans.'}</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align:center; padding: 50px 0; color: #7a6fa5;'>
            <p style='font-size: 2.5rem;'>🔮</p>
            <p style='font-size: 1.1rem; color: #c084fc; font-weight: 600;'>Set values above and click Predict</p>
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("<div class='sec-head'>About the Dataset</div>", unsafe_allow_html=True)
    st.markdown("""<p class='sec-desc'>
        The <strong>UCI Student Performance Dataset</strong> contains 1,044 records from two Portuguese secondary schools,
        covering Math and Portuguese courses. Collected in 2005-2006 via school reports and questionnaires,
        it includes 33 features per student. G3 (final grade, 0-20 scale) is our prediction target.
    </p>""", unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    for col, (l, v) in zip([s1,s2,s3,s4,s5], [
        ("Students", len(df_raw)), ("Features", "33"),
        ("Math", len(df_raw[df_raw['subject']=='Math'])),
        ("Portuguese", len(df_raw[df_raw['subject']=='Portuguese'])),
        ("Avg Grade", f"{df_raw['G3'].mean():.1f}")
    ]):
        with col:
            st.markdown(f"<div class='stat-box'><div class='stat-lbl'>{l}</div><div class='stat-val'>{v}</div></div>", unsafe_allow_html=True)

    st.markdown("")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("<div class='sec-head'>Grade Distribution</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>Most students score between 10-14. The spike at 0 represents dropouts. Dashed line = passing threshold.</p>", unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=df_raw['G3'], nbinsx=21, marker=dict(color='#a855f7', line=dict(color='#c084fc', width=1)), opacity=0.85))
        fig.add_vline(x=10, line_dash="dash", line_color="#ec4899", annotation_text="Pass", annotation_font_color="#ec4899")
        fig.update_layout(**PL, height=320, xaxis_title='G3', yaxis_title='Students')
        fig.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("<div class='sec-head'>Grade by Subject</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>Portuguese students score slightly higher on average (11.9 vs 10.4). Math has wider spread and more low outliers.</p>", unsafe_allow_html=True)
        fig = go.Figure()
        for s, c in [('Math','#ec4899'),('Portuguese','#a855f7')]:
            fig.add_trace(go.Box(y=df_raw[df_raw['subject']==s]['G3'], name=s, marker_color=c, line_color=c, boxmean=True))
        fig.update_layout(**PL, height=320, yaxis_title='G3', legend=dict(font=dict(color='#9d8ec7')))
        fig.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig, use_container_width=True)

    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("<div class='sec-head'>G2 vs G3 Correlation</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>G2 and G3 are highly correlated — points cluster tightly around the diagonal. This is why G2 dominates feature importance.</p>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_raw['G2'], y=df_raw['G3'], mode='markers', marker=dict(color='#a855f7', opacity=0.4, size=5), name='Students'))
        fig.add_trace(go.Scatter(x=[0,20], y=[0,20], mode='lines', line=dict(color='#ec4899', dash='dash', width=2), name='Perfect'))
        fig.update_layout(**PL, height=320, xaxis_title='G2', yaxis_title='G3', legend=dict(font=dict(color='#9d8ec7')))
        fig.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig, use_container_width=True)

    with ch4:
        st.markdown("<div class='sec-head'>Absences vs Grade</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>Weak but negative relationship. Students with high absences rarely score above 15. The effect is indirect — absences lower G2, which lowers G3.</p>", unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(x=df_raw['absences'], y=df_raw['G3'], mode='markers', marker=dict(color='#ec4899', opacity=0.35, size=5)))
        fig.update_layout(**PL, height=320, xaxis_title='Absences', yaxis_title='G3', showlegend=False)
        fig.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig, use_container_width=True)

    ch5, ch6 = st.columns(2)

    with ch5:
        st.markdown("<div class='sec-head'>Study Time vs Grade</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>More study time correlates with slightly higher grades, but variance is large at every level.</p>", unsafe_allow_html=True)
        sa = df_raw.groupby('studytime')['G3'].agg(['mean','std']).reset_index()
        sl = {1:'<2h', 2:'2-5h', 3:'5-10h', 4:'>10h'}
        fig = go.Figure(go.Bar(
            x=[sl[s] for s in sa['studytime']], y=sa['mean'],
            marker_color=['#ec4899','#c084fc','#a855f7','#7c3aed'],
            text=[f"{v:.1f}" for v in sa['mean']], textposition='outside',
            textfont=dict(color='#e0d4f7', size=13),
            error_y=dict(type='data', array=sa['std'].tolist(), color='#7a6fa5', thickness=1.5),
        ))
        fig.update_layout(**PL, height=320, yaxis=dict(range=[0,18], gridcolor='rgba(168,85,247,0.08)'), yaxis_title='Avg G3', xaxis_title='Study Time')
        st.plotly_chart(fig, use_container_width=True)

    with ch6:
        st.markdown("<div class='sec-head'>Failures vs Grade</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>Clear negative pattern: 0 failures averages 12.1, while 3 failures averages only 6.8. Most students (82%) have 0 failures.</p>", unsafe_allow_html=True)
        fa = df_raw.groupby('failures')['G3'].agg(['mean','count']).reset_index()
        fig = go.Figure(go.Bar(
            x=[str(f) for f in fa['failures']], y=fa['mean'],
            marker_color=['#22c55e','#c084fc','#ec4899','#f43f5e','#dc2626'][:len(fa)],
            text=[f"{v:.1f} (n={c})" for v, c in zip(fa['mean'], fa['count'])],
            textposition='outside', textfont=dict(color='#e0d4f7', size=12),
        ))
        fig.update_layout(**PL, height=320, yaxis=dict(range=[0,16], gridcolor='rgba(168,85,247,0.08)'), yaxis_title='Avg G3', xaxis_title='Past Failures')
        st.plotly_chart(fig, use_container_width=True)

    # Feature table
    st.markdown("<div class='sec-head'>Feature Descriptions</div>", unsafe_allow_html=True)
    st.markdown("""
    <table class='s-table'>
        <tr><th>Feature</th><th>Description</th><th>Range</th><th>Type</th></tr>
        <tr><td>G1</td><td>First period grade</td><td>0–20</td><td>Numeric</td></tr>
        <tr><td>G2</td><td>Second period grade (strongest predictor)</td><td>0–20</td><td>Numeric</td></tr>
        <tr><td>Absences</td><td>School absences</td><td>0–30</td><td>Numeric</td></tr>
        <tr><td>Study Time</td><td>Weekly: 1(<2h), 2(2-5h), 3(5-10h), 4(>10h)</td><td>1–4</td><td>Ordinal</td></tr>
        <tr><td>Failures</td><td>Past class failures</td><td>0–4</td><td>Numeric</td></tr>
        <tr><td>Higher Ed</td><td>Wants higher education</td><td>Yes/No</td><td>Binary</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("<div class='sec-head'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<p class='sec-desc'>Random Forest Regressor with 200 trees, max depth 10. Trained on 80% of data, tested on 20%.</p>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (l, v, s) in zip([m1,m2,m3,m4], [
        ("MAE", f"{metrics['mae']:.3f}", "Avg error in points"),
        ("RMSE", f"{metrics['rmse']:.3f}", "Penalizes big errors"),
        ("R²", f"{metrics['r2']:.3f}", "Variance explained"),
        ("Adj R²", f"{metrics['adj_r2']:.3f}", "Adjusted for features"),
    ]):
        with col:
            st.markdown(f"<div class='stat-box'><div class='stat-lbl'>{l}</div><div class='stat-val'>{v}</div><div class='stat-lbl'>{s}</div></div>", unsafe_allow_html=True)

    st.markdown("")

    # Metric explanations
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"""<div class='info-box'>
            <div class='info-title'>📏 MAE = {metrics['mae']:.3f}</div>
            <div class='info-text'>On average, predictions are off by ~{metrics['mae']:.1f} points on a 0-20 scale. If the model predicts 14, the actual is typically between {14-metrics['mae']:.1f} and {14+metrics['mae']:.1f}.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class='info-box'>
            <div class='info-title'>📐 RMSE = {metrics['rmse']:.3f}</div>
            <div class='info-text'>Like MAE but penalizes larger errors more. The gap between RMSE and MAE is only {metrics['rmse']-metrics['mae']:.2f}, meaning the model rarely makes big mistakes.</div>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown(f"""<div class='info-box'>
            <div class='info-title'>📊 R² = {metrics['r2']:.3f}</div>
            <div class='info-text'>The model explains {metrics['r2']*100:.1f}% of grade variance. This is {'excellent' if metrics['r2']>0.85 else 'very good'} — especially with only 6 features out of 33 available.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class='info-box'>
            <div class='info-title'>🔧 Adj R² = {metrics['adj_r2']:.3f}</div>
            <div class='info-text'>R² adjusted for feature count. It's very close to R² ({metrics['r2']:.3f}), confirming all 6 features contribute meaningfully — none are redundant.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    p1, p2 = st.columns(2)

    with p1:
        st.markdown("<div class='sec-head'>Predicted vs Actual</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>Points near the diagonal = accurate predictions. The model does well in the 8-16 range but struggles with dropouts (G3=0).</p>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics['y_test'], y=metrics['y_pred'], mode='markers', marker=dict(color='#a855f7', opacity=0.5, size=6), name='Predictions'))
        fig.add_trace(go.Scatter(x=[0,20], y=[0,20], mode='lines', line=dict(color='#ec4899', dash='dash', width=2), name='Perfect'))
        fig.update_layout(**PL, height=350, xaxis_title='Actual G3', yaxis_title='Predicted G3', legend=dict(font=dict(color='#9d8ec7')))
        fig.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        st.markdown("<div class='sec-head'>Error Distribution</div>", unsafe_allow_html=True)
        st.markdown("<p class='sec-desc'>Centered at 0 = no bias. Most errors are within ±2 points. The left tail comes from dropouts the model couldn't predict.</p>", unsafe_allow_html=True)
        errs = [p - a for p, a in zip(metrics['y_pred'], metrics['y_test'])]
        fig = go.Figure(go.Histogram(x=errs, nbinsx=30, marker=dict(color='#ec4899', line=dict(color='#f472b6', width=1)), opacity=0.8))
        fig.add_vline(x=0, line_dash="dash", line_color="#a855f7", line_width=2)
        fig.update_layout(**PL, height=350, xaxis_title='Error (Pred - Actual)', yaxis_title='Count')
        fig.update_xaxes(gridcolor='rgba(168,85,247,0.08)')
        fig.update_yaxes(gridcolor='rgba(168,85,247,0.08)')
        st.plotly_chart(fig, use_container_width=True)

    # Config table
    st.markdown("<div class='sec-head'>Model Configuration</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <table class='s-table'>
        <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        <tr><td>Algorithm</td><td>Random Forest Regressor</td><td>Ensemble of 200 decision trees</td></tr>
        <tr><td>n_estimators</td><td>200</td><td>Number of trees</td></tr>
        <tr><td>max_depth</td><td>10</td><td>Prevents overfitting</td></tr>
        <tr><td>Scaler</td><td>StandardScaler</td><td>Zero mean, unit variance</td></tr>
        <tr><td>Split</td><td>80/20</td><td>{metrics['n_train']} train / {metrics['n_test']} test</td></tr>
        <tr><td>Features</td><td>6</td><td>G1, G2, absences, studytime, failures, higher_yes</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: ABOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("<div class='sec-head'>About This Project</div>", unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
        <div class='info-title'>🎯 Goal</div>
        <div class='info-text'>Predict a student's final grade (G3) using machine learning, enabling early identification of at-risk students and timely intervention.</div>
    </div>""", unsafe_allow_html=True)

    # Team
    st.markdown("<div class='sec-head'>👩‍💻 Team</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    for col, (name, roll) in zip([t1,t2,t3], [("Iqra Ansari","03"), ("Shiwani Pandey","31"), ("Sonal Sharma","37")]):
        with col:
            st.markdown(f"""<div class='stat-box'>
                <div style='font-size:1.5rem;'>🎓</div>
                <div class='stat-val' style='font-size:1.1rem;'>{name}</div>
                <div class='stat-lbl'>Roll No. {roll} · SY BSc Data Science</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='margin-top:12px;'>
        <div class='info-title'>📚 Course</div>
        <div class='info-text'><strong>Subject:</strong> Machine Learning · <strong>Program:</strong> SY BSc Data Science</div>
    </div>""", unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""<div class='info-box'>
            <div class='info-title'>📊 Dataset</div>
            <div class='info-text'>
                <strong>UCI Student Performance Dataset</strong> — collected from two Portuguese secondary schools (2005-2006).
                1,044 records: 395 Math + 649 Portuguese students with 33 attributes each.<br><br>
                <em>Citation: P. Cortez & A. Silva, "Using Data Mining to Predict Secondary School Student Performance," EUROSIS, 2008.</em>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
            <div class='info-title'>🔬 Methodology</div>
            <div class='info-text'>
                <strong>1.</strong> Merged Math + Portuguese datasets (1,044 records)<br>
                <strong>2.</strong> Selected 6 key features from 33 available<br>
                <strong>3.</strong> Applied StandardScaler normalization<br>
                <strong>4.</strong> Trained Random Forest (200 trees, depth 10)<br>
                <strong>5.</strong> Evaluated on 20% holdout set<br>
                <strong>6.</strong> Deployed as Streamlit web app
            </div>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown("""<div class='info-box'>
            <div class='info-title'>🤖 Why Random Forest?</div>
            <div class='info-text'>
                <strong>Non-linear:</strong> Captures complex patterns that Linear Regression misses.<br><br>
                <strong>Robust:</strong> Averages 200 trees, resistant to noise and outliers.<br><br>
                <strong>Interpretable:</strong> Provides feature importance scores.<br><br>
                <strong>No overfitting:</strong> max_depth=10 keeps it generalizable.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
            <div class='info-title'>🛠️ Tech Stack</div>
            <div class='info-text'>
                <strong>Python</strong> · <strong>Scikit-learn</strong> · <strong>Pandas</strong> · <strong>NumPy</strong> · <strong>Plotly</strong> · <strong>Streamlit</strong> · <strong>Matplotlib</strong>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
            <div class='info-title'>⚠️ Limitations</div>
            <div class='info-text'>
                <strong>Data age:</strong> From 2005-2006 Portuguese schools.<br>
                <strong>Feature subset:</strong> 6 of 33 features used for clean deployment.<br>
                <strong>G2 dominance:</strong> G2 heavily influences predictions; removing it would reveal more about behavioral factors.
            </div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; padding: 30px 0 15px 0; color: #4a4270; font-size: 0.75rem;'>
    Built with Streamlit & Scikit-learn · Random Forest · UCI Student Performance Dataset
</div>
""", unsafe_allow_html=True)
