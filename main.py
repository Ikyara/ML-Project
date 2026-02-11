import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Student G3 Predictor", layout="centered")

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

st.title('Student Final Grade (G3) Predictor')
st.write('Enter the following student details to predict their final grade.')

# ── Input Features ───────────────────────────────────────────────────────────
st.header("Student Performance & Habits")
col1, col2 = st.columns(2)

with col1:
    G1 = st.slider('First Period Grade (G1)', 0, 20, 10)
    G2 = st.slider('Second Period Grade (G2)', 0, 20, 10)
    absences = st.number_input('Number of School Absences', 0, 93, 5)

with col2:
    studytime = st.slider('Weekly Study Time (1-4)', 1, 4, 2)
    failures = st.slider('Past Class Failures (0-4)', 0, 4, 0)
    higher_val = st.radio('Wants Higher Education?', ['Yes', 'No'], index=0)

# ── Build input DataFrame with EXACTLY the features the model expects ────────
def create_input_df(G1, G2, absences, studytime, failures, higher_val, feature_columns):
    input_data = {
        'G1': G1,
        'G2': G2,
        'absences': absences,
        'studytime': studytime,
        'failures': failures,
        'higher_yes': 1 if higher_val == 'Yes' else 0,
    }
    # Create DataFrame with columns in the exact order the model was trained on
    return pd.DataFrame([input_data])[feature_columns]

input_df = create_input_df(G1, G2, absences, studytime, failures, higher_val, feature_columns)

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button('Predict Final Grade (G3)'):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_grade = np.clip(prediction[0], 0, 20)  # G3 is 0-20

    st.subheader('Predicted Final Grade (G3):')
    st.success(f'{predicted_grade:.2f}')

    # Show a helpful context bar
    st.progress(predicted_grade / 20.0)
    if predicted_grade >= 14:
        st.info("This student is predicted to perform well!")
    elif predicted_grade >= 10:
        st.info("This student is predicted to pass.")
    else:
        st.warning("This student may be at risk of failing.")
