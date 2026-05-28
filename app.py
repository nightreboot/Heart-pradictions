import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# ── Load artefacts ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model   = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))
    scaler  = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    return model, scaler, columns

model, scaler, COLUMNS = load_artifacts()

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_input_df(age, sex, chest_pain, resting_bp, cholesterol,
                   fasting_bs, resting_ecg, max_hr, exercise_angina,
                   oldpeak, st_slope):
    """One-hot encode raw inputs and align with training columns."""
    raw = {
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": int(fasting_bs),
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        # one-hot
        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA":  int(chest_pain == "TA"),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST":     int(resting_ecg == "ST"),
        "ExerciseAngina_Y":  int(exercise_angina == "Yes"),
        "ST_Slope_Flat":     int(st_slope == "Flat"),
        "ST_Slope_Up":       int(st_slope == "Up"),
    }
    df = pd.DataFrame([raw])[COLUMNS]
    return df


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Attack Risk Predictor")
st.markdown(
    "Fill in your clinical details below and click **Predict** to get an "
    "instant risk assessment powered by a K-Nearest Neighbours model."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA = Atypical Angina · NAP = Non-Anginal Pain · "
                 "TA = Typical Angina · ASY = Asymptomatic",
        )
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120
        )
        cholesterol = st.number_input(
            "Serum Cholesterol (mg/dL)", min_value=0, max_value=700, value=200
        )
        fasting_bs = st.checkbox("Fasting Blood Sugar > 120 mg/dL")

    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            help="ST = ST-T wave abnormality · LVH = Left Ventricular Hypertrophy",
        )
        max_hr = st.number_input(
            "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150
        )
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input(
            "Oldpeak (ST depression)", min_value=0.0, max_value=10.0,
            value=1.0, step=0.1, format="%.1f"
        )
        st_slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            ["Up", "Flat", "Down"],
        )

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_df     = build_input_df(age, sex, chest_pain, resting_bp,
                                  cholesterol, fasting_bs, resting_ecg,
                                  max_hr, exercise_angina, oldpeak, st_slope)
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    risk_pct = proba[1] * 100

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** — estimated probability: **{risk_pct:.1f}%**")
        st.markdown(
            "> The model suggests elevated risk. Please consult a cardiologist "
            "for a thorough evaluation."
        )
    else:
        st.success(f"✅ **Low Risk of Heart Disease** — estimated probability: **{risk_pct:.1f}%**")
        st.markdown(
            "> The model suggests low risk. Continue maintaining a healthy "
            "lifestyle and regular check-ups."
        )

    # probability bar
    st.markdown("**Risk Probability**")
    st.progress(int(risk_pct))
    c1, c2 = st.columns(2)
    c1.metric("No Disease", f"{proba[0]*100:.1f}%")
    c2.metric("Heart Disease", f"{proba[1]*100:.1f}%")

    # show input summary
    with st.expander("📋 Input Summary"):
        summary = {
            "Age": age, "Sex": sex, "Chest Pain": chest_pain,
            "Resting BP": resting_bp, "Cholesterol": cholesterol,
            "Fasting BS > 120": fasting_bs, "Resting ECG": resting_ecg,
            "Max HR": max_hr, "Exercise Angina": exercise_angina,
            "Oldpeak": oldpeak, "ST Slope": st_slope,
        }
        st.table(pd.DataFrame(summary, index=["Value"]).T)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational purposes only and does "
    "not constitute medical advice. Always consult a qualified healthcare professional."
)