import streamlit as st
import numpy as np
import joblib

# ========================
# Load the model
# ========================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("pregnancy_risk_model.joblib")
    scaler = joblib.load("pregnancy_scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# ========================
# Streamlit Interface
# ========================
st.set_page_config(page_title="Pregnancy Risk", page_icon="👶", layout="centered")

st.title("👶 Pregnancy Risk Prediction")

st.markdown(
    """
This application uses a **Random Forest** model to estimate pregnancy risk based on 6 clinical measurements.  
The possible classes are:

- 🟢 **0: Low-risk pregnancy**  
- 🟠 **1: Moderate-risk pregnancy**  
- 🔴 **2: High-risk pregnancy**
    """
)

st.markdown("---")

# ========================
# Input of the 6 features
# (adjust ranges if needed)
# ========================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=10, max_value=60, value=30)
    tension_sys = st.number_input("Systolic blood pressure (mmHg)", min_value=70, max_value=200, value=120)
    tension_dia = st.number_input("Diastolic blood pressure (mmHg)", min_value=40, max_value=130, value=80)

with col2:
    glycemia = st.number_input("Blood glucose (mmol/L)", min_value=2.0, max_value=25.0, value=5.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, value=36.8, step=0.1)
    resting_hr = st.number_input("Resting heart rate (bpm)", min_value=40, max_value=180, value=75)

st.markdown("---")

if st.button("🩺 Evaluate Risk"):
    # Prepare input
    X = np.array([[age, tension_sys, tension_dia, glycemia, temperature, resting_hr]])

    # Apply the same normalization used during training
    X_scaled = scaler.transform(X)

    # Prediction
    y_pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]

    # Labels + colors
    labels = {
        0: "Low-risk pregnancy",
        1: "Moderate-risk pregnancy",
        2: "High-risk pregnancy",
    }

    colors = {
        0: "#2ecc71",  # green
        1: "#f39c12",  # orange
        2: "#e74c3c",  # red
    }

    st.markdown("### Result")

    st.markdown(
        f"""
        <div style="
            padding: 1.2rem;
            border-radius: 0.75rem;
            background-color: {colors[y_pred]};
            color: white;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;">
            {labels[y_pred]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show probabilities (optional but useful)
    st.markdown("#### Model estimated probabilities")
    st.write(
        {
            "Class 0 (low-risk)": round(proba[0], 3),
            "Class 1 (moderate-risk)": round(proba[1], 3),
            "Class 2 (high-risk)": round(proba[2], 3),
        }
    )
else:
    st.info("Enter the values and click **Evaluate Risk**.")
