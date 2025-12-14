import numpy as np
import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Lifestyle Persona Classifier",
    page_icon="🏃",
    layout="centered",
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Global */
    .main {
        background-color: #fafafa;
    }
    h1, h2, h3 {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Card containers */
    .card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e5e5;
        background-color: #ffffff;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
        margin-top: 0.75rem;
        margin-bottom: 1.25rem;
    }

    .persona-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background-color: #ecfdf3;
        color: #166534;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .confidence-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background-color: #eff6ff;
        color: #1d4ed8;
        font-weight: 500;
        font-size: 0.85rem;
        margin-left: 0.4rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Paths to saved artifacts
# -------------------------------------------------
MODEL_PATH = "saved/activity_cnn_5classes.h5"
SCALER_PATH = "saved/scaler_activity.pkl"
ENCODER_PATH = "saved/label_encoder_activity.pkl"

# Load trained model, scaler, encoder (cached)
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


model, scaler, label_encoder = load_artifacts()

feature_cols = [
    "steps",
    "average_stress_level",
    "resting_heart_rate_in_beats_per_minute",
]

class_names = list(label_encoder.classes_)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def predict_persona(steps: float, stress: float, rest_hr: float):
    row_df = pd.DataFrame(
        [{
            "steps": steps,
            "average_stress_level": stress,
            "resting_heart_rate_in_beats_per_minute": rest_hr,
        }],
        columns=feature_cols,
    )

    x_scaled = scaler.transform(row_df)
    # Conv1D input: (batch, time_steps, channels/features)
    x_cnn = x_scaled.reshape(1, 3, 1).astype("float32")

    probs = model.predict(x_cnn, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return pred_label, probs


def generate_feedback(steps, stress, rest_hr, predicted_label, probs):
    confidence = float(np.max(probs)) * 100

    summary = (
        f"Today: **{steps:.0f} steps**, "
        f"**stress {stress:.1f}**, "
        f"**resting HR {rest_hr:.1f} bpm**."
    )

    if predicted_label == "over_trained":
        msg = (
            "Very high training load with high stress and elevated resting heart rate. "
            "Your body may not be fully recovering. Consider an easier day, more sleep, "
            "and focus on recovery and relaxation."
        )
    elif predicted_label == "high_workout":
        msg = (
            "You are highly active. This is good for fitness, but remember to include rest days "
            "and listen to early signs of fatigue or poor sleep."
        )
    elif predicted_label == "healthy":
        msg = (
            "Balanced profile. Activity, stress and resting heart rate look healthy. "
            "Try to keep that pattern consistent across the week."
        )
    elif predicted_label == "low_activity":
        msg = (
            "Stress and resting heart rate look acceptable, but your activity is relatively low today. "
            "Even 10–20 minutes of walking or light movement can already help."
        )
    else:  # lazy_obese
        msg = (
            "Low activity and relatively high resting heart rate. "
            "Start with small, realistic movement goals (for example +1,500 steps per day) "
            "and increase gradually."
        )

    footer = f"Model confidence for this persona: **{confidence:.1f}%**."

    return summary + "\n\n" + msg + "\n\n" + footer


# -------------------------------------------------
# Layout
# -------------------------------------------------
st.title("Lifestyle Persona Classifier")
st.markdown(
    "Use your daily activity and recovery signals to estimate your **lifestyle persona**. "
    "The model was trained on wearable data and distinguishes between five personas."
)

with st.expander("What personas are available?", expanded=False):
    st.markdown(
        """
        - `healthy` – balanced activity, stress and resting HR  
        - `high_workout` – consistently high activity load  
        - `low_activity` – moderate stress and HR, low movement  
        - `lazy_obese` – very low activity with higher resting HR  
        - `over_trained` – high training load with high stress and elevated resting HR  
        """
    )

st.markdown("### Input your daily values")

input_card = st.container()
with input_card:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="metric-label">Steps per day</div>', unsafe_allow_html=True)
        steps = st.slider(
            "",
            min_value=0,
            max_value=50000,
            value=9000,
            step=500,
            help="Total number of steps recorded today.",
        )

        st.markdown('<div class="metric-label">Average stress level (0–100)</div>', unsafe_allow_html=True)
        stress = st.slider(
            "",
            min_value=0,
            max_value=100,
            value=38,
            step=1,
            help="Daily average stress index from your wearable.",
        )

    with col2:
        st.markdown('<div class="metric-label">Resting heart rate (bpm)</div>', unsafe_allow_html=True)
        rest_hr = st.slider(
            "",
            min_value=30,
            max_value=120,
            value=50,
            step=1,
            help="Lowest stable heart rate during rest or sleep.",
        )

    classify = st.button("Run classification", use_container_width=True)

# -------------------------------------------------
# Results
# -------------------------------------------------
if classify:
    pred_label, probs = predict_persona(steps, stress, rest_hr)

    # Result card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Predicted persona")
    st.markdown(
        f'<span class="persona-pill">{pred_label}</span>',
        unsafe_allow_html=True,
    )

    feedback_text = generate_feedback(steps, stress, rest_hr, pred_label, probs)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(feedback_text)
    st.markdown("</div>", unsafe_allow_html=True)

    # Probabilities card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Class probabilities")

    prob_percent = (probs * 100).round(1)
    prob_df = pd.DataFrame(
        {
            "Persona": class_names,
            "Probability (%)": prob_percent,
        }
    ).sort_values("Probability (%)", ascending=False)

    st.dataframe(
        prob_df,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Set your daily values and click **Run classification** to see the result.")
