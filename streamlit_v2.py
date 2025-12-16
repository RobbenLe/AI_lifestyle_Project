import numpy as np
import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# --------------------------
# Paths (same folder as this file)
# --------------------------
MODEL_PATH = "saved/activity_cnn_5classes_v2.keras"
SCALER_PATH = "saved/scaler_activity_v2.pkl"
ENCODER_PATH = "saved/label_encoder_activity_v2.pkl"

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()
class_names = list(label_encoder.classes_)

# --------------------------
# Predict + feedback
# --------------------------
def classify_and_feedback(steps, stress, hr_avg):
    feature_cols = ["steps", "average_stress_level", "heart_rate_per_point"]

    row_df = pd.DataFrame([{
        "steps": float(steps),
        "average_stress_level": float(stress),
        "heart_rate_per_point": float(hr_avg),
    }], columns=feature_cols)

    x_scaled = scaler.transform(row_df)
    x_cnn = x_scaled.reshape(1, 3, 1).astype("float32")

    probs = model.predict(x_cnn, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(probs)) * 100

    summary = (
        f"Today: {steps:.0f} steps, stress {stress:.1f}, "
        f"average HR {hr_avg:.1f} bpm.\n\n"
    )

    if pred_label == "over_trained":
        msg = (
            "You are training very hard and your stress and heart signals are high. "
            "Consider adding a lighter day, more sleep, hydration, and recovery activities."
        )
    elif pred_label == "high_workout":
        msg = (
            "You have a high activity level. Great work. "
            "Balance hard training with recovery to stay consistent long-term."
        )
    elif pred_label == "healthy":
        msg = (
            "Your activity, stress, and heart rate are well balanced. "
            "This is a sustainable and healthy pattern. Keep it up."
        )
    elif pred_label == "low_activity":
        msg = (
            "Your movement is a bit low today. "
            "Even 10–20 minutes of walking or light exercise can make a big difference."
        )
    else:  # lazy_obese
        msg = (
            "Your activity is quite low and your heart is working relatively hard. "
            "Start with small daily goals (short walks) and increase slowly."
        )

    probs_dict = {cls: float(p) for cls, p in zip(class_names, probs)}

    feedback_text = (
        f"Predicted persona: {pred_label}\n"
        f"Model confidence: {confidence:.1f}%\n\n"
        + summary
        + msg
    )

    return pred_label, confidence, probs_dict, feedback_text


# --------------------------
# Streamlit UI
# --------------------------
st.title("Lifestyle Persona Classifier (v2)")
st.write(
    "Input your daily values and the CNN model will classify into 5 personas: "
    "`high_workout`, `healthy`, `low_activity`, `lazy_obese`, `over_trained`."
)

col1, col2 = st.columns(2)

with col1:
    steps = st.number_input("Steps per day", min_value=0, max_value=60000, value=8000, step=500)
    stress = st.slider("Average stress level (0–100)", min_value=0, max_value=100, value=40, step=1)

with col2:
    hr_avg = st.slider("Average heart rate today (bpm)", min_value=70, max_value=140, value=95, step=1)

if st.button("Classify"):
    pred_label, confidence, probs_dict, feedback_text = classify_and_feedback(steps, stress, hr_avg)

    st.subheader("Result")
    st.write(feedback_text)

    st.subheader("Class probabilities")
    st.write(probs_dict)

    # Optional: show the top-2 classes
    top2 = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:2]
    st.caption(f"Top-2: {top2[0][0]} ({top2[0][1]:.2f}), {top2[1][0]} ({top2[1][1]:.2f})")
