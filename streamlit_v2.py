import numpy as np
import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# --------------------------
# Page config (professional layout)
# --------------------------
st.set_page_config(
    page_title="Lifestyle Persona Classifier",
    page_icon="🧠",
    layout="wide"
)

# --------------------------
# Simple styling (clean + professional)
# --------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
h1 { margin-bottom: 0.2rem; }
.small-note { color: #6b7280; font-size: 0.9rem; }
.badge {
  display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
  font-size: 0.85rem; font-weight: 600;
}
.badge-green { background:#dcfce7; color:#166534; }
.badge-yellow { background:#fef9c3; color:#854d0e; }
.badge-red { background:#fee2e2; color:#991b1b; }
.card {
  border: 1px solid #e5e7eb; border-radius: 14px; padding: 1rem;
  background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.hr { border-top:1px solid #e5e7eb; margin: 0.75rem 0; }
.metric-title { color:#374151; font-size:0.9rem; margin-bottom:0.15rem; }
.metric-value { font-size:1.6rem; font-weight:700; margin:0; }
.metric-sub { color:#6b7280; font-size:0.85rem; margin-top:0.15rem; }
</style>
""", unsafe_allow_html=True)

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
# NEW: Colored range bar under slider (updates live)
# --------------------------
def colored_range_bar(value, vmin, vmax, segments, height=10):
    """
    Draw a color-coded bar under a slider.

    segments: list of tuples (start, end, color_hex)
      Example: [(0, 5000, "#ef4444"), (5000, 8000, "#f59e0b"), ...]
    """
    value = max(vmin, min(vmax, float(value)))
    span = (vmax - vmin) if vmax != vmin else 1
    pos_pct = ((value - vmin) / span) * 100

    stops = []
    for a, b, c in segments:
        a_pct = ((a - vmin) / span) * 100
        b_pct = ((b - vmin) / span) * 100
        a_pct = max(0, min(100, a_pct))
        b_pct = max(0, min(100, b_pct))
        stops.append(f"{c} {a_pct:.2f}% {b_pct:.2f}%")
    gradient = ", ".join(stops)

    st.markdown(
        f"""
        <div style="position:relative; width:100%; margin-top:-6px; margin-bottom:12px;">
          <div style="
            height:{height}px;
            border-radius:999px;
            background: linear-gradient(90deg, {gradient});
            border: 1px solid #e5e7eb;
          "></div>

          <div style="
            position:absolute;
            top:-5px;
            left: calc({pos_pct:.2f}% - 6px);
            width:12px; height:12px;
            border-radius:999px;
            background:#111827;
            border:2px solid white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
          "></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------
# Updated Risk/Status helpers (MATCH your color thresholds)
# --------------------------
def steps_status(steps: float):
    # Red: 0–5k, Yellow: 5k–8k, Green: 8k–15k, Red: >15k
    if steps < 3000:
        return "Low", "badge-red", "0–3k"
    if steps >= 3000 and steps <= 6000:
        return "Moderate", "badge-yellow", "3k–6k"
    if steps >=6000 and steps <= 19000:
        return "Good", "badge-green", "6k–19k"
    return "Too high", "badge-red", ">19k"

def stress_status(stress: float):
    # Green: 0–45, Yellow: 45–70, Red: 71–100
    if stress <= 40:
        return "Good", "badge-green", "0–40"
    if stress >= 40 and stress <= 60:
        return "Moderate", "badge-yellow", "40–60"
    return "High", "badge-red", "60–100"

def hr_status(hr_avg: float):
    # Red: 60–80, Green: 80–110, Yellow: 110–125, Red: 125–140
    if hr_avg >= 60 and hr_avg <= 82:
        return "Low", "badge-red", "60–79"
    if hr_avg >= 82 and hr_avg <= 94:
        return "Moderate", "badge-yellow", "82–94"
    if hr_avg >= 94 and hr_avg <= 130:
        return "Good", "badge-green", "94–130"
    return "High", "badge-red", "130–140"

def persona_badge(persona: str):
    mapping = {
        "healthy": ("badge-green", "Balanced"),
        "high_workout": ("badge-green", "Active"),
        "low_activity": ("badge-yellow", "Needs movement"),
        "lazy_obese": ("badge-red", "Sedentary risk"),
        "over_trained": ("badge-red", "Recovery needed"),
    }
    return mapping.get(persona, ("badge-yellow", "Persona"))

def confidence_note(conf: float):
    if conf >= 85:
        return "High confidence"
    if conf >= 60:
        return "Medium confidence (check top-2)"
    return "Low confidence (results may be uncertain)"

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

    summary = f"Today: {steps:.0f} steps, stress {stress:.1f}, average HR {hr_avg:.1f} bpm."

    if pred_label == "overtrained_stressed":
        msg = (
            "You are pushing hard and recovery signals look high. "
            "Consider a lighter day, more sleep, hydration, and recovery activities."
        )
    elif pred_label == "high_workout":
        msg = "Strong activity level. Keep it up, and include recovery days to avoid burnout."
    elif pred_label == "healthy":
        msg = "Balanced profile. Your activity, stress, and heart rate look stable and sustainable."
    elif pred_label == "low_activity":
        msg = "Movement is a bit low today. A short walk (10–20 minutes) can already help a lot."
    else:  # Sedentary_stressed
        msg = "Activity is quite low. Start small: short walks, gentle movement, and build consistency."

    probs_dict = {cls: float(p) for cls, p in zip(class_names, probs)}
    return pred_label, confidence, probs_dict, summary, msg

# --------------------------
# Header
# --------------------------
st.markdown("# Lifestyle Persona Classifier (v2)")
st.markdown(
    "<div class='small-note'>Enter daily values to get a persona classification and simple health feedback.</div>",
    unsafe_allow_html=True
)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# --------------------------
# Layout: Inputs (left) + Results (right)
# --------------------------
left, right = st.columns([1.05, 1.2], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input")

    # CHANGED: steps from number_input -> slider so we can show colored bar
    steps = st.slider(
        "Steps per day",
        min_value=0,
        max_value=30000,
        value=8000,
        step=100,
        help="Total steps in one day."
    )
    colored_range_bar(
        value=steps,
        vmin=0,
        vmax=30000,
        segments=[
            (0, 3000, "#ef4444"),       # red
            (3000, 6000, "#f59e0b"),    # yellow
            (6000, 19000, "#22c55e"),   # green
            (19000, 30000, "#ef4444"),  # red
        ]
    )

    stress = st.slider(
        "Average stress level (0–100)",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
        help="Daily average stress score (0 is low, 100 is high)."
    )
    colored_range_bar(
        value=stress,
        vmin=0,
        vmax=100,
        segments=[
            (0, 40, "#22c55e"),    # green
            (40, 60, "#f59e0b"),   # yellow
            (60, 100, "#ef4444"),  # red (70 boundary is ok)
        ]
    )

    # CHANGED: HR range to exactly your desired max 140
    hr_avg = st.slider(
        "Average heart rate today (bpm)",
        min_value=60,
        max_value=140,
        value=95,
        step=1,
        help="Daily average heart rate (bpm)."
    )
    colored_range_bar(
        value=hr_avg,
        vmin=60,
        vmax=140,
        segments=[
            (60, 82, "#ef4444"),    # red
            (82, 94, "#f59e0b"),   # green
            (94, 130, "#22c55e"),  # yellow
            (130, 140, "#ef4444"),  # red
        ]
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    s_label, s_css, s_range = steps_status(steps)
    st_label, st_css, st_range = stress_status(stress)
    hr_label, hr_css, hr_range = hr_status(hr_avg)

    st.markdown("**Quick status**")
    st.markdown(
        f"- Steps: <span class='badge {s_css}'>{s_label}</span> <span class='small_note'>({s_range})</span><br>"
        f"- Stress: <span class='badge {st_css}'>{st_label}</span> <span class='small_note'>({st_range})</span><br>"
        f"- Heart rate: <span class='badge {hr_css}'>{hr_label}</span> <span class='small_note'>({hr_range})</span>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Classify", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Result")

    if run_btn:
        pred_label, confidence, probs_dict, summary, msg = classify_and_feedback(steps, stress, hr_avg)

        p_css, p_text = persona_badge(pred_label)
        conf_text = confidence_note(confidence)

        st.markdown(
            f"<p class='metric-title'>Predicted persona</p>"
            f"<p class='metric-value'>{pred_label} <span class='badge {p_css}'>{p_text}</span></p>"
            f"<p class='metric-sub'>Confidence: {confidence:.1f}% • {conf_text}</p>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown(f"**Summary**: {summary}")
        st.markdown(f"**Feedback**: {msg}")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### Probabilities")

        prob_table = pd.DataFrame({
            "persona": list(probs_dict.keys()),
            "probability": list(probs_dict.values()),
        }).sort_values("probability", ascending=False)

        for _, r in prob_table.iterrows():
            st.write(f"{r['persona']}: {r['probability']:.3f}")
            st.progress(min(max(float(r["probability"]), 0.0), 1.0))

        top2 = prob_table.head(2).values.tolist()
        st.caption(f"Top-2: {top2[0][0]} ({top2[0][1]:.3f}), {top2[1][0]} ({top2[1][1]:.3f})")

    else:
        st.markdown(
            "<div class='small-note'>Click <b>Classify</b> to see the prediction, feedback, and probabilities.</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
