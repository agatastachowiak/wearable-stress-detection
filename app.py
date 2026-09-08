import streamlit as st
import numpy as np
import pickle
import json
from scipy.signal import find_peaks

# ---- Page setup ----
st.set_page_config(page_title="Wearable Stress Detector", page_icon="💓")
st.title("Wearable Stress & Affect Detector")
st.write(
    "This app classifies physiological state (baseline, stress, or amusement) "
    "from wearable sensor data (EDA, heart rate/BVP, skin temperature, motion), "
    "using a model trained on the WESAD public dataset. Select an example below, "
    "or upload your own signal window, to see a live prediction."
)

# ---- Load model, scaler, and sample data (cached so it only loads once) ----
@st.cache_resource
def load_model():
    with open('stress_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_samples():
    with open('sample_windows.json', 'r') as f:
        return json.load(f)

model, scaler = load_model()
samples = load_samples()

# ---- Feature extraction (mirrors exactly what was done in training) ----
def extract_heart_features(bvp_win, fs_bvp=64):
    peaks, _ = find_peaks(np.array(bvp_win), distance=fs_bvp * 0.4)
    if len(peaks) < 3:
        return np.nan, np.nan
    ibi_sec = np.diff(peaks) / fs_bvp
    heart_rate = 60 / ibi_sec.mean()
    hrv = ibi_sec.std()
    return heart_rate, hrv

def compute_features(window):
    eda = np.array(window['eda'])
    bvp = np.array(window['bvp'])
    temp = np.array(window['temp'])
    acc = np.array(window['acc'])

    acc_magnitude = np.sqrt((acc ** 2).sum(axis=1))
    heart_rate, hrv = extract_heart_features(bvp)

    if np.isnan(heart_rate):
        return None  # not enough clean heartbeat data to make a prediction

    return np.array([[
        eda.mean(), eda.std(), eda.max() - eda.min(),
        heart_rate, hrv,
        temp.mean(), temp.std(),
        acc_magnitude.mean(), acc_magnitude.std(),
    ]])

# ---- UI: choose input method ----
st.subheader("1. Choose a signal window")
option = st.selectbox("Select an example, or choose 'Upload my own':",
                       list(samples.keys()) + ["Upload my own (JSON)"])

window = None
if option == "Upload my own (JSON)":
    uploaded = st.file_uploader(
        "Upload a JSON file with 'eda', 'bvp', 'temp', 'acc' arrays (60-second window)",
        type="json"
    )
    if uploaded:
        window = json.load(uploaded)
else:
    window = samples[option]

# ---- Run prediction ----
if window is not None:
    st.subheader("2. Signal preview")
    st.line_chart(window['eda'])
    st.caption("EDA (skin conductance) signal for this window")

    features = compute_features(window)

    if features is None:
        st.warning("Couldn't detect enough heartbeats in this window to make a reliable prediction.")
    else:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        label_map = {1: "Baseline (calm)", 2: "Stress", 3: "Amusement"}
        st.subheader("3. Prediction")
        st.metric("Predicted state", label_map[prediction])

        st.write("Confidence breakdown:")
        for class_id, prob in zip(model.classes_, probabilities):
            st.write(f"{label_map[class_id]}: {prob:.1%}")
            st.progress(float(prob))

st.divider()
st.caption(
    "Model: Linear SVM trained on 520 windows across 15 subjects from the WESAD "
    "dataset (Schmidt et al., 2018). Features: EDA statistics, heart rate & "
    "variability (derived from BVP), skin temperature, and motion. "
    "Mean balanced accuracy: 60.6% (chance level: 33.3%)."
)
