import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt

# ----------------------------
# Caching models for speed
# ----------------------------
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.save")
    autoencoder = load_model("lstm_autoencoder.keras")
    classifier = load_model("attack_classifier.keras")
    return scaler, autoencoder, classifier

scaler, autoencoder, classifier = load_assets()

# Get feature names from scaler
feature_names = scaler.feature_names_in_

# Shapes
n_features = autoencoder.input_shape[2]
num_classes = classifier.output_shape[1]

# TODO: Replace with your real class names
class_names = [f"Class {i}" for i in range(num_classes)]

# ----------------------------
# Session state
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙️ Settings")

# Load precomputed threshold if you have it
# Example: joblib.dump(threshold, "threshold.save") after training
try:
    auto_threshold = joblib.load("threshold.save")
except:
    auto_threshold = 0.05  # fallback

use_manual = st.sidebar.checkbox("Use Manual Threshold Override", value=False)

if use_manual:
    threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, float(auto_threshold), 0.001)
else:
    threshold = float(auto_threshold)

st.sidebar.write(f"**Active Threshold:** {threshold:.6f}")

# ----------------------------
# Main UI
# ----------------------------
st.title("🚨 Intrusion Detection System (IDS) Dashboard")
st.write(f"Expected number of features: **{n_features}**")

# Data source
data_source = st.selectbox("Data Source", ["Random (Simulated)", "Upload CSV"])

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with features", type=["csv"])

def get_features():
    if data_source == "Random (Simulated)":
        return np.random.rand(n_features) * 100

    elif data_source == "Upload CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != n_features:
            st.error(f"CSV must have exactly {n_features} features.")
            return np.random.rand(n_features) * 100
        return df.iloc[-1].values

    else:
        return np.random.rand(n_features) * 100

# ----------------------------
# Prediction function
# ----------------------------
def run_prediction():
    features = get_features()

    # Scale
    features_df = pd.DataFrame(features.reshape(1, -1), columns=feature_names)
    features_scaled = scaler.transform(features_df)

    # Reshape for LSTM AE
    features_seq = features_scaled.reshape(1, 1, -1)

    # Reconstruction
    pred = autoencoder.predict(features_seq, verbose=0)
    error = np.mean(np.square(features_seq - pred))

    # Decision
    if error > threshold:
        pred_class = classifier.predict(features_scaled, verbose=0)
        class_idx = np.argmax(pred_class, axis=1)[0]
        confidence = np.max(pred_class, axis=1)[0]
        prediction = class_names[class_idx]
        result = f"Anomaly: {prediction} (Conf: {confidence:.2f})"
        alert = True
    else:
        result = "Normal"
        alert = False

    # Store history
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.history.append({
        "time": timestamp,
        "error": float(error),
        "result": result,
        "alert": alert
    })

    return features, error, result, alert

# ----------------------------
# Controls
# ----------------------------
st.header("🔍 Live Prediction")

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict Now"):
        features, error, result, alert = run_prediction()

with col2:
    st.session_state.auto_mode = st.checkbox("Enable Auto Prediction (every 5 sec)", value=st.session_state.auto_mode)

# Auto refresh logic
if st.session_state.auto_mode:
    features, error, result, alert = run_prediction()
    time.sleep(5)
    st.rerun()

# ----------------------------
# Show latest result
# ----------------------------
if st.session_state.history:
    last = st.session_state.history[-1]

    st.subheader("📊 Current Prediction")
    st.write(f"Reconstruction Error: **{last['error']:.6f}**")
    st.write(f"Threshold: **{threshold:.6f}**")

    if last["alert"]:
        st.error(f"🚨 ALERT: {last['result']}")
        st.write("**Recommendation:** Investigate this traffic immediately.")
    else:
        st.success("✅ Normal Traffic")
        st.write("**Recommendation:** No action needed.")

# ----------------------------
# Alerts section
# ----------------------------
st.header("🚨 Alerts")

alerts = [h for h in st.session_state.history if h["alert"]]

if alerts:
    for alert in alerts[-5:]:
        st.warning(f"{alert['time']} | {alert['result']} | Error: {alert['error']:.4f}")
else:
    st.info("No alerts yet.")

# ----------------------------
# History + Plot
# ----------------------------
st.header("📜 Prediction History")

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True)

    st.subheader("📈 Reconstruction Error Over Time")
    fig, ax = plt.subplots()
    ax.plot(df_hist["time"], df_hist["error"], marker="o")
    ax.axhline(y=threshold, linestyle="--", label="Threshold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title("Error Trend")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("No predictions yet.")

st.write("⚠️ Note: This is a demo IDS dashboard. Integrate with real network traffic for production use.")
