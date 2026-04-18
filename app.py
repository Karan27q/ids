"""
SOC (Security Operations Center) Intrusion Detection Dashboard
Enterprise-grade real-time threat monitoring - LSTM Autoencoder + Classifier
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Dense as KerasDense
from tensorflow.keras.layers import TimeDistributed as KerasTimeDistributed
import json
import h5py

# Workaround for models saved with newer/quantized Keras layers: some configs
# include `quantization_config` which older deserializers don't accept.
# Provide a Dense subclass that strips that key from config during deserialization.
class DenseNoQuant(KerasDense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


# TimeDistributed may wrap layers whose configs include `quantization_config`.
# Ensure nested configs are stripped before deserialization.
class TimeDistributedNoQuant(KerasTimeDistributed):
    @classmethod
    def from_config(cls, config):
        def strip_quant(obj):
            if isinstance(obj, dict):
                obj.pop("quantization_config", None)
                for v in obj.values():
                    strip_quant(v)
            elif isinstance(obj, list):
                for item in obj:
                    strip_quant(item)
        strip_quant(config)
        return super().from_config(config)


def _strip_quant_from_h5(path):
    try:
        with h5py.File(path, "r+") as f:
            # Keras HDF5 stores the model config as a JSON string in attrs['model_config']
            if "model_config" in f.attrs:
                cfg = f.attrs["model_config"]
                if isinstance(cfg, (bytes, bytearray)):
                    cfg = cfg.decode("utf-8")
                data = json.loads(cfg)

                def _rec_strip(o):
                    if isinstance(o, dict):
                        o.pop("quantization_config", None)
                        for v in o.values():
                            _rec_strip(v)
                    elif isinstance(o, list):
                        for item in o:
                            _rec_strip(item)

                _rec_strip(data)
                f.attrs["model_config"] = json.dumps(data)
    except Exception:
        pass
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import uuid
from datetime import datetime, timedelta
import sys, os

# ensure local `src` package is importable (utils.py lives in ./src)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from utils import render_alert_badge, stream_simulation, evaluate_models
except Exception:
    # fall back silently if utils cannot be imported (will still run existing app)
    render_alert_badge = None
    stream_simulation = None
    evaluate_models = None

# =============================================================================
# DARK THEME & SOC STYLING
# =============================================================================
st.set_page_config(page_title="SOC Threat Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

SOC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');

:root {
    --bg-dark: #0d1117;
    --bg-card: #161b22;
    --bg-card-hover: #21262d;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --critical: #f85149;
    --high: #db6d28;
    --medium: #d29922;
    --low: #3fb950;
}

.stApp { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
.main .block-container { padding: 2rem 2rem 3rem; max-width: 100%; }

/* KPI Cards */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    font-family: 'Inter', sans-serif;
}
.kpi-card:hover { background: var(--bg-card-hover); }
.kpi-value { font-size: 1.75rem; font-weight: 700; color: var(--text); }
.kpi-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-trend { font-size: 0.75rem; margin-top: 4px; }

/* Severity badges */
.sev-critical { background: #f85149 !important; color: white !important; padding: 4px 10px; border-radius: 4px; font-weight: 600; }
.sev-high { background: #db6d28 !important; color: white !important; padding: 4px 10px; border-radius: 4px; font-weight: 600; }
.sev-medium { background: #d29922 !important; color: #0d1117 !important; padding: 4px 10px; border-radius: 4px; font-weight: 600; }
.sev-low { background: #3fb950 !important; color: white !important; padding: 4px 10px; border-radius: 4px; font-weight: 600; }

/* Critical alert pulse */
@keyframes pulse-critical {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.6); }
    50% { opacity: 0.95; box-shadow: 0 0 20px 4px rgba(248, 81, 73, 0.4); }
}
.pulse-critical { animation: pulse-critical 1.5s ease-in-out infinite; }

/* Panel headers */
.panel-header {
    font-size: 1rem; font-weight: 600; color: var(--text);
    border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; margin-bottom: 1rem;
}
div[data-testid="stMetric"] { background: var(--bg-card); padding: 1rem; border-radius: 8px; border: 1px solid var(--border); }
div[data-testid="stMetric"] label { color: var(--text-muted) !important; }
div[data-testid="stMetric"] div { color: var(--text) !important; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(SOC_CSS, unsafe_allow_html=True)

# =============================================================================
# CONFIG & HELPERS
# =============================================================================
SEVERITY_COLORS = {"Critical": "#f85149", "High": "#db6d28", "Medium": "#d29922", "Low": "#3fb950"}
PLOTLY_TEMPLATE = "plotly_dark"

def hybrid_risk_score(recon_err, attack_proba, thresh, w_ae=0.4, w_clf=0.6):
    norm_err = min(recon_err / max(thresh * 2, 1e-6), 1.0)
    return w_ae * norm_err + w_clf * attack_proba

def severity_level(score):
    if score >= 0.8: return "Critical"
    if score >= 0.6: return "High"
    if score >= 0.4: return "Medium"
    return "Low"

def gen_sim_ip():
    return f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"

def _normalize_col(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def load_csv_smart(uploaded_file_or_df, expected_features):
    """Load CSV and align columns to model features. Matches by name (strip, case-insensitive). Pass DataFrame to reuse one read."""
    if isinstance(uploaded_file_or_df, pd.DataFrame):
        df = uploaded_file_or_df
    else:
        df = pd.read_csv(uploaded_file_or_df)
    df.columns = df.columns.str.strip()
    csv_cols = {_normalize_col(c): c for c in df.columns}
    out = {}
    for feat in expected_features:
        key = _normalize_col(feat)
        if feat in df.columns:
            out[feat] = df[feat].values
        elif key in csv_cols:
            out[feat] = df[csv_cols[key]].values
        else:
            out[feat] = np.zeros(len(df))
    return pd.DataFrame(out, columns=expected_features)

def _get_label_col(df):
    """Return Label column name if present (any common name)."""
    for c in df.columns:
        n = c.strip().lower()
        if n in ("label", "attack type", "attack_type", "labels", "class", "target"):
            return c
    return None

# =============================================================================
# LOAD MODELS
# =============================================================================
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.save")
    # Strip `quantization_config` keys from saved model JSON (HDF5 .keras files)
    # before deserializing so older TF/Keras runtimes can load them.
    for p in ("lstm_autoencoder.h5", "attack_classifier.h5"):
        _strip_quant_from_h5(p)

    custom_objs = {"Dense": DenseNoQuant, "TimeDistributed": TimeDistributedNoQuant}
    autoencoder = load_model("lstm_autoencoder.h5", custom_objects=custom_objs, compile=False)
    classifier = load_model("attack_classifier.h5", custom_objects=custom_objs, compile=False)
    return scaler, autoencoder, classifier

scaler, autoencoder, classifier = load_assets()
feature_names = scaler.feature_names_in_
n_features = autoencoder.input_shape[2]
num_classes = classifier.output_shape[1]
try:
    class_names = joblib.load("class_names.save")
except Exception:
    class_names = [f"Class {i}" for i in range(num_classes)]

# =============================================================================
# SESSION STATE
# =============================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False
if "incidents" not in st.session_state:
    st.session_state.incidents = {}
if "incident_counter" not in st.session_state:
    st.session_state.incident_counter = 0
if "last_processed_csv_name" not in st.session_state:
    st.session_state.last_processed_csv_name = None
if "last_csv_result" not in st.session_state:
    st.session_state.last_csv_result = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("🛡️ SOC Control")
    st.markdown("---")
    try:
        threshold = float(joblib.load("threshold.save"))
    except Exception:
        threshold = 0.05
    st.caption(f"Anomaly threshold (from model): {threshold:.4f}")

    data_source = st.selectbox("Data source", ["Random (Simulated)", "Upload CSV"])
    uploaded_file = st.file_uploader("CSV upload", type=["csv"]) if data_source == "Upload CSV" else None

    st.markdown("---")
    severity_filter = st.multiselect("Filter severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
    search_ip = st.text_input("Search by IP", placeholder="e.g. 10.0.0.1")
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=st.session_state.auto_mode)
    st.session_state.auto_mode = auto_refresh
    st.markdown("---")
    st.subheader("Model Evaluation")
    eval_file = st.file_uploader("Evaluation CSV", type=["csv"], key="eval_csv")
    if st.button("Run Model Evaluation"):
        if evaluate_models is None:
            st.warning("Model evaluation utility not available.")
        elif eval_file is None:
            st.warning("Upload a CSV file to run evaluation.")
        else:
            with st.spinner("Evaluating models..."):
                try:
                    # Use current app directory for model files (models live alongside app.py)
                    models_dir = os.path.dirname(__file__)
                    # Read uploaded CSV to detect which columns are present
                    try:
                        eval_df = pd.read_csv(eval_file)
                        # reset file pointer for downstream readers
                        try:
                            eval_file.seek(0)
                        except Exception:
                            pass
                    except Exception:
                        eval_df = None

                    fcols = None
                    missing = []
                    if eval_df is not None and 'feature_names' in globals():
                        # Map normalized column names to actual csv columns
                        csv_cols = {_normalize_col(c): c for c in eval_df.columns}
                        matched = []
                        for feat in feature_names.tolist():
                            if feat in eval_df.columns:
                                matched.append(feat)
                            else:
                                key = _normalize_col(feat)
                                if key in csv_cols:
                                    matched.append(csv_cols[key])
                                else:
                                    missing.append(feat)
                        if matched:
                            fcols = matched
                    # If no matched feature columns, let evaluate_models auto-detect numeric features
                    if missing:
                        st.warning(f"Evaluation: {len(missing)} expected features not found in CSV (showing first 10): {missing[:10]}")

                    results = evaluate_models(models_dir, eval_file, feature_columns=fcols)
                    st.success("Evaluation complete")
                    st.session_state.last_evaluation = results
                    # Show key metrics prominently and allow download
                    metrics = results.get("metrics", {})
                    if metrics:
                        st.write("**Evaluation Metrics**")
                        for k, v in metrics.items():
                            st.write(f"- {k}: {v}")
                    else:
                        st.info("No labeled metrics produced; see raw results below.")
                    st.json(results)
                    st.download_button("Download Evaluation Results (JSON)", data=json.dumps(results, default=list), file_name="evaluation_results.json", mime="application/json")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

def get_features():
    """Returns features; in Random mode sometimes (features, severity) to force Critical/High alerts."""
    if data_source == "Random (Simulated)":
        r = random.random()
        # Generate some Critical/High simulated alerts so KPIs and feed show variety
        if r < 0.10:
            return (np.random.rand(n_features) * 100, "Critical")
        if r < 0.25:
            return (np.random.rand(n_features) * 100, "High")
        return np.random.rand(n_features) * 100
    elif data_source == "Upload CSV" and uploaded_file:
        try:
            df = load_csv_smart(uploaded_file, feature_names.tolist())
            return df.iloc[-1].values
        except Exception:
            return np.random.rand(n_features) * 100
    return np.random.rand(n_features) * 100

# =============================================================================
# THREAT DETECTION ENGINE
# =============================================================================
def run_prediction(features=None, label_override=None, force_severity=None):
    """Run detection on one sample. If features is None, use get_features(). label_override/force_severity force alert (CSV or simulated)."""
    if features is None:
        raw = get_features()
        if isinstance(raw, tuple):
            features, force_severity = raw[0], raw[1]
        else:
            features = raw
    features = np.asarray(features).reshape(-1)
    if len(features) != n_features:
        return None
    features_df = pd.DataFrame(features.reshape(1, -1), columns=feature_names)
    features_scaled = scaler.transform(features_df)
    features_seq = features_scaled.reshape(1, 1, -1)

    pred_ae = autoencoder.predict(features_seq, verbose=0)
    error = np.mean(np.square(features_seq - pred_ae))

    pred_class = classifier.predict(features_scaled, verbose=0)
    class_idx = int(np.argmax(pred_class, axis=1)[0])
    confidence = float(np.max(pred_class, axis=1)[0])
    prediction = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"

    benign_idx = class_names.index("Benign") if "Benign" in class_names else 0
    attack_proba = 1.0 - pred_class[0][benign_idx] if benign_idx < len(pred_class[0]) else confidence
    # Use only trained threshold and fixed bars (no manual tuning)
    is_anomaly_ae = error > threshold
    is_anomaly_clf = class_idx != benign_idx and confidence > 0.2
    is_anomaly_prob = attack_proba > 0.35
    alert = is_anomaly_ae or is_anomaly_clf or is_anomaly_prob
    # CSV Label override: if data says it's an attack, always alert and set severity so Critical/High KPIs update
    if label_override is not None and str(label_override).strip():
        lbl = str(label_override).strip().lower()
        if lbl != "benign":
            alert = True
            prediction = str(label_override).strip()

    risk_score = hybrid_risk_score(error, attack_proba, threshold)
    severity = severity_level(risk_score)
    # Severity from CSV Label when present so critical/high alerts update (DDoS, Bot, etc.)
    if alert and label_override is not None and str(label_override).strip():
        lbl = str(label_override).strip().lower()
        if any(x in lbl for x in ("ddos", "bot", "botnet", "infiltration")):
            severity = "Critical"
        elif any(x in lbl for x in ("bruteforce", "dos", "webattack", "portscan", "sql", "xss")):
            severity = "High"
        else:
            severity = "Medium"
    # Random (Simulated) mode: sometimes force Critical/High so KPIs and feed show variety
    if force_severity:
        alert = True
        severity = force_severity
        display_attack_type = f"Simulated ({force_severity})"
    # Never show "Benign" as attack type for an alert: if we alerted on prob/recon but classifier says Benign, show suspected attack
    elif alert and (prediction == "Benign" or class_idx == benign_idx):
        display_attack_type = f"Suspected Attack ({attack_proba*100:.0f}% prob)"
        # Bump severity when we're alerting on probability/recon so it's not all Low
        if attack_proba >= 0.6 or error > threshold:
            severity = "High" if attack_proba >= 0.7 else "Medium"
    else:
        display_attack_type = prediction if alert else "Normal"

    entry = {
        "time": time.strftime("%H:%M:%S"),
        "datetime": datetime.now().isoformat(),
        "error": float(error),
        "alert": alert,
        "risk_score": risk_score,
        "severity": severity,
        "attack_type": display_attack_type,
        "confidence": confidence,
        "source_ip": gen_sim_ip(),
        "dest_ip": gen_sim_ip(),
        "status": "New" if alert else "Normal",
        "incident_id": None,
    }

    if alert:
        st.session_state.incident_counter += 1
        inc_id = f"INC-{st.session_state.incident_counter:05d}"
        entry["incident_id"] = inc_id
        entry["status"] = "New"
        sla_due = (datetime.now() + timedelta(minutes=15)).strftime("%H:%M")
        st.session_state.incidents[inc_id] = {
            "id": inc_id, "severity": severity, "attack_type": display_attack_type,
            "assigned": "Unassigned", "status": "New", "sla_due": sla_due,
            "time": entry["time"], "source_ip": entry["source_ip"], "dest_ip": entry["dest_ip"],
            "risk_score": risk_score, "confidence": confidence
        }

    st.session_state.history.append(entry)
    return entry

# =============================================================================
# CSV UPLOAD: NOTIFICATION + AUTO-SCAN (UP TO 500 LOGS)
# =============================================================================
if data_source == "Upload CSV" and uploaded_file is not None:
    file_name = uploaded_file.name
    is_new_upload = file_name != st.session_state.last_processed_csv_name

    if is_new_upload:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()
        # Align CSV columns to model features (handles name/order/case differences)
        df_csv = load_csv_smart(df_raw, feature_names.tolist())
        label_col = _get_label_col(df_raw)
        has_label = label_col is not None
        n_rows = len(df_csv)
        max_rows = 500
        if n_rows > max_rows:
            idx = df_raw.sample(n=max_rows, random_state=42).index
            df_csv = df_csv.loc[idx].reset_index(drop=True)
            df_raw = df_raw.loc[idx].reset_index(drop=True)
            n_rows = max_rows

        try:
            st.toast(f"CSV uploaded: {file_name}. Analyzing {n_rows} log entries...", icon="📂")
        except Exception:
            pass

        upload_placeholder = st.empty()
        with upload_placeholder.container():
            st.success(f"✅ **Log file received:** `{file_name}` — Running threat detection on **{n_rows}** entries (trained threshold only).")
            if has_label:
                st.caption("📌 CSV has a Label column: rows labeled as attack will always appear as alerts.")
            progress_bar = st.progress(0, text="Scanning logs...")
            alerts_from_csv = 0
            processed_rows = []
            for i in range(n_rows):
                try:
                    feats = df_csv.iloc[i].values.astype(float)
                    if len(feats) == n_features:
                        csv_label = df_raw.iloc[i][label_col] if has_label else None
                        entry = run_prediction(features=feats, label_override=csv_label)
                        if entry:
                            # track per-row processed entries for later review/download
                            processed_rows.append(entry)
                            if entry.get("alert"):
                                alerts_from_csv += 1
                except Exception:
                    pass
                progress_bar.progress((i + 1) / n_rows, text=f"Scanning... {i+1}/{n_rows} | Threats: {alerts_from_csv}")
            progress_bar.empty()
            st.session_state.last_processed_csv_name = file_name
            st.session_state.last_csv_result = {"file": file_name, "rows": n_rows, "alerts": alerts_from_csv}
            # save processed rows (as list of dicts) for UI preview and download
            st.session_state.last_csv_entries = processed_rows
        try:
            st.toast(f"Analysis complete. {alerts_from_csv} threat(s) from {n_rows} entries.", icon="🛡️")
        except Exception:
            pass
        st.rerun()

# =============================================================================
# MAIN LAYOUT
# =============================================================================
# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🛡️ SOC Threat Operations Center")
    st.caption("Real-time intrusion detection • LSTM Autoencoder + ML Classifier • CICIDS")
with col_h2:
    if st.button("▶️ Scan Now", type="primary"):
        run_prediction()
        st.rerun()

# Last CSV analysis notification (persistent)
if st.session_state.get("last_csv_result"):
    r = st.session_state.last_csv_result
    st.info(f"📁 **Last log file analyzed:** `{r['file']}` — **{r['alerts']}** threat(s) detected from **{r['rows']}** log entries. Review the Live Threat Feed and Incident Queue below.")
    # Show processed CSV preview and allow download
    if st.session_state.get("last_csv_entries"):
        try:
            df_last = pd.DataFrame(st.session_state.last_csv_entries)
            with st.expander("Preview processed CSV results (last upload)"):
                st.dataframe(df_last.head(50), use_container_width=True)
                csv_data = df_last.to_csv(index=False)
                st.download_button("Download processed results CSV", data=csv_data, file_name=f"processed_{r['file']}", mime="text/csv")
                # attempt to persist a copy alongside the app (best-effort)
                try:
                    out_path = os.path.join(os.path.dirname(__file__), f"processed_{r['file']}")
                    df_last.to_csv(out_path, index=False)
                except Exception:
                    pass
        except Exception:
            pass

# Critical alert banner
alerts = [h for h in st.session_state.history if h["alert"]]
critical_alerts = [a for a in alerts if a.get("severity") == "Critical"]
if critical_alerts:
    # If `render_alert_badge` is available use it for a compact badge, keep pulse banner for visibility
    try:
        badge_html = render_alert_badge("Critical") if render_alert_badge else ""
    except Exception:
        badge_html = ""
    st.markdown(
        f'<div class="pulse-critical" style="background:#f85149;color:white;padding:12px 20px;border-radius:8px;margin-bottom:1.5rem;font-weight:600;">'
        f'⚠️ CRITICAL THREAT DETECTED — {len(critical_alerts)} critical alert(s) require immediate attention {badge_html}</div>',
        unsafe_allow_html=True
    )

# =============================================================================
# 1. KPI SUMMARY ROW
# =============================================================================
total = len(st.session_state.history)
alert_count = len(alerts)
critical_count = len([a for a in alerts if a.get("severity") == "Critical"])
high_count = len([a for a in alerts if a.get("severity") == "High"])
normal_pct = ((total - alert_count) / total * 100) if total else 100
avg_score = np.mean([h.get("risk_score", 0) for h in st.session_state.history]) if st.session_state.history else 0

# Simple accuracy proxy: among alerts, what % had high confidence
acc_proxy = np.mean([h.get("confidence", 0) for h in alerts]) * 100 if alerts else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, "Active Alerts", str(alert_count), "📊", "↓" if alert_count < 5 else "↑"),
    (k2, "Critical", str(critical_count), "🔴", ""),
    (k3, "High Severity", str(high_count), "🟠", ""),
    (k4, "Normal Traffic %", f"{normal_pct:.1f}%", "🟢", "↑" if normal_pct > 80 else "↓"),
    (k5, "Model Confidence %", f"{acc_proxy:.0f}%", "📈", ""),
    (k6, "Avg Anomaly Score", f"{avg_score:.0%}", "📉", ""),
]
for col, label, value, icon, trend in kpis:
    with col:
        st.metric(label, f"{icon} {value}", trend if trend else None)

st.markdown("---")

# =============================================================================
# 2. LIVE THREAT FEED
# =============================================================================
st.subheader("📡 Live Threat Feed")
filtered_alerts = [a for a in alerts if a.get("severity") in severity_filter]
if search_ip:
    filtered_alerts = [a for a in filtered_alerts if search_ip in a.get("source_ip", "") or search_ip in a.get("dest_ip", "")]

SEV_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

if filtered_alerts:
    df_feed = pd.DataFrame(filtered_alerts[-50:])
    df_feed["Severity"] = df_feed["severity"].map(lambda s: f"{SEV_EMOJI.get(s, '⚪')} {s}")
    df_feed = df_feed[["time", "source_ip", "dest_ip", "attack_type", "Severity", "error", "confidence", "status"]]
    df_feed.columns = ["Timestamp", "Source IP", "Dest IP", "Attack Type", "Severity", "Anomaly Score", "Confidence %", "Status"]
    df_feed["Confidence %"] = (df_feed["Confidence %"] * 100).round(1).astype(str) + "%"
    df_feed["Anomaly Score"] = df_feed["Anomaly Score"].round(6)
    st.dataframe(df_feed, use_container_width=True, hide_index=True)

    # Action buttons row
    c_ack, c_esc, c_block, c_exp = st.columns(4)
    with c_ack:
        if st.button("Acknowledge Selected"):
            st.info("Acknowledge feature: integrate with incident workflow.")
    with c_esc:
        if st.button("Escalate"):
            st.info("Escalate: assign to L2 analyst.")
    with c_block:
        if st.button("Block IP"):
            st.info("Block IP: add to blocklist.")
    with c_exp:
        csv_data = pd.DataFrame(filtered_alerts).to_csv(index=False)
        st.download_button("Export CSV", data=csv_data, file_name="threat_feed_export.csv", mime="text/csv")
else:
    st.info("No threats detected. Threat feed will populate as traffic is analyzed.")

st.markdown("---")

# =============================================================================
# 3. REAL-TIME VISUALIZATION PANEL
# =============================================================================
st.subheader("📊 Real-Time Threat Intelligence")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    df["idx"] = range(len(df))

    v1, v2 = st.columns(2)
    with v1:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df["idx"], y=df["error"], mode="lines", name="Anomaly Score", line=dict(color="#58a6ff")))
        fig_line.add_hline(y=threshold, line_dash="dash", line_color="#f85149")
        fig_line.update_layout(title="Live Anomaly Score", template=PLOTLY_TEMPLATE, height=280, margin=dict(l=40, r=40))
        st.plotly_chart(fig_line, use_container_width=True)

    with v2:
        if alerts:
            sev_counts = pd.Series([a.get("severity", "Low") for a in alerts]).value_counts()
            colors = [SEVERITY_COLORS.get(s, "#8b949e") for s in sev_counts.index]
            fig_pie = go.Figure(data=[go.Pie(labels=sev_counts.index, values=sev_counts.values, marker_colors=colors, hole=0.5)])
            fig_pie.update_layout(title="Severity Distribution", template=PLOTLY_TEMPLATE, height=280, margin=dict(l=40, r=40))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No severity distribution (no alerts yet).")

    v3, v4 = st.columns(2)
    with v3:
        if alerts:
            df_alert = df[df["alert"]]
            attack_counts = df_alert["attack_type"].value_counts().head(10)
            fig_bar = px.bar(x=attack_counts.values, y=attack_counts.index, orientation="h", title="Alerts by Attack Type")
            fig_bar.update_layout(template=PLOTLY_TEMPLATE, height=280, margin=dict(l=40, r=40), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No attack types to display.")

    with v4:
        df["bin"] = (df["idx"] // 5) * 5
        vol = df.groupby("bin").size()
        fig_vol = go.Figure(go.Bar(x=vol.index, y=vol.values, marker_color="#3fb950"))
        fig_vol.update_layout(title="Traffic Volume (5-sample bins)", template=PLOTLY_TEMPLATE, height=280, margin=dict(l=40, r=40))
        st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4. ATTACK INTELLIGENCE + 5. MODEL INTELLIGENCE
# =============================================================================
col_attack, col_model = st.columns(2)

with col_attack:
    st.subheader("🎯 Attack Intelligence")
    if alerts:
        df_a = pd.DataFrame(alerts)
        top_ips = df_a["source_ip"].value_counts().head(5)
        top_attacks = df_a["attack_type"].value_counts().head(5)
        st.write("**Top Attacking IPs**")
        st.dataframe(top_ips.reset_index().rename(columns={"index": "IP", "source_ip": "Count"}), hide_index=True, use_container_width=True)
        st.write("**Top Attack Types**")
        st.dataframe(top_attacks.reset_index().rename(columns={"index": "Type", "attack_type": "Count"}), hide_index=True, use_container_width=True)
        # Simulated protocol/port distribution for demo
        protocols = pd.Series({"TCP": 62, "UDP": 28, "ICMP": 10})
        fig_proto = go.Figure(data=[go.Pie(labels=protocols.index, values=protocols.values, hole=0.5, marker_colors=["#58a6ff", "#3fb950", "#d29922"])])
        fig_proto.update_layout(title="Protocol Distribution", template=PLOTLY_TEMPLATE, height=200, margin=dict(t=40))
        st.plotly_chart(fig_proto, use_container_width=True)
    else:
        st.info("No attack intelligence data yet.")

with col_model:
    st.subheader("🧠 Model Intelligence")
    if st.session_state.history:
        last = st.session_state.history[-1]
        err = last.get("error", 0)
        rs = last.get("risk_score", 0)
        pred_class = last.get("attack_type", "Normal")
        conf = last.get("confidence", 0) * 100
        fp_rate = 0  # would need labeled data
        st.metric("Current Anomaly Score", f"{err:.4f}")
        st.metric("Threshold", f"{threshold:.4f}")
        st.progress(min(rs, 1.0))
        st.caption("Risk score vs threshold")
        st.metric("Predicted Class", pred_class)
        st.metric("Model Confidence", f"{conf:.1f}%")
    else:
        st.info("Run a scan to see model intelligence.")

st.markdown("---")

# =============================================================================
# 6. INCIDENT RESPONSE QUEUE
# =============================================================================
st.subheader("📋 Incident Response Queue")

if st.session_state.incidents:
    inc_df = pd.DataFrame(st.session_state.incidents.values())
    inc_df = inc_df[["id", "severity", "attack_type", "assigned", "status", "sla_due", "time"]]
    inc_df.columns = ["Incident ID", "Severity", "Attack Type", "Assigned", "Status", "SLA Due", "Detected"]
    st.dataframe(inc_df, use_container_width=True, hide_index=True)
else:
    st.info("No incidents in queue.")

st.markdown("---")

# =============================================================================
# THREAT LOG ARCHIVE
# =============================================================================
with st.expander("📜 Threat Log Archive (Full History)"):
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    else:
        st.info("No log entries yet.")

# Auto-refresh
if st.session_state.auto_mode:
    run_prediction()
    time.sleep(5)
    st.rerun()
