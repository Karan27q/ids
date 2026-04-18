"""Utility helpers for the IDS project.

Implements small, self-contained utilities referenced in IMPROVEMENTS.md:
- render_alert_badge
- stream_simulation
- evaluate_models

These functions are written to be robust to missing optional dependencies
and provide informative exceptions when something isn't available.
"""
from typing import Any, Callable, Dict, Generator, Optional
import os
import time
import json

import numpy as np
import pandas as pd

SEVERITY_COLORS = {
    "Critical": "#dc3545",
    "High": "#fd7e14",
    "Medium": "#ffc107",
    "Low": "#28a745",
}


def render_alert_badge(severity: str) -> str:
    """Return an HTML badge for a severity string usable in Streamlit.

    Example output:
        '<span style="background:#dc3545;...">Critical</span>'
    """
    color = SEVERITY_COLORS.get(severity, "#6c757d")
    safe_text = str(severity)
    return f'<span style="background:{color};color:white;padding:4px 8px;border-radius:4px">{safe_text}</span>'


def stream_simulation(
    uploaded_file_or_df: Any,
    run_prediction: Optional[Callable[[Dict[str, Any]], Any]] = None,
    chunk_size: int = 10,
    sleep_time: Optional[float] = None,
) -> Generator[Any, None, None]:
    """Simulate a streaming ingestion from a CSV (or iterate a DataFrame).

    - If `uploaded_file_or_df` is a path or file-like, it is read with
      `pd.read_csv(..., chunksize=chunk_size)`.
    - If it is a `pd.DataFrame`, rows are iterated directly.
    - If `run_prediction` is provided it will be called with the row dict
      and the function will yield its return value. Otherwise the row dict
      is yielded.
    - If `sleep_time` (seconds) is provided, the function sleeps between
      yielded rows.
    """
    if isinstance(uploaded_file_or_df, str):
        reader = pd.read_csv(uploaded_file_or_df, chunksize=chunk_size)
        for chunk in reader:
            for _, row in chunk.iterrows():
                row_dict = row.to_dict()
                yield run_prediction(row_dict) if run_prediction else row_dict
                if sleep_time:
                    time.sleep(sleep_time)

    elif hasattr(uploaded_file_or_df, "read"):
        reader = pd.read_csv(uploaded_file_or_df, chunksize=chunk_size)
        for chunk in reader:
            for _, row in chunk.iterrows():
                row_dict = row.to_dict()
                yield run_prediction(row_dict) if run_prediction else row_dict
                if sleep_time:
                    time.sleep(sleep_time)

    elif isinstance(uploaded_file_or_df, pd.DataFrame):
        for _, row in uploaded_file_or_df.iterrows():
            row_dict = row.to_dict()
            yield run_prediction(row_dict) if run_prediction else row_dict
            if sleep_time:
                time.sleep(sleep_time)

    else:
        raise ValueError("uploaded_file_or_df must be a file path, file-like, or pandas.DataFrame")


def evaluate_models(models_dir: str, test_csv_path: str, feature_columns: Optional[list] = None,
                    label_column: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate available models on a test CSV and return metrics dictionary.

    This helper will attempt to load the following files from `models_dir` if
    present (in order):
      - `lstm_autoencoder.keras` or `lstm_autoencoder.h5` (Keras)
      - `attack_classifier.keras` or `attack_classifier.h5` (Keras)
      - `xgb_model.json` or `xgb_model.save` (joblib)
      - `scaler.save` (joblib)

    The function returns a dict with keys: `metrics`, `recon_errors` (if AE),
    `y_true`, `y_pred_proba` (if classifier), and `notes`.
    """
    notes = []
    results: Dict[str, Any] = {"metrics": {}, "notes": []}

    # load test data
    df = pd.read_csv(test_csv_path)

    # detect label column if not provided
    if label_column is None:
        for candidate in ("label", "Label", "result", "attack", "Attack"):
            if candidate in df.columns:
                label_column = candidate
                break

    y_true = None
    if label_column and label_column in df.columns:
        y_true = df[label_column].copy()

    # feature columns
    if feature_columns is None:
        # drop obvious non-feature cols
        drop_cols = {label_column} if label_column else set()
        feature_columns = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]

    X = df[feature_columns].values

    # attempt to load optional scaler
    scaler = None
    try:
        import joblib
        scaler_path = os.path.join(models_dir, "scaler.save")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)
            notes.append("Applied scaler from scaler.save")
    except Exception:
        notes.append("No scaler applied or joblib not available")

    # reshape for LSTM autoencoder (assume timesteps=1 if AE expects sequences)
    X_seq = X.reshape((X.shape[0], 1, X.shape[1]))

    # Try loading Keras
    ae = None
    clf = None
    try:
        from tensorflow.keras.models import load_model
        ae_path = None
        for name in ("lstm_autoencoder.keras", "lstm_autoencoder.h5"):
            p = os.path.join(models_dir, name)
            if os.path.exists(p):
                ae_path = p
                break
        if ae_path:
            ae = load_model(ae_path)
            notes.append(f"Loaded autoencoder: {os.path.basename(ae_path)}")

        clf_path = None
        for name in ("attack_classifier.keras", "attack_classifier.h5"):
            p = os.path.join(models_dir, name)
            if os.path.exists(p):
                clf_path = p
                break
        if clf_path:
            clf = load_model(clf_path)
            notes.append(f"Loaded classifier: {os.path.basename(clf_path)}")
    except Exception:
        notes.append("TensorFlow/Keras not available or model load failed")

    # fallback: try joblib XGBoost/sklearn classifier
    if clf is None:
        try:
            import joblib
            for name in ("xgb_model.save", "xgb_model.json", "attack_classifier.save"):
                p = os.path.join(models_dir, name)
                if os.path.exists(p):
                    clf = joblib.load(p)
                    notes.append(f"Loaded classifier via joblib: {name}")
                    break
        except Exception:
            notes.append("No joblib classifier available or joblib missing")

    # compute reconstruction errors if AE available
    recon_errors = None
    if ae is not None:
        try:
            preds = ae.predict(X_seq, verbose=0)
            recon_errors = np.mean(np.square(X_seq - preds), axis=(1, 2))
            results["recon_errors"] = recon_errors.tolist()
            notes.append("Computed reconstruction errors from AE")
        except Exception as e:
            notes.append(f"Failed to compute AE recon errors: {e}")

    # classifier predictions
    y_pred_proba = None
    y_pred = None
    if clf is not None:
        try:
            # keras classifier
            if hasattr(clf, "predict") and not hasattr(clf, "feature_names_in_"):
                # try to predict on non-sequence input
                try:
                    y_prob = clf.predict(X, verbose=0)
                except Exception:
                    # maybe clf expects sequences
                    y_prob = clf.predict(X_seq, verbose=0)
                y_pred_proba = np.asarray(y_prob)
                if y_pred_proba.ndim > 1:
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = (y_pred_proba >= 0.5).astype(int)
            else:
                # sklearn/xgboost
                if hasattr(clf, "predict_proba"):
                    y_pred_proba = clf.predict_proba(X)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = clf.predict(X)
            results["y_pred_proba"] = y_pred_proba.tolist() if y_pred_proba is not None else None
            results["y_pred"] = (y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred)
            notes.append("Computed classifier predictions")
        except Exception as e:
            notes.append(f"Failed to compute classifier predictions: {e}")

    # metrics
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        metrics = {}
        if y_true is not None:
            # if y_true is strings, attempt binary benign vs attack
            if y_pred is not None:
                # try to coerce y_true to numeric binary
                if y_true.dtype == object:
                    y_bin = (y_true.str.lower() != "benign").astype(int)
                    # determine pred bin
                    if y_pred_proba is not None:
                        # assume first column is benign if columns match
                        if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim > 1:
                            benign_idx = 0
                            attack_prob = 1 - y_pred_proba[:, benign_idx]
                            y_pred_bin = (attack_prob >= 0.5).astype(int)
                        else:
                            y_pred_bin = y_pred
                    else:
                        y_pred_bin = y_pred
                    metrics["accuracy"] = float(accuracy_score(y_bin, y_pred_bin))
                    metrics["f1"] = float(f1_score(y_bin, y_pred_bin))
                    metrics["precision"] = float(precision_score(y_bin, y_pred_bin))
                    metrics["recall"] = float(recall_score(y_bin, y_pred_bin))
                else:
                    # numeric labels
                    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                    metrics["f1"] = float(f1_score(y_true, y_pred, average="macro"))
            # add recon-based metrics if available
        results["metrics"] = metrics
    except Exception:
        notes.append("sklearn.metrics not available or metrics computation failed")

    results["notes"] = notes
    return results
