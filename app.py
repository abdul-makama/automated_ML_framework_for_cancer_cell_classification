from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pycaret.classification import load_model


MODEL_BASENAME = "cancer_classification_pipeline"
MODEL_FILE = Path(f"{MODEL_BASENAME}.pkl")

# ---------------------------------------------------------------------------
# Feature metadata: (min, max, default, step)
# Ranges are derived from the Wisconsin Breast Cancer Dataset statistics.
# ---------------------------------------------------------------------------
FEATURE_RANGES: dict[str, Tuple[float, float, float, float]] = {
    # ── Mean features (suffix 1) ────────────────────────────────────────────
    "radius1":            (6.0,   30.0,  14.0,  0.1),
    "texture1":           (9.0,   40.0,  19.0,  0.1),
    "perimeter1":         (40.0,  200.0, 92.0,  0.5),
    "area1":              (140.0, 2510.0,654.0, 5.0),
    "smoothness1":        (0.05,  0.17,  0.096, 0.001),
    "compactness1":       (0.02,  0.35,  0.104, 0.001),
    "concavity1":         (0.0,   0.43,  0.089, 0.001),
    "concave_points1":    (0.0,   0.20,  0.049, 0.001),
    "symmetry1":          (0.10,  0.30,  0.181, 0.001),
    "fractal_dimension1": (0.05,  0.10,  0.063, 0.001),
    # ── SE features (suffix 2) ─────────────────────────────────────────────
    "radius2":            (0.1,   3.0,   0.405, 0.01),
    "texture2":           (0.3,   5.0,   1.22,  0.01),
    "perimeter2":         (0.5,   22.0,  2.87,  0.05),
    "area2":              (6.0,   542.0, 40.3,  0.5),
    "smoothness2":        (0.001, 0.032, 0.007, 0.0005),
    "compactness2":       (0.002, 0.135, 0.025, 0.001),
    "concavity2":         (0.0,   0.40,  0.032, 0.001),
    "concave_points2":    (0.0,   0.053, 0.012, 0.001),
    "symmetry2":          (0.007, 0.079, 0.021, 0.001),
    "fractal_dimension2": (0.0008,0.030, 0.004, 0.0005),
    # ── Worst features (suffix 3) ───────────────────────────────────────────
    "radius3":            (7.0,   38.0,  16.3,  0.1),
    "texture3":           (12.0,  50.0,  25.7,  0.1),
    "perimeter3":         (50.0,  252.0, 107.0, 0.5),
    "area3":              (185.0, 4254.0,881.0, 5.0),
    "smoothness3":        (0.07,  0.23,  0.132, 0.001),
    "compactness3":       (0.027, 1.06,  0.255, 0.005),
    "concavity3":         (0.0,   1.25,  0.272, 0.005),
    "concave_points3":    (0.0,   0.291, 0.115, 0.001),
    "symmetry3":          (0.15,  0.66,  0.290, 0.001),
    "fractal_dimension3": (0.055, 0.21,  0.084, 0.001),
}

DEFAULT_FEATURES: List[str] = list(FEATURE_RANGES.keys())

# Labels returned by model.predict() — classes_=['B','M'] → index 0,1
CLASS_LABELS = {0: "Benign (B)", 1: "Malignant (M)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_expected_features(model) -> List[str]:
    """Return feature names expected by the fitted estimator."""
    try:
        est = model.named_steps["actual_estimator"]
        if hasattr(est, "feature_names_in_"):
            return list(est.feature_names_in_)
    except Exception:
        pass
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return DEFAULT_FEATURES


def decode_label(raw) -> str:
    """Map raw prediction string/int to a human-readable label."""
    s = str(raw).strip().upper()
    if s in ("M", "1"):
        return "🔴 Malignant (M)"
    if s in ("B", "0"):
        return "🟢 Benign (B)"
    return str(raw)


@st.cache_resource
def get_model():
    return load_model(MODEL_BASENAME)


def run_prediction_single(model, feature_df: pd.DataFrame):
    """
    Run prediction directly via the sklearn pipeline to avoid
    PyCaret's predict_model inverse-label bug ('unseen labels: 0.0').

    Returns (label_str, confidence_float).
    """
    label_series = model.predict(feature_df)          # returns 'B' or 'M'
    proba_arr = model.predict_proba(feature_df)       # shape (n, 2)

    label = str(label_series.iloc[0]).strip().upper()
    # classes_ = [0,1] → [B,M]; confidence for the predicted class
    classes = model.named_steps["actual_estimator"].classes_  # [0, 1]
    # Map predicted letter back to index
    letter_to_idx = {"B": 0, "M": 1}
    pred_idx = letter_to_idx.get(label, 0)
    confidence = float(proba_arr[0][pred_idx])
    return label, confidence


def run_batch_prediction(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Batch predict; returns df with prediction_label and confidence columns."""
    labels = model.predict(feature_df)
    probas = model.predict_proba(feature_df)
    out = feature_df.copy()
    out["prediction_label"] = labels.values
    out["prediction_readable"] = out["prediction_label"].apply(decode_label)
    letter_to_idx = {"B": 0, "M": 1}
    out["confidence"] = [
        float(probas[i][letter_to_idx.get(str(labels.iloc[i]).strip().upper(), 0)])
        for i in range(len(labels))
    ]
    return out


# ---------------------------------------------------------------------------
# Page config & header
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cancer Cell Classifier",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stSlider > label { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Cancer Cell Classification")
st.caption("Wisconsin Breast Cancer Dataset · PyCaret + Logistic Regression")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

if not MODEL_FILE.exists():
    st.error(
        "Model file `cancer_classification_pipeline.pkl` not found. "
        "Run the final save cell in `train.ipynb` to generate it."
    )
    st.stop()

try:
    model = get_model()
except Exception as exc:
    st.error(f"Failed to load model pipeline: {exc}")
    st.stop()

expected_features = extract_expected_features(model)

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------

mode = st.radio(
    "Prediction mode",
    ["Single prediction", "Batch CSV"],
    horizontal=True,
)

# ===========================================================================
# SINGLE PREDICTION
# ===========================================================================
if mode == "Single prediction":
    st.subheader("Single Sample Prediction")
    st.write("Adjust the sliders below, then press **Predict**.")

    # Group features into three groups (mean / SE / worst)
    mean_feats  = [f for f in expected_features if f.endswith("1")]
    se_feats    = [f for f in expected_features if f.endswith("2")]
    worst_feats = [f for f in expected_features if f.endswith("3")]

    input_row: dict[str, float] = {}

    def make_sliders(features: List[str], ncols: int = 2):
        cols = st.columns(ncols)
        for i, feat in enumerate(features):
            lo, hi, default, step = FEATURE_RANGES.get(feat, (0.0, 1.0, 0.5, 0.01))
            with cols[i % ncols]:
                input_row[feat] = st.slider(
                    label=feat,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    key=f"slider_{feat}",
                )

    tab_mean, tab_se, tab_worst = st.tabs(
        ["📐 Mean features", "📊 SE features", "⚠️ Worst features"]
    )
    with tab_mean:
        make_sliders(mean_feats)
    with tab_se:
        make_sliders(se_feats)
    with tab_worst:
        make_sliders(worst_feats)

    st.divider()
    col_btn, col_res = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

    if predict_clicked:
        # Ensure all features are present in insertion order
        ordered_row = {f: input_row.get(f, FEATURE_RANGES[f][2]) for f in expected_features}
        input_df = pd.DataFrame([ordered_row], columns=expected_features)

        try:
            label, confidence = run_prediction_single(model, input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        readable = decode_label(label)
        is_malignant = label == "M"

        with col_res:
            if is_malignant:
                st.error(f"**Prediction: {readable}**")
            else:
                st.success(f"**Prediction: {readable}**")

            st.metric("Model Confidence", f"{confidence:.2%}")

        st.dataframe(input_df, use_container_width=True)

# ===========================================================================
# BATCH PREDICTION
# ===========================================================================
else:
    st.subheader("Batch Prediction from CSV")
    st.write(
        "Upload a CSV with one row per sample and the required feature columns. "
        f"Expected columns: `{expected_features}`"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            st.stop()

        missing = [col for col in expected_features if col not in data.columns]
        extra   = [col for col in data.columns if col not in expected_features]

        if missing:
            st.error(f"Missing required columns ({len(missing)}): {missing}")
            st.stop()

        if extra:
            st.warning(
                f"Extra columns detected ({len(extra)}) — ignored during prediction: {extra}"
            )

        feature_data = data[expected_features].copy()

        try:
            predictions = run_batch_prediction(model, feature_data)
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")
            st.stop()

        n_malignant = (predictions["prediction_label"] == "M").sum()
        n_benign    = (predictions["prediction_label"] == "B").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total samples", len(predictions))
        col2.metric("🟢 Benign",     n_benign)
        col3.metric("🔴 Malignant",  n_malignant)

        st.success(f"Generated predictions for {len(predictions)} rows.")
        st.dataframe(predictions, use_container_width=True)

        st.download_button(
            "⬇️ Download predictions CSV",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
